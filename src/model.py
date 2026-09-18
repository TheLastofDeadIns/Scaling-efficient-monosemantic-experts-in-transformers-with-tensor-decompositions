"""Architectures for the modular-arithmetic MoE study.

Seven layer types replace the single hidden layer of the reference MLP:

    mlp        dense two-layer MLP (baseline, no experts)
    moe_full   unfactorised MoE: N independent experts stored explicitly
    moe_hd     MONET horizontal decomposition (product-key composition)
    moe_vd     MONET vertical decomposition
    moe_cp     mu-MoE with CP-factorised expert tensor
    moe_tucker mu-MoE with Tucker-factorised expert tensor
    moe_tt     mu-MoE with Tensor-Train / Tensor-Ring factorised expert tensor

All MoE variants share one product-key router with ``n = sqrt(N)`` keys per
side and ``n_heads`` heads, so differences between them come from the expert
parameterisation alone.

Implementation notes that matter for the write-up
-------------------------------------------------
1.  Every factorised forward pass is computed *without materialising* the
    N-expert weight tensor.  The rearranged contractions are verified against
    the naive per-expert sum in ``tests/test_layers_numpy.py``.
2.  MONET estimates routing quantiles with BatchNorm to avoid a hardware-
    unfriendly top-k.  At N <= 4096 an exact top-k is cheap, so we use it and
    keep BatchNorm as an option (``router_bn``).  This is a deliberate
    deviation and is documented in the report.
3.  The auxiliary losses are computed on the *full* softmax, not on the
    top-k-masked routing weights, because the masked weights contain exact
    zeros and log(0) is undefined.
4.  mu-MoE layers are multilinear: the expert maps are linear in x, with no
    elementwise nonlinearity, whereas MONET experts apply sigma inside.  Set
    ``mu_nonlinear=True`` for a hybrid that inserts the activation on the
    input factor, used as an ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHS = (
    "mlp",
    "moe_full",
    "moe_hd",
    "moe_vd",
    "moe_cp",
    "moe_tucker",
    "moe_tt",
)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    arch: str = "moe_hd"
    d_in: int = 226
    d_out: int = 113

    # dense baseline
    hidden: int = 100

    # expert grid: N = n_keys ** 2 total composed experts
    n_keys: int = 16
    expert_dim: int = 8          # m, must be even for moe_vd
    n_heads: int = 4
    top_k: int = 4               # per side; top_k <= n_keys
    router_bn: bool = False
    router_split: str = "natural"  # "natural" | "interleave" | "random"

    # tensor ranks
    cp_rank: int = 128
    tucker_rank: int = 32        # shared across all four modes
    tt_rank: int = 16            # R2 = R3 = R4
    tr_rank: int = 1             # R1; 1 => Tensor-Train, >1 => Tensor-Ring
    comp_rank: int = 0           # 0 => dense core; r > 0 => rank-r expert-mode coupling

    # losses
    lambda_aux: float = 1e-3
    unif_mode: str = "token"     # "token" (MONET eq. 16) | "batch"

    activation: str = "relu"     # "relu" | "sqrelu" | "gelu"
    mu_nonlinear: bool = False

    seed: int = 0

    def n_experts(self) -> int:
        return self.n_keys ** 2


def _act_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "sqrelu":
        return lambda z: F.relu(z) ** 2
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unknown activation {name!r}")


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #

class ProductKeyRouter(nn.Module):
    """Two-sided product-key router.

    Returns dense weights g1, g2 of shape (B, H, n) that are zero outside the
    per-head top-k support and sum to one over that support, plus the full
    softmax probabilities used by the auxiliary losses.

    With the one-hot encoding of ``data.py`` and ``router_split="natural"``,
    the first half of x is onehot(a) and the second half is onehot(b), so the
    logits reduce to lookups into a learned (n, P) table.  ``interleave`` and
    ``random`` destroy that alignment and serve as the ablation that checks
    whether the model *finds* the factorisation or is simply handed it.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d_in, n, H = cfg.d_in, cfg.n_keys, cfg.n_heads
        half = d_in // 2
        self.d1 = half
        self.d2 = d_in - half

        self.register_buffer("perm", self._make_perm(cfg), persistent=True)

        self.key1 = nn.Parameter(torch.empty(H, n, self.d1))
        self.key2 = nn.Parameter(torch.empty(H, n, self.d2))
        nn.init.normal_(self.key1, std=self.d1 ** -0.5)
        nn.init.normal_(self.key2, std=self.d2 ** -0.5)

        self.bn = nn.BatchNorm1d(2 * H * n) if cfg.router_bn else None

    @staticmethod
    def _make_perm(cfg: ModelConfig) -> torch.Tensor:
        d = cfg.d_in
        if cfg.router_split == "natural":
            return torch.arange(d)
        if cfg.router_split == "interleave":
            # even coordinates to side 1, odd to side 2: mixes a and b bits
            idx = torch.arange(d)
            return torch.cat([idx[0::2], idx[1::2]])
        if cfg.router_split == "random":
            g = torch.Generator().manual_seed(cfg.seed + 12345)
            return torch.randperm(d, generator=g)
        raise ValueError(f"unknown router_split {cfg.router_split!r}")

    def forward(self, x: torch.Tensor):
        cfg = self.cfg
        xp = x[:, self.perm]
        x1, x2 = xp[:, : self.d1], xp[:, self.d1:]

        z1 = torch.einsum("bd,hnd->bhn", x1, self.key1)
        z2 = torch.einsum("bd,hnd->bhn", x2, self.key2)

        if self.bn is not None:
            B, H, n = z1.shape
            z = torch.cat([z1.reshape(B, H * n), z2.reshape(B, H * n)], dim=1)
            z = self.bn(z)
            z1 = z[:, : H * n].reshape(B, H, n)
            z2 = z[:, H * n:].reshape(B, H, n)

        p1 = torch.softmax(z1, dim=-1)
        p2 = torch.softmax(z2, dim=-1)

        g1 = self._sparsify(z1, cfg.top_k)
        g2 = self._sparsify(z2, cfg.top_k)
        return g1, g2, p1, p2

    @staticmethod
    def _sparsify(z: torch.Tensor, k: int) -> torch.Tensor:
        if k >= z.shape[-1]:
            return torch.softmax(z, dim=-1)
        topv, topi = torch.topk(z, k, dim=-1)
        w = torch.softmax(topv, dim=-1)
        out = torch.zeros_like(z)
        return out.scatter(-1, topi, w)


def router_aux_loss(p1: torch.Tensor, p2: torch.Tensor, mode: str = "token"):
    """MONET's uniformity + ambiguity losses (eqs. 16-17).

    ``token`` follows the paper literally: the log is taken per token and then
    averaged.  ``batch`` first averages the routing distribution over the batch
    and then measures its KL to uniform, which is the usual load-balancing
    form.  We report both in an ablation.
    """
    eps = 1e-9
    if mode == "token":
        l_unif = -(p1.clamp_min(eps).log().mean() + p2.clamp_min(eps).log().mean()) / 2.0
    elif mode == "batch":
        q1 = p1.mean(dim=0).clamp_min(eps)
        q2 = p2.mean(dim=0).clamp_min(eps)
        l_unif = -(q1.log().mean() + q2.log().mean()) / 2.0
    else:
        raise ValueError(f"unknown unif_mode {mode!r}")

    l_amb = ((1.0 - p1.max(dim=-1).values).mean()
             + (1.0 - p2.max(dim=-1).values).mean()) / 2.0
    return l_unif, l_amb


# --------------------------------------------------------------------------- #
# expert layers
# --------------------------------------------------------------------------- #

class _MoEBase(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = ProductKeyRouter(cfg)
        self.act = _act_fn(cfg.activation)

    def experts(self, x, g1, g2):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x):
        g1, g2, p1, p2 = self.router(x)
        out = self.experts(x, g1, g2)
        return out, (g1, g2, p1, p2)


def _kaiming(shape, fan_in, generator=None):
    t = torch.empty(*shape)
    nn.init.normal_(t, std=(2.0 / fan_in) ** 0.5)
    return t


class MoEFull(nn.Module):
    """Unfactorised control: N independent experts, each a small MLP.

    Memory and compute grow as O(N), which is exactly the scaling that the
    factorised variants avoid.  Only run this for small n_keys; it is a
    control, not a candidate.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = ProductKeyRouter(cfg)
        self.act = _act_fn(cfg.activation)
        n, m = cfg.n_keys, cfg.expert_dim
        self.U = nn.Parameter(_kaiming((n, n, m, cfg.d_in), cfg.d_in))
        self.V = nn.Parameter(_kaiming((n, n, cfg.d_out, m), m))
        self.b1 = nn.Parameter(torch.zeros(n, n, m))
        self.b2 = nn.Parameter(torch.zeros(n, n, cfg.d_out))

    def forward(self, x):
        g1, g2, p1, p2 = self.router(x)
        h = self.act(torch.einsum("bd,ijmd->bijm", x, self.U) + self.b1)
        gij = torch.einsum("bhi,bhj->bij", g1, g2)
        out = torch.einsum("bij,bijm,ijom->bo", gij, h, self.V)
        out = out + torch.einsum("bij,ijo->bo", gij, self.b2)
        return out, (g1, g2, p1, p2)


class MoEHD(_MoEBase):
    """MONET horizontal decomposition: E_ij(x) = V_j sigma(U_i x + b1_i) + b2_j."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        n, m = cfg.n_keys, cfg.expert_dim
        self.U = nn.Parameter(_kaiming((n, m, cfg.d_in), cfg.d_in))
        self.V = nn.Parameter(_kaiming((n, cfg.d_out, m), m))
        self.b1 = nn.Parameter(torch.zeros(n, m))
        self.b2 = nn.Parameter(torch.zeros(n, cfg.d_out))

    def experts(self, x, g1, g2):
        h = self.act(torch.einsum("bd,nmd->bnm", x, self.U) + self.b1)
        t = torch.einsum("bhi,bim->bhm", g1, h)       # collapse bottom experts
        u = torch.einsum("bhj,bhm->bjm", g2, t)       # distribute to top experts
        out = torch.einsum("bjm,jom->bo", u, self.V)
        s1 = g1.sum(dim=-1)                            # (B, H)
        c = torch.einsum("bhj,bh->bj", g2, s1)
        return out + torch.einsum("bj,jo->bo", c, self.b2)


class MoEVD(_MoEBase):
    """MONET vertical decomposition (eq. 13-15, expanded as in appendix A.1)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        n, m = cfg.n_keys, cfg.expert_dim
        if m % 2 != 0:
            raise ValueError("expert_dim must be even for moe_vd")
        m2 = m // 2
        d1 = (cfg.d_out + 1) // 2
        d2 = cfg.d_out - d1
        self.m2, self.dv1, self.dv2 = m2, d1, d2

        self.U1 = nn.Parameter(_kaiming((n, m2, cfg.d_in), cfg.d_in))
        self.U2 = nn.Parameter(_kaiming((n, m2, cfg.d_in), cfg.d_in))
        self.V11 = nn.Parameter(_kaiming((n, d1, m2), m2))
        self.V12 = nn.Parameter(_kaiming((n, d1, m2), m2))
        self.V21 = nn.Parameter(_kaiming((n, d2, m2), m2))
        self.V22 = nn.Parameter(_kaiming((n, d2, m2), m2))
        self.b11 = nn.Parameter(torch.zeros(n, m2))
        self.b21 = nn.Parameter(torch.zeros(n, m2))
        self.b12 = nn.Parameter(torch.zeros(n, d1))
        self.b22 = nn.Parameter(torch.zeros(n, d2))

    def experts(self, x, g1, g2):
        p1 = self.act(torch.einsum("bd,nmd->bnm", x, self.U1) + self.b11)
        p2 = self.act(torch.einsum("bd,nmd->bnm", x, self.U2) + self.b21)

        s1 = g1.sum(dim=-1)                                   # (B, H)
        s2 = g2.sum(dim=-1)                                   # (B, H)
        A = torch.einsum("bhi,bh->bi", g1, s2)                # weight on index i
        Bc = torch.einsum("bhj,bh->bj", g2, s1)               # weight on index j

        x11 = torch.einsum("bi,bim,iom->bo", A, p1, self.V11)
        x22 = torch.einsum("bj,bjm,jom->bo", Bc, p2, self.V22)

        q2 = torch.einsum("bhj,bjm->bhm", g2, p2)
        r1 = torch.einsum("bhi,bhm->bim", g1, q2)
        x12 = torch.einsum("bim,iom->bo", r1, self.V12)

        q1 = torch.einsum("bhi,bim->bhm", g1, p1)
        r2 = torch.einsum("bhj,bhm->bjm", g2, q1)
        x21 = torch.einsum("bjm,jom->bo", r2, self.V21)

        x13 = torch.einsum("bi,io->bo", A, self.b12)
        x23 = torch.einsum("bj,jo->bo", Bc, self.b22)

        return torch.cat([x11 + x12 + x13, x21 + x22 + x23], dim=-1)


class MoECP(_MoEBase):
    """mu-MoE with a CP-factorised weight tensor W in R^{O x I x n x n}."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        R, n = cfg.cp_rank, cfg.n_keys
        self.Gout = nn.Parameter(_kaiming((R, cfg.d_out), R))
        self.Gin = nn.Parameter(_kaiming((R, cfg.d_in), cfg.d_in))
        self.Ga = nn.Parameter(torch.randn(R, n) * 0.1 + 1.0)
        self.Gb = nn.Parameter(torch.randn(R, n) * 0.1 + 1.0)

    def experts(self, x, g1, g2):
        px = torch.einsum("bd,rd->br", x, self.Gin)
        if self.cfg.mu_nonlinear:
            px = self.act(px)
        pa = torch.einsum("bhi,ri->bhr", g1, self.Ga)
        pb = torch.einsum("bhj,rj->bhr", g2, self.Gb)
        return torch.einsum("br,bhr,bhr,ro->bo", px, pa, pb, self.Gout)


class MoETucker(_MoEBase):
    """mu-MoE with a Tucker-factorised weight tensor.

    ``cfg.comp_rank`` controls how the core couples the two expert modes.
    With ``comp_rank = 0`` the core is dense, which is the original layer.
    With ``comp_rank = r > 0`` the (a, c) slice of the core is constrained to
    rank r,

        core[p, q, a, c] = sum_s W[p, q, s] Ca[s, a] Cb[s, c],

    so r = 1 makes the dependence on the two routing vectors separable --- the
    rank-one composition that product-key routing implements --- and large r
    recovers the dense core.  The initialisation is matched so that the
    entries of the implied core have the same variance at every r.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        R, n = cfg.tucker_rank, cfg.n_keys
        Ra = min(R, n)
        self.Ra = Ra
        self.comp_rank = getattr(cfg, "comp_rank", 0)
        if self.comp_rank > 0:
            r = self.comp_rank
            self.W = nn.Parameter(
                torch.randn(R, R, r) * (R ** -1.5) * Ra / (r ** 0.5))
            self.Ca = nn.Parameter(torch.randn(r, Ra) * (Ra ** -0.5))
            self.Cb = nn.Parameter(torch.randn(r, Ra) * (Ra ** -0.5))
        else:
            self.core = nn.Parameter(torch.randn(R, R, Ra, Ra) * (R ** -1.5))
        self.Gout = nn.Parameter(_kaiming((cfg.d_out, R), R))
        self.Gin = nn.Parameter(_kaiming((cfg.d_in, R), cfg.d_in))
        self.Ga = nn.Parameter(torch.randn(n, Ra) * 0.1 + 1.0)
        self.Gb = nn.Parameter(torch.randn(n, Ra) * 0.1 + 1.0)

    def dense_core(self) -> torch.Tensor:
        """The (R, R, Ra, Ra) core the layer represents, for verification."""
        if self.comp_rank > 0:
            return torch.einsum("pqs,sa,sc->pqac", self.W, self.Ca, self.Cb)
        return self.core

    def experts(self, x, g1, g2):
        px = torch.einsum("bd,dq->bq", x, self.Gin)
        if self.cfg.mu_nonlinear:
            px = self.act(px)
        pa = torch.einsum("bhi,ia->bha", g1, self.Ga)
        pb = torch.einsum("bhj,jc->bhc", g2, self.Gb)
        if self.comp_rank > 0:
            ua = torch.einsum("sa,bha->bhs", self.Ca, pa)
            ub = torch.einsum("sc,bhc->bhs", self.Cb, pb)
            core = torch.einsum("pqs,bq,bhs,bhs->bp", self.W, px, ua, ub)
        else:
            core = torch.einsum("pqac,bq,bha,bhc->bp", self.core, px, pa, pb)
        return torch.einsum("bp,op->bo", core, self.Gout)


class MoETT(_MoEBase):
    """mu-MoE with a Tensor-Train (tr_rank=1) or Tensor-Ring factorisation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        R1, R = cfg.tr_rank, cfg.tt_rank
        n = cfg.n_keys
        self.G1 = nn.Parameter(torch.randn(R1, cfg.d_out, R) * (R ** -0.5))
        self.G2 = nn.Parameter(torch.randn(R, cfg.d_in, R) * (cfg.d_in ** -0.5))
        self.G3 = nn.Parameter(torch.randn(R, n, R) * (R ** -0.5))
        self.G4 = nn.Parameter(torch.randn(R, n, R1) * (R ** -0.5))

    def experts(self, x, g1, g2):
        f1 = torch.einsum("bd,qdr->bqr", x, self.G2)
        if self.cfg.mu_nonlinear:
            f1 = self.act(f1)
        f2 = torch.einsum("bhi,rin->bhrn", g1, self.G3)
        f3 = torch.einsum("bhj,njp->bhnp", g2, self.G4)
        return torch.einsum("poq,bqr,bhrn,bhnp->bo", self.G1, f1, f2, f3)


class DenseMLP(nn.Module):
    """Reference two-layer MLP with a single hidden layer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.act = _act_fn(cfg.activation)
        self.fc1 = nn.Linear(cfg.d_in, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))), None


# --------------------------------------------------------------------------- #
# top-level model
# --------------------------------------------------------------------------- #

_REGISTRY = {
    "mlp": DenseMLP,
    "moe_full": MoEFull,
    "moe_hd": MoEHD,
    "moe_vd": MoEVD,
    "moe_cp": MoECP,
    "moe_tucker": MoETucker,
    "moe_tt": MoETT,
}


class ModAddModel(nn.Module):
    """Wraps a layer and exposes logits plus the auxiliary loss."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.arch not in _REGISTRY:
            raise ValueError(f"unknown arch {cfg.arch!r}, expected one of {ARCHS}")
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.layer = _REGISTRY[cfg.arch](cfg)
        self._last_route = None

    def forward(self, x):
        logits, route = self.layer(x)
        self._last_route = route
        return logits

    def aux_loss(self):
        if self._last_route is None:
            return torch.zeros((), device=next(self.parameters()).device)
        _, _, p1, p2 = self._last_route
        l_unif, l_amb = router_aux_loss(p1, p2, self.cfg.unif_mode)
        return self.cfg.lambda_aux * (l_unif + l_amb)

    def routing(self):
        """Return (g1, g2) for the last forward pass, or None for the MLP."""
        if self._last_route is None:
            return None
        return self._last_route[0], self._last_route[1]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_params(cfg: ModelConfig) -> int:
    """Parameter count without building the model on GPU."""
    return ModAddModel(cfg).n_params()


def match_budget(cfg: ModelConfig, target: int, knob: str, lo: int, hi: int) -> int:
    """Smallest value of ``knob`` in [lo, hi] whose parameter count >= target.

    Used to build parameter-matched comparisons across architectures.
    """
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = ModelConfig(**{**cfg.__dict__, knob: mid})
        if count_params(trial) >= target:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    return best
