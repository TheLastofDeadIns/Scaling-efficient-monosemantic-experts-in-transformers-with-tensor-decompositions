"""NumPy verification of the rearranged forward passes in ``src/model.py``.

Every factorised layer computes its output without ever materialising the
N-expert weight tensor.  That rearrangement is the single most dangerous part
of the implementation: a wrong contraction still trains and still drives the
loss down, it just computes a different function than the one described in the
report.

This file re-implements each fast forward with *exactly the same einsum
strings* as the torch code, computes the naive per-expert sum independently,
and asserts they agree.  It needs only NumPy, so it can be run anywhere.

    python tests/test_layers_numpy.py
"""

import numpy as np

RNG = np.random.default_rng(0)
TOL = 1e-9


def relu(z):
    return np.maximum(z, 0.0)


def make_routing(B, H, n, k):
    """Dense top-k routing weights, as produced by ProductKeyRouter."""
    def one(z):
        idx = np.argsort(-z, axis=-1)[..., :k]
        out = np.zeros_like(z)
        top = np.take_along_axis(z, idx, axis=-1)
        w = np.exp(top - top.max(-1, keepdims=True))
        w = w / w.sum(-1, keepdims=True)
        np.put_along_axis(out, idx, w, axis=-1)
        return out
    return one(RNG.normal(size=(B, H, n))), one(RNG.normal(size=(B, H, n)))


# --------------------------------------------------------------------------- #

def check_hd(B=5, H=3, n=6, m=4, D=11, O=7, k=2):
    x = RNG.normal(size=(B, D))
    g1, g2 = make_routing(B, H, n, k)
    U = RNG.normal(size=(n, m, D))
    V = RNG.normal(size=(n, O, m))
    b1 = RNG.normal(size=(n, m))
    b2 = RNG.normal(size=(n, O))

    h = relu(np.einsum("bd,nmd->bnm", x, U) + b1)
    t = np.einsum("bhi,bim->bhm", g1, h)
    u = np.einsum("bhj,bhm->bjm", g2, t)
    fast = np.einsum("bjm,jom->bo", u, V)
    s1 = g1.sum(-1)
    c = np.einsum("bhj,bh->bj", g2, s1)
    fast = fast + np.einsum("bj,jo->bo", c, b2)

    naive = np.zeros((B, O))
    for hh in range(H):
        for i in range(n):
            for j in range(n):
                w = g1[:, hh, i] * g2[:, hh, j]
                e = h[:, i, :] @ V[j].T + b2[j]
                naive += w[:, None] * e
    return np.abs(fast - naive).max()


def check_vd(B=5, H=3, n=6, m=4, D=11, O=7, k=2):
    x = RNG.normal(size=(B, D))
    g1, g2 = make_routing(B, H, n, k)
    m2 = m // 2
    d1 = (O + 1) // 2
    d2 = O - d1
    U1, U2 = RNG.normal(size=(n, m2, D)), RNG.normal(size=(n, m2, D))
    V11 = RNG.normal(size=(n, d1, m2))
    V12 = RNG.normal(size=(n, d1, m2))
    V21 = RNG.normal(size=(n, d2, m2))
    V22 = RNG.normal(size=(n, d2, m2))
    b11, b21 = RNG.normal(size=(n, m2)), RNG.normal(size=(n, m2))
    b12, b22 = RNG.normal(size=(n, d1)), RNG.normal(size=(n, d2))

    p1 = relu(np.einsum("bd,nmd->bnm", x, U1) + b11)
    p2 = relu(np.einsum("bd,nmd->bnm", x, U2) + b21)
    s1, s2 = g1.sum(-1), g2.sum(-1)
    A = np.einsum("bhi,bh->bi", g1, s2)
    Bc = np.einsum("bhj,bh->bj", g2, s1)

    x11 = np.einsum("bi,bim,iom->bo", A, p1, V11)
    x22 = np.einsum("bj,bjm,jom->bo", Bc, p2, V22)
    q2 = np.einsum("bhj,bjm->bhm", g2, p2)
    r1 = np.einsum("bhi,bhm->bim", g1, q2)
    x12 = np.einsum("bim,iom->bo", r1, V12)
    q1 = np.einsum("bhi,bim->bhm", g1, p1)
    r2 = np.einsum("bhj,bhm->bjm", g2, q1)
    x21 = np.einsum("bjm,jom->bo", r2, V21)
    x13 = np.einsum("bi,io->bo", A, b12)
    x23 = np.einsum("bj,jo->bo", Bc, b22)
    fast = np.concatenate([x11 + x12 + x13, x21 + x22 + x23], axis=-1)

    naive = np.zeros((B, O))
    for hh in range(H):
        for i in range(n):
            for j in range(n):
                w = (g1[:, hh, i] * g2[:, hh, j])[:, None]
                top = p1[:, i, :] @ V11[i].T + p2[:, j, :] @ V12[i].T + b12[i]
                bot = p1[:, i, :] @ V21[j].T + p2[:, j, :] @ V22[j].T + b22[j]
                naive += w * np.concatenate([top, bot], axis=-1)
    return np.abs(fast - naive).max()


def check_full(B=5, H=3, n=4, m=3, D=9, O=6, k=2):
    x = RNG.normal(size=(B, D))
    g1, g2 = make_routing(B, H, n, k)
    U = RNG.normal(size=(n, n, m, D))
    V = RNG.normal(size=(n, n, O, m))
    b1 = RNG.normal(size=(n, n, m))
    b2 = RNG.normal(size=(n, n, O))

    h = relu(np.einsum("bd,ijmd->bijm", x, U) + b1)
    gij = np.einsum("bhi,bhj->bij", g1, g2)
    fast = np.einsum("bij,bijm,ijom->bo", gij, h, V)
    fast = fast + np.einsum("bij,ijo->bo", gij, b2)

    naive = np.zeros((B, O))
    for hh in range(H):
        for i in range(n):
            for j in range(n):
                w = (g1[:, hh, i] * g2[:, hh, j])[:, None]
                naive += w * (h[:, i, j, :] @ V[i, j].T + b2[i, j])
    return np.abs(fast - naive).max()


def check_cp(B=5, H=3, n=6, D=11, O=7, R=5, k=2):
    x = RNG.normal(size=(B, D))
    g1, g2 = make_routing(B, H, n, k)
    Gout = RNG.normal(size=(R, O))
    Gin = RNG.normal(size=(R, D))
    Ga = RNG.normal(size=(R, n))
    Gb = RNG.normal(size=(R, n))

    px = np.einsum("bd,rd->br", x, Gin)
    pa = np.einsum("bhi,ri->bhr", g1, Ga)
    pb = np.einsum("bhj,rj->bhr", g2, Gb)
    fast = np.einsum("br,bhr,bhr,ro->bo", px, pa, pb, Gout)

    W = np.einsum("ro,rd,ri,rj->odij", Gout, Gin, Ga, Gb)
    naive = np.einsum("odij,bd,bhi,bhj->bo", W, x, g1, g2)
    return np.abs(fast - naive).max()


def check_tucker(B=5, H=3, n=6, D=11, O=7, R=4, Ra=3, k=2):
    x = RNG.normal(size=(B, D))
    g1, g2 = make_routing(B, H, n, k)
    core = RNG.normal(size=(R, R, Ra, Ra))
    Gout = RNG.normal(size=(O, R))
    Gin = RNG.normal(size=(D, R))
    Ga = RNG.normal(size=(n, Ra))
    Gb = RNG.normal(size=(n, Ra))

    px = np.einsum("bd,dq->bq", x, Gin)
    pa = np.einsum("bhi,ia->bha", g1, Ga)
    pb = np.einsum("bhj,jc->bhc", g2, Gb)
    cc = np.einsum("pqac,bq,bha,bhc->bp", core, px, pa, pb)
    fast = np.einsum("bp,op->bo", cc, Gout)

    W = np.einsum("pqac,op,dq,ia,jc->odij", core, Gout, Gin, Ga, Gb)
    naive = np.einsum("odij,bd,bhi,bhj->bo", W, x, g1, g2)
    return np.abs(fast - naive).max()


def check_tt(B=5, H=3, n=6, D=11, O=7, R=4, R1=2, k=2):
    x = RNG.normal(size=(B, D))
    g1, g2 = make_routing(B, H, n, k)
    G1 = RNG.normal(size=(R1, O, R))
    G2 = RNG.normal(size=(R, D, R))
    G3 = RNG.normal(size=(R, n, R))
    G4 = RNG.normal(size=(R, n, R1))

    f1 = np.einsum("bd,qdr->bqr", x, G2)
    f2 = np.einsum("bhi,rin->bhrn", g1, G3)
    f3 = np.einsum("bhj,njp->bhnp", g2, G4)
    fast = np.einsum("poq,bqr,bhrn,bhnp->bo", G1, f1, f2, f3)

    W = np.einsum("poq,qdr,rin,njp->odij", G1, G2, G3, G4)
    naive = np.einsum("odij,bd,bhi,bhj->bo", W, x, g1, g2)
    return np.abs(fast - naive).max()


CHECKS = {
    "moe_hd": check_hd,
    "moe_vd": check_vd,
    "moe_full": check_full,
    "moe_cp": check_cp,
    "moe_tucker": check_tucker,
    "moe_tt": check_tt,
}


def main():
    ok = True
    for name, fn in CHECKS.items():
        err = fn()
        status = "OK " if err < TOL else "FAIL"
        ok &= err < TOL
        print(f"{status} {name:12s} max|fast - naive| = {err:.3e}")
    print("\nall passed" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
