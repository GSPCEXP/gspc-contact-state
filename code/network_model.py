import csv
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
K_CU = 401.0
PHI = {"dense": 0.27734, "porous": 0.3284}
BEDS = ("dense", "porous")
PACKINGS = (1, 2, 3, 4, 5)


def rows(name):
    with open(os.path.join(DATA, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_network(bed, packing, core):
    p = rows("networks/%s_p%d_particles.csv" % (bed, packing))
    c = rows("networks/%s_p%d_contacts.csv" % (bed, packing))
    z = np.array([float(r["z_um"]) for r in p]) * 1e-6
    i = np.array([int(r["i"]) for r in c])
    j = np.array([int(r["j"]) for r in c])
    a = np.array([float(r["a_um"]) for r in c]) * 1e-6
    k = next(r for r in core if r["bed"] == bed and int(r["packing"]) == packing)
    area = (float(k["x_hi_um"]) - float(k["x_lo_um"])) * (float(k["y_hi_um"]) - float(k["y_lo_um"])) * 1e-12
    t = 0.1 * (z.max() - z.min())
    hot = z < z.min() + t
    cold = z > z.max() - t
    return dict(z=z, i=i, j=j, a=a, area=area, hot=hot, cold=cold, length=z[cold].min() - z[hot].max())


def conductivity(n, g):
    i, j, hot, cold = n["i"], n["j"], n["hot"], n["cold"]
    m = len(hot)
    e = g > 0
    i, j, g = i[e], j[e], g[e]
    ncomp, lab = connected_components(sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(m, m)).tocsr(), directed=False)
    plate = np.zeros(ncomp, bool)
    plate[np.unique(lab[hot | cold])] = True
    free = plate[lab] & ~hot & ~cold
    fid = -np.ones(m, int)
    fid[free] = np.arange(free.sum())
    v = np.full(m, np.nan)
    v[hot] = 1.0
    v[cold] = 0.0
    diag = np.zeros(free.sum())
    rhs = np.zeros(free.sum())
    r, c, x = [], [], []
    for u, w in ((i, j), (j, i)):
        fu, fw = fid[u], fid[w]
        a = fu >= 0
        np.add.at(diag, fu[a], g[a])
        b = a & (fw >= 0)
        r.append(fu[b])
        c.append(fw[b])
        x.append(-g[b])
        d = a & (hot | cold)[w]
        np.add.at(rhs, fu[d], g[d] * v[w[d]])
    r.append(np.arange(len(diag)))
    c.append(np.arange(len(diag)))
    x.append(diag)
    lap = sp.csr_matrix((np.concatenate(x), (np.concatenate(r), np.concatenate(c))), shape=(len(diag), len(diag)))
    v[free] = spsolve(lap.tocsc(), rhs)
    vi, vj = v[i], v[j]
    ok = ~np.isnan(vi) & ~np.isnan(vj)
    a = ok & hot[i] & ~hot[j]
    b = ok & hot[j] & ~hot[i]
    current = np.sum(g[a] * (vi[a] - vj[a])) + np.sum(g[b] * (vj[b] - vi[b]))
    return current * n["length"] / n["area"]


def measured():
    pel = {r["pellet"]: float(r["porosity"]) for r in rows("pellets.csv")}
    shots = rows("lfa_shots.csv")
    out = {}
    for state, stage in (("pressed", "as pressed"), ("heated", "after cooling")):
        phi, lnk = [], []
        for p, f in pel.items():
            k = [float(r["k_W_mK"]) for r in shots if r["pellet"] == p and r["stage"] == stage]
            phi.append(f)
            lnk.append(np.log(np.mean(k)))
        coef = np.polyfit(phi, lnk, 1)
        out[state] = {b: float(np.exp(np.polyval(coef, PHI[b]))) for b in BEDS}
    return out


def main():
    core = rows("networks/cores.csv")
    meas = measured()
    for b in BEDS:
        kc = []
        for p in PACKINGS:
            n = load_network(b, p, core)
            kc.append(conductivity(n, 2 * K_CU * n["a"]))
        kc = np.array(kc)
        print("%s bed: kc %.1f +- %.1f W/mK, k/kc %.3f as pressed, %.3f after heating" % (
            b, kc.mean(), kc.std(ddof=1) / np.sqrt(len(PACKINGS)), meas["pressed"][b] / kc.mean(), meas["heated"][b] / kc.mean()))


if __name__ == "__main__":
    main()
