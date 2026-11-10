"""
3D mass-spring model of the sponge (the model used for the results).

The sponge is a 2 (x, width, seen by the camera) x 9.5 (y, height) x 7.4
(z, depth, toward the camera) cm block, discretised as a grid of point masses:

    nx x ny x nz = 3 x 11 x 4 nodes

joined by springs along every cell edge (structural), every face diagonal and
every body diagonal (shear). Every spring has a dashpot. Every cell is also
split into 6 tetrahedra whose SIGNED VOLUME resists change (the 3D version of
the 2D area term): springs alone only see lengths, so a pure spring grid can
turn a cell inside out and stay stuck in that wrong shape; real foam cannot.

The bottom layer of nodes is pinned (taped to the table). The marker dots are
on the front face (z = +depth/2, toward the camera); the tracked video gives
their x, y.

Loads and contacts (same as the 2D development model, sim.py):
  * gravity;
  * the phone: a rigid plate y = c(t) - a(t) * x (uniform along the depth),
    also moving sideways by plate_x(t), all taken from the tracking. One-sided
    (it can push, never pull) with Coulomb friction mu; normal taken as
    vertical (phone tilt <= ~12 degrees);
  * the table y = 0: one-sided, sticking.
Damping is Rayleigh damping: spring dashpots (zeta, resist the rate of
deformation) + mass-proportional damping (alpha, resists the speed of motion).

Units: cm, s. Positions are all the video gives (never forces), so the model
is written PER UNIT MASS (total mass 1): stiffnesses are stiffness/mass.

Parameters: sim.Params (k_axial, k_shear, zeta, mu, alpha).
"""
import numpy as np
from numba import njit

from sim import G, Params  # noqa: F401  (Params re-exported for convenience)


class SpongeModel3D:
    def __init__(self, width=2.0, height=9.5, depth=7.4, nx=3, ny=11, nz=4, bow=None):
        """bow: optional (ys, us) - measured sideways offset us [cm] of the
        sponge's centre line at heights ys [cm] before contact (its small
        initial bow/lean). Applied through the whole depth."""
        self.width, self.height, self.depth = width, height, depth
        self.nx, self.ny, self.nz = nx, ny, nz
        self.bow = bow
        xs = np.linspace(-width / 2, width / 2, nx)
        ys = np.linspace(0.0, height, ny)
        zs = np.linspace(-depth / 2, depth / 2, nz)
        self.dx, self.dy, self.dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
        idx = lambda i, j, k: (k * ny + j) * nx + i
        self.idx = idx
        rest = np.zeros((nx * ny * nz, 3))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    rest[idx(i, j, k)] = (xs[i], ys[j], zs[k])
        rest[:, 0] += self.bow_offset(rest[:, 1])
        self.rest = rest
        self.n = len(rest)

        # ---- springs ----
        axial, shear = set(), set()

        def add(s, a, b):
            s.add((min(a, b), max(a, b)))

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    a = idx(i, j, k)
                    for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        if i + di < nx and j + dj < ny and k + dk < nz:
                            add(axial, a, idx(i + di, j + dj, k + dk))
                    # face and body diagonals of the cell starting at (i, j, k)
                    if i + 1 < nx and j + 1 < ny and k + 1 < nz:
                        c = [[[idx(i + p, j + q, k + r) for r in (0, 1)] for q in (0, 1)] for p in (0, 1)]
                        for p in (0, 1):     # faces x = const
                            add(shear, c[p][0][0], c[p][1][1]); add(shear, c[p][1][0], c[p][0][1])
                        for q in (0, 1):     # faces y = const
                            add(shear, c[0][q][0], c[1][q][1]); add(shear, c[1][q][0], c[0][q][1])
                        for r in (0, 1):     # faces z = const
                            add(shear, c[0][0][r], c[1][1][r]); add(shear, c[1][0][r], c[0][1][r])
                        add(shear, c[0][0][0], c[1][1][1]); add(shear, c[1][0][0], c[0][1][1])
                        add(shear, c[0][1][0], c[1][0][1]); add(shear, c[0][0][1], c[1][1][0])
        axial, shear = sorted(axial), sorted(shear - axial)
        self.springs = np.array(axial + shear)
        self.is_shear = np.r_[np.zeros(len(axial), bool), np.ones(len(shear), bool)]
        d = rest[self.springs[:, 1]] - rest[self.springs[:, 0]]
        self.L0 = np.linalg.norm(d, axis=1)

        # ---- tetrahedra (6 per cell, Kuhn split), positively oriented ----
        tets = []
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    v = [idx(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in range(8)]
                    for path in ((1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1)):
                        b1 = path[0]
                        b2 = b1 | path[1]
                        tets.append((v[0], v[b1], v[b2], v[7]))
        tets = np.array(tets)
        vol = self._volumes(rest, tets)
        flip = vol < 0
        tets[flip, 2], tets[flip, 3] = tets[flip, 3], tets[flip, 2].copy()
        self.tets = tets
        self.V0 = self._volumes(rest, tets)

        self.fixed = rest[:, 1] <= 1e-9            # bottom layer taped down
        self.m_node = 1.0 / self.n
        self.front = [idx(i, j, nz - 1) for j in range(ny) for i in range(nx)]

    @staticmethod
    def _volumes(x, tets):
        a, b, c, d = (x[tets[:, q]] for q in range(4))
        return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0

    def bow_offset(self, y):
        y = np.asarray(y, float)
        if self.bow is None:
            return np.zeros_like(y)
        ys, us = self.bow
        out = np.interp(y, ys, us)
        hi = y > ys[-1]
        if np.any(hi):
            slope = (us[-1] - us[-2]) / (ys[-1] - ys[-2])
            out[hi] = us[-1] + slope * (y[hi] - ys[-1])
        return out

    # ---------- dots: material points on the front face ----------------------
    def embed(self, pts_rest):
        """Map tracked dot positions (x, y in the first frame) to front-face
        cells + bilinear weights."""
        ids, ws = [], []
        k = self.nz - 1
        for x, y in pts_rest:
            x = x - self.bow_offset(np.array([y]))[0]
            u = np.clip((x + self.width / 2) / self.dx, 0, self.nx - 1 - 1e-9)
            v = np.clip(y / self.dy, 0, self.ny - 1 - 1e-9)
            i, j = int(u), int(v)
            fu, fv = u - i, v - j
            ids.append([self.idx(i, j, k), self.idx(i + 1, j, k), self.idx(i, j + 1, k), self.idx(i + 1, j + 1, k)])
            ws.append([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv])
        return np.array(ids), np.array(ws)

    @staticmethod
    def interp(x, emb):
        ids, w = emb
        return (x[ids][..., :2] * w[..., None]).sum(1)     # x, y of the dots

    # ---------- simulation ---------------------------------------------------
    def simulate_clip(self, p, clip, record_pts=None):
        return self.simulate(p, clip.t, clip.plate_c, clip.plate_a, clip.plate_x, record_pts)

    def pluck(self, p, duration=0.4, fps=240.0, speed=15.0):
        """Virtual pluck: sideways velocity growing toward the top (first
        bending mode shape), no phone. Returns (t, sideways x of the top
        front dots)."""
        t = np.arange(int(duration * fps)) / fps
        v0 = np.zeros_like(self.rest)
        v0[:, 0] = speed * (self.rest[:, 1] / self.height) ** 2
        top = self.embed(np.array([[-0.65, 8.8], [0.65, 8.8]]))
        _, P = self.simulate(p, t, np.full(len(t), 1e3), np.zeros(len(t)), None, top, v0=v0)
        return t, P[:, :, 0].mean(1)

    def simulate(self, p, t, plate_c, plate_a, plate_x=None, record_pts=None, v0=None):
        a_ = np.nan_to_num(plate_a, nan=0.0)
        c_ = np.where(np.isfinite(plate_c), plate_c, 1e3)
        x_ = np.zeros(len(t)) if plate_x is None else np.nan_to_num(plate_x, nan=0.0)
        m = self.m_node
        k = np.where(self.is_shear, p.k_shear, p.k_axial)
        kc = k / self.L0
        c_d = p.zeta * 2.0 * np.sqrt(kc * m)
        # volume term: force = -(kV*(V-V0) + cV*dV/dt)/V0 * dV/dx, |dV/dx|^2 ~ h^4/36
        h = (self.dx * self.dy * self.dz) ** (1 / 3)
        g2 = h ** 4 / 36.0
        kV = p.k_axial
        k_eff_V = kV * g2 / self.V0.min()
        cV = p.zeta * 2.0 * np.sqrt(k_eff_V * m) * self.V0.min() / g2
        frame_dt = float(np.median(np.diff(t)))
        nspr = np.bincount(self.springs.ravel(), minlength=self.n).max()
        w_max = np.sqrt(4.0 * (nspr * kc.max() + 4 * k_eff_V) / m)
        c_node = nspr * c_d.max() + 4 * cV * g2 / self.V0.min() + p.alpha * m
        dt_max = min(0.9 / w_max, 0.5 * m / c_node)
        substeps = int(np.ceil(frame_dt / dt_max))
        self.last_substeps = substeps
        dt = frame_dt / substeps
        v_init = np.zeros_like(self.rest) if v0 is None else v0.astype(np.float64)
        X = _run3d(self.rest.copy(), v_init, self.springs, kc, c_d, self.L0, self.tets, self.V0,
                   kV, cV, ~self.fixed, c_, a_, x_, p.mu, p.alpha, substeps, dt, frame_dt, m, G)
        P = None
        if record_pts is not None:
            P = np.stack([self.interp(X[f], record_pts) for f in range(len(t))])
        return X, P


@njit(cache=True)
def _run3d(x, v, springs, kc, c_d, L0, tets, V0, kV, cV, free,
           plate_c, plate_a, plate_x, mu, alpha, substeps, dt, frame_dt, m, g):
    """Time stepping (semi-implicit Euler), compiled with numba."""
    n = x.shape[0]
    T = plate_c.shape[0]
    F = np.zeros_like(x)
    on_p = np.zeros(n, np.bool_)
    anc_t = np.full((n, 2), np.nan)          # x, z anchor on the table
    X = np.empty((T, n, 3))
    for f in range(T):
        f1 = min(f + 1, T - 1)
        c0, c1 = plate_c[f], plate_c[f1]
        a0, a1 = plate_a[f], plate_a[f1]
        vx_p = (plate_x[f1] - plate_x[f]) / frame_dt
        for s in range(substeps):
            al = s / substeps
            pc = c0 + al * (c1 - c0)
            pa = a0 + al * (a1 - a0)
            F[:, :] = 0.0
            # ---- springs + dashpots ----
            for e in range(springs.shape[0]):
                i, j = springs[e, 0], springs[e, 1]
                d0 = x[j, 0] - x[i, 0]
                d1 = x[j, 1] - x[i, 1]
                d2 = x[j, 2] - x[i, 2]
                L = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + 1e-12
                u0, u1, u2 = d0 / L, d1 / L, d2 / L
                rv = (v[j, 0] - v[i, 0]) * u0 + (v[j, 1] - v[i, 1]) * u1 + (v[j, 2] - v[i, 2]) * u2
                fm = kc[e] * (L - L0[e]) + c_d[e] * rv
                F[i, 0] += fm * u0
                F[i, 1] += fm * u1
                F[i, 2] += fm * u2
                F[j, 0] -= fm * u0
                F[j, 1] -= fm * u1
                F[j, 2] -= fm * u2
            # ---- signed-volume (anti-inversion) forces (scalar code: fast) ----
            for e in range(tets.shape[0]):
                a, b, c, d = tets[e, 0], tets[e, 1], tets[e, 2], tets[e, 3]
                e1x, e1y, e1z = x[b, 0] - x[a, 0], x[b, 1] - x[a, 1], x[b, 2] - x[a, 2]
                e2x, e2y, e2z = x[c, 0] - x[a, 0], x[c, 1] - x[a, 1], x[c, 2] - x[a, 2]
                e3x, e3y, e3z = x[d, 0] - x[a, 0], x[d, 1] - x[a, 1], x[d, 2] - x[a, 2]
                # gradients of V = e1 . (e2 x e3) / 6 w.r.t. b, c, d (a = minus their sum)
                gbx = (e2y * e3z - e2z * e3y) / 6.0
                gby = (e2z * e3x - e2x * e3z) / 6.0
                gbz = (e2x * e3y - e2y * e3x) / 6.0
                gcx = (e3y * e1z - e3z * e1y) / 6.0
                gcy = (e3z * e1x - e3x * e1z) / 6.0
                gcz = (e3x * e1y - e3y * e1x) / 6.0
                gdx = (e1y * e2z - e1z * e2y) / 6.0
                gdy = (e1z * e2x - e1x * e2z) / 6.0
                gdz = (e1x * e2y - e1y * e2x) / 6.0
                gax, gay, gaz = -(gbx + gcx + gdx), -(gby + gcy + gdy), -(gbz + gcz + gdz)
                V = e1x * gbx + e1y * gby + e1z * gbz
                Vd = (gax * v[a, 0] + gay * v[a, 1] + gaz * v[a, 2]
                      + gbx * v[b, 0] + gby * v[b, 1] + gbz * v[b, 2]
                      + gcx * v[c, 0] + gcy * v[c, 1] + gcz * v[c, 2]
                      + gdx * v[d, 0] + gdy * v[d, 1] + gdz * v[d, 2])
                s_ = -(kV * (V - V0[e]) + cV * Vd) / V0[e]
                F[a, 0] += s_ * gax
                F[a, 1] += s_ * gay
                F[a, 2] += s_ * gaz
                F[b, 0] += s_ * gbx
                F[b, 1] += s_ * gby
                F[b, 2] += s_ * gbz
                F[c, 0] += s_ * gcx
                F[c, 1] += s_ * gcy
                F[c, 2] += s_ * gcz
                F[d, 0] += s_ * gdx
                F[d, 1] += s_ * gdy
                F[d, 2] += s_ * gdz
            # ---- integrate free nodes, with contacts ----
            for i in range(n):
                if not free[i]:
                    continue
                for q in range(3):
                    F[i, q] -= m * alpha * v[i, q]          # mass-proportional damping
                # sideways (x) motion: Coulomb friction while touching the phone
                if on_p[i]:
                    fn = F[i, 1] - m * g
                    if fn < 0.0:
                        fn = 0.0
                    ft = F[i, 0]
                    vrel = v[i, 0] - vx_p
                    if abs(vrel) < 1e-6 and abs(ft) <= mu * fn:
                        v[i, 0] = vx_p
                    else:
                        sgn = np.sign(vrel) if abs(vrel) >= 1e-6 else np.sign(ft)
                        v[i, 0] += dt * (ft - sgn * mu * fn) / m
                        if np.sign(v[i, 0] - vx_p) == -sgn:
                            v[i, 0] = vx_p
                    # depth (z) direction: phone does not move along z; same friction
                    fz = F[i, 2]
                    if abs(v[i, 2]) < 1e-6 and abs(fz) <= mu * fn:
                        v[i, 2] = 0.0
                    else:
                        sz = np.sign(v[i, 2]) if abs(v[i, 2]) >= 1e-6 else np.sign(fz)
                        v[i, 2] += dt * (fz - sz * mu * fn) / m
                        if np.sign(v[i, 2]) == -sz:
                            v[i, 2] = 0.0
                else:
                    v[i, 0] += dt * F[i, 0] / m
                    v[i, 2] += dt * F[i, 2] / m
                v[i, 1] += dt * (F[i, 1] / m - g)
                for q in range(3):
                    x[i, q] += dt * v[i, q]
                # phone: solid above y = pc - pa*x (one-sided)
                yp = pc - pa * x[i, 0]
                if x[i, 1] > yp:
                    x[i, 1] = yp
                    vy = ((c1 - c0) - (a1 - a0) * x[i, 0]) / frame_dt
                    if v[i, 1] > vy:
                        v[i, 1] = vy
                    on_p[i] = True
                else:
                    on_p[i] = False
                # table: solid below y = 0 (sticking)
                if x[i, 1] < 0.0:
                    if np.isnan(anc_t[i, 0]):
                        anc_t[i, 0] = x[i, 0]
                        anc_t[i, 1] = x[i, 2]
                    x[i, 0] = anc_t[i, 0]
                    x[i, 2] = anc_t[i, 1]
                    x[i, 1] = 0.0
                    v[i, 0] = 0.0
                    v[i, 2] = 0.0
                    if v[i, 1] < 0.0:
                        v[i, 1] = 0.0
                else:
                    anc_t[i, 0] = np.nan
        X[f] = x
    return X
