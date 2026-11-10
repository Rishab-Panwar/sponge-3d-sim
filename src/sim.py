"""
2D mass-spring model of the sponge column (side view, plane strain).

The sponge is a 9.5 x 2 cm rectangle (x 7.4 cm deep, uniform along the depth,
so a 2D side-view model is reasonable). It is discretised as a grid of point
masses joined by springs:

    o---o---o      structural springs (-- and |) resist stretch / squash
    | X | X |      shear springs (X)            resist the cells slanting
    o---o---o      bending emerges from the grid: one side squashes while the
    | X | X |      other stretches, like a real beam, so it can also buckle
    o---o---o
    ^^^^^^^^^      bottom row pinned (taped to the table)

Every spring also carries a dashpot (damping on its rate of stretch).

Area term: each grid cell is split into two triangles, and every triangle
resists changes of its signed area (with the same stiffness as the axial
springs, and the same damping ratio). Springs alone only see lengths, so a
pure spring grid can fold a cell inside out and stay stuck in that wrong
shape; real foam cannot. The signed-area term forbids this inversion.

Driving input: the phone, a rigid plate y = c(t) - a(t) * x that also moves
sideways by plate_x(t), all taken directly from the tracking. It is a
one-sided contact (it can push, never pull) with Coulomb friction: a touching
node moves with the plate while the sideways spring force on it is below
mu * (normal force), and slides otherwise. It is released as soon as the
plate moves away faster than the node follows. The normal is taken as
vertical (the phone tilts by at most ~12 degrees). The table (y = 0) is a
sticking one-sided contact (rarely touched apart from the taped base).

Units: cm, s. The video gives positions only, never forces, so the model is
written PER UNIT MASS: the total mass is 1, and the fitted stiffnesses are
"stiffness / mass" (1/s^2 scale). Multiply by the real mass to get N/m.

Parameters (all positive):
    k_axial   stiffness of the structural springs   [cm/s^2 per unit strain]
    k_shear   stiffness of the shear springs         [same units]
    zeta      material damping: dashpot on every spring, c = zeta * 2*sqrt(k * m_node)
              (resists the RATE OF DEFORMATION)
    mu        friction coefficient between phone and foam
    alpha     mass-proportional damping [1/s]: force -alpha * m * v on every node
              (resists the SPEED OF MOTION, e.g. air pumped through the pores)
    Together zeta and alpha are classic Rayleigh damping. alpha mainly damps
    the slow sideways sway; zeta mainly damps fast squashing modes.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

G = 981.0  # cm/s^2


@dataclass
class Params:
    k_axial: float = 2.0e5
    k_shear: float = 5.0e4
    zeta: float = 0.05
    mu: float = 1.0
    alpha: float = 0.0
    k_base: float = 0.0     # 0 = bottom row rigidly pinned (used). >0 = compliant base:
                            # an experiment, tested and rejected (README), kept off.

    def as_array(self):
        return np.array([self.k_axial, self.k_shear, self.zeta, self.mu, self.alpha, self.k_base])

    @staticmethod
    def from_array(v):
        return Params(*[float(x) for x in v])


class SpongeModel:
    def __init__(self, width=2.0, height=9.5, nx=3, ny=11, bow=None):
        """bow: optional (ys, us) - measured sideways offset us [cm] of the
        sponge's centre line at heights ys [cm] before contact. The rest
        (stress-free) shape is the straight rectangle shifted by this offset,
        i.e. the sponge's small initial bow/lean (the "imperfection" that
        decides which way it buckles)."""
        self.bow = bow
        self.width, self.height, self.nx, self.ny = width, height, nx, ny
        xs = np.linspace(-width / 2, width / 2, nx)
        ys = np.linspace(0.0, height, ny)
        X, Y = np.meshgrid(xs, ys)                 # row j = height level j
        self.rest = np.stack([X.ravel(), Y.ravel()], 1)
        self.rest[:, 0] += self.bow_offset(self.rest[:, 1])
        self.n = nx * ny
        idx = lambda i, j: j * nx + i
        axial, shear = [], []
        for j in range(ny):
            for i in range(nx):
                if i + 1 < nx:
                    axial.append((idx(i, j), idx(i + 1, j)))
                if j + 1 < ny:
                    axial.append((idx(i, j), idx(i, j + 1)))
                if i + 1 < nx and j + 1 < ny:
                    shear.append((idx(i, j), idx(i + 1, j + 1)))
                    shear.append((idx(i + 1, j), idx(i, j + 1)))
        self.springs = np.array(axial + shear)
        self.is_shear = np.r_[np.zeros(len(axial), bool), np.ones(len(shear), bool)]
        self.L0 = np.linalg.norm(self.rest[self.springs[:, 1]] - self.rest[self.springs[:, 0]], axis=1)
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):           # counter-clockwise triangles
                a, b = idx(i, j), idx(i + 1, j)
                c, d = idx(i + 1, j + 1), idx(i, j + 1)
                tris += [(a, b, c), (a, c, d)]
        self.tris = np.array(tris)
        pa, pb, pc = (self.rest[self.tris[:, k]] for k in range(3))
        self.A0 = 0.5 * ((pb - pa)[:, 0] * (pc - pa)[:, 1] - (pb - pa)[:, 1] * (pc - pa)[:, 0])
        self.fixed = np.zeros(self.n, bool)
        self.fixed[:nx] = True                     # bottom row taped down
        self.m_node = 1.0 / self.n                 # total mass = 1 (per-unit-mass model)
        self.dx, self.dy = width / (nx - 1), height / (ny - 1)

    # ---------- mapping dots (material points) to the deformed grid --------
    def bow_offset(self, y):
        """Sideways offset of the rest shape at height y (linear extrapolation
        beyond the measured heights)."""
        if self.bow is None:
            return np.zeros_like(y, dtype=float)
        ys, us = self.bow
        out = np.interp(y, ys, us)
        hi = y > ys[-1]
        if hi.any():
            slope = (us[-1] - us[-2]) / (ys[-1] - ys[-2])
            out[hi] = us[-1] + slope * (y[hi] - ys[-1])
        return out

    def embed(self, pts_rest):
        """For each material point (rest position), find its grid cell and
        bilinear weights, so deformed positions can be interpolated."""
        out = []
        for x, y in pts_rest:
            x = x - self.bow_offset(np.array([y]))[0]     # back to the straight frame
            u = np.clip((x + self.width / 2) / self.dx, 0, self.nx - 1 - 1e-9)
            v = np.clip(y / self.dy, 0, self.ny - 1 - 1e-9)
            i, j = int(u), int(v)
            fu, fv = u - i, v - j
            ids = [j * self.nx + i, j * self.nx + i + 1,
                   (j + 1) * self.nx + i, (j + 1) * self.nx + i + 1]
            w = [(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]
            out.append((ids, w))
        ids = np.array([o[0] for o in out])
        w = np.array([o[1] for o in out])
        return ids, w

    @staticmethod
    def interp(x, emb):
        ids, w = emb
        return (x[ids] * w[..., None]).sum(1)

    # ---------- dynamics ----------------------------------------------------
    def simulate_clip(self, p, clip, record_pts=None):
        return self.simulate(p, clip.t, clip.plate_c, clip.plate_a, clip.plate_x, record_pts)

    def pluck(self, p, duration=0.4, fps=240.0, speed=15.0):
        """Virtual pluck test: start straight, give the column a sideways
        velocity that grows towards the top (first bending mode shape), no
        phone, and record the top row's sideways position. Returns (t, x_top)."""
        t = np.arange(int(duration * fps)) / fps
        far = np.full(len(t), 1e3)
        v0 = np.zeros_like(self.rest)
        v0[:, 0] = speed * (self.rest[:, 1] / self.height) ** 2
        top = self.embed(np.array([[-0.65, 8.8], [0.65, 8.8]]))
        _, P = self.simulate(p, t, far, np.zeros(len(t)), None, top, v0=v0)
        return t, P[:, :, 0].mean(1)

    def simulate(self, p, t, plate_c, plate_a, plate_x=None, record_pts=None, substeps=None,
                 v0=None):
        """Simulate over the frame times t, driven by the plate trajectory.

        Returns positions at every frame: grid (T, n, 2) and, if record_pts
        (an embedding) is given, those material points (T, N, 2)."""
        a_ = np.nan_to_num(plate_a, nan=0.0)
        c_ = np.where(np.isfinite(plate_c), plate_c, 1e3)     # invisible = far away
        x_ = np.zeros(len(t)) if plate_x is None else np.nan_to_num(plate_x, nan=0.0)
        S = self.springs
        k = np.where(self.is_shear, p.k_shear, p.k_axial)
        kc = k / self.L0                                      # force per cm of stretch
        c_d = p.zeta * 2.0 * np.sqrt(kc * self.m_node)        # dashpot per spring
        m = self.m_node
        # stable explicit step: well under the fastest spring period
        # area term: force = -(kA*(A-A0) + cA*dA/dt)/A0 * dA/dx. With
        # |dA/dx|^2 ~ h^2/4 its effective node stiffness is kA*h^2/(4*A0);
        # choose cA for the same damping ratio zeta as the springs.
        h2 = (self.dx ** 2 + self.dy ** 2) / 4.0
        kA = p.k_axial
        k_eff_A = kA * h2 / self.A0.min()
        cA = p.zeta * 2.0 * np.sqrt(k_eff_A * m) * self.A0.min() / h2
        w_max = np.sqrt((16.0 * (kc.max() + k_eff_A) + p.k_base) / m)
        frame_dt = np.median(np.diff(t))
        # explicit step limits: stiffness (dt < ~2/w_max) and damping
        # (dt < 2*m/c_node, c_node = total dashpot on one node, ~8 springs)
        c_node = 8.0 * c_d.max() + cA * h2 / self.A0.min()
        dt_max = min(0.9 / w_max, 0.5 * m / c_node)
        if substeps is None:
            substeps = int(np.ceil(frame_dt / dt_max))
        dt = frame_dt / substeps
        free = ~self.fixed

        T = len(t)
        v_init = np.zeros_like(self.rest) if v0 is None else v0.astype(np.float64)
        # base: pinned, or (k_base > 0) held by springs to where it was taped,
        # so the column can rock slightly on a compliant base
        base = self.fixed.copy()
        kb = np.zeros(self.n)
        cb = np.zeros(self.n)
        free = ~self.fixed
        if p.k_base > 0:
            kb[base] = p.k_base
            cb[base] = p.zeta * 2.0 * np.sqrt(p.k_base * m)
            free = np.ones(self.n, bool)
        X = _run(self.rest.copy(), v_init, self.springs, kc, c_d, self.L0,
                 self.tris, self.A0, kA, cA, free, base, kb, cb,
                 c_, a_, x_, p.mu, p.alpha, substeps, dt, frame_dt, m, G)
        P = None
        if record_pts is not None:
            P = np.stack([self.interp(X[f], record_pts) for f in range(T)])
        return X, P


@njit(cache=True)
def _run(x, v, springs, kc, c_d, L0, tris, A0, kA, cA, free, base, kb, cb,
         plate_c, plate_a, plate_x, mu, alpha, substeps, dt, frame_dt, m, g):
    """Time-stepping kernel (compiled with numba). Semi-implicit Euler.

    Phone contact: one-sided, Coulomb friction relative to the moving plate.
    Table contact: one-sided and sticking (anchored at first-touch x)."""
    n = x.shape[0]
    T = plate_c.shape[0]
    x0 = x.copy()                   # rest positions (base anchors)
    F = np.zeros_like(x)
    on_p = np.zeros(n, np.bool_)    # node touching the phone (previous substep)
    anc_t = np.full(n, np.nan)      # anchor x on the table
    X = np.empty((T, n, 2))
    ns = springs.shape[0]
    for f in range(T):
        f1 = min(f + 1, T - 1)
        c0, c1 = plate_c[f], plate_c[f1]
        a0, a1 = plate_a[f], plate_a[f1]
        vx_p = (plate_x[f1] - plate_x[f]) / frame_dt         # plate sideways velocity
        for s in range(substeps):
            al = s / substeps
            pc = c0 + al * (c1 - c0)
            pa = a0 + al * (a1 - a0)
            # ---- spring + dashpot forces ----
            F[:, :] = 0.0
            for e in range(ns):
                i, j = springs[e, 0], springs[e, 1]
                dx = x[j, 0] - x[i, 0]
                dy = x[j, 1] - x[i, 1]
                L = np.sqrt(dx * dx + dy * dy) + 1e-12
                ux, uy = dx / L, dy / L
                rv = (v[j, 0] - v[i, 0]) * ux + (v[j, 1] - v[i, 1]) * uy
                fm = kc[e] * (L - L0[e]) + c_d[e] * rv
                F[i, 0] += fm * ux
                F[i, 1] += fm * uy
                F[j, 0] -= fm * ux
                F[j, 1] -= fm * uy
            # ---- signed-area (anti-inversion) forces ----
            for e in range(tris.shape[0]):
                a, b, c = tris[e, 0], tris[e, 1], tris[e, 2]
                ax_, ay_ = x[a, 0], x[a, 1]
                bx_, by_ = x[b, 0], x[b, 1]
                cx_, cy_ = x[c, 0], x[c, 1]
                A = 0.5 * ((bx_ - ax_) * (cy_ - ay_) - (by_ - ay_) * (cx_ - ax_))
                # gradient of A with respect to each corner
                gax, gay = 0.5 * (by_ - cy_), 0.5 * (cx_ - bx_)
                gbx, gby = 0.5 * (cy_ - ay_), 0.5 * (ax_ - cx_)
                gcx, gcy = 0.5 * (ay_ - by_), 0.5 * (bx_ - ax_)
                Ad = (gax * v[a, 0] + gay * v[a, 1] + gbx * v[b, 0] + gby * v[b, 1]
                      + gcx * v[c, 0] + gcy * v[c, 1])
                s_ = -(kA * (A - A0[e]) + cA * Ad) / A0[e]
                F[a, 0] += s_ * gax
                F[a, 1] += s_ * gay
                F[b, 0] += s_ * gbx
                F[b, 1] += s_ * gby
                F[c, 0] += s_ * gcx
                F[c, 1] += s_ * gcy
            for i in range(n):
                if not free[i]:
                    continue
                if base[i]:                          # compliant base spring
                    F[i, 0] -= kb[i] * (x[i, 0] - x0[i, 0]) + cb[i] * v[i, 0]
                    F[i, 1] -= kb[i] * (x[i, 1] - x0[i, 1]) + cb[i] * v[i, 1]
                F[i, 0] -= m * alpha * v[i, 0]      # mass-proportional damping
                F[i, 1] -= m * alpha * v[i, 1]
                # ---- sideways motion, with friction if touching the phone ----
                if on_p[i]:
                    fn = F[i, 1] - m * g            # net push into the phone
                    if fn < 0.0:
                        fn = 0.0
                    ft = F[i, 0]
                    vrel = v[i, 0] - vx_p
                    if abs(vrel) < 1e-6 and abs(ft) <= mu * fn:
                        v[i, 0] = vx_p             # sticking: carried by the phone
                    else:
                        sgn = np.sign(vrel) if abs(vrel) >= 1e-6 else np.sign(ft)
                        v[i, 0] += dt * (ft - sgn * mu * fn) / m
                        if np.sign(v[i, 0] - vx_p) == -sgn:
                            v[i, 0] = vx_p         # friction stopped the slip
                else:
                    v[i, 0] += dt * F[i, 0] / m
                v[i, 1] += dt * (F[i, 1] / m - g)
                x[i, 0] += dt * v[i, 0]
                x[i, 1] += dt * v[i, 1]
                # ---- phone: solid above y = pc - pa*x (one-sided) ----
                yp = pc - pa * x[i, 0]
                if x[i, 1] > yp:
                    x[i, 1] = yp
                    vy = ((c1 - c0) - (a1 - a0) * x[i, 0]) / frame_dt
                    if v[i, 1] > vy:
                        v[i, 1] = vy
                    on_p[i] = True
                else:
                    on_p[i] = False
                # ---- table: solid below y = 0 (the base itself sits in it) ----
                if x[i, 1] < 0.0 and not base[i]:
                    if np.isnan(anc_t[i]):
                        anc_t[i] = x[i, 0]
                    x[i, 0] = anc_t[i]
                    x[i, 1] = 0.0
                    v[i, 0] = 0.0
                    if v[i, 1] < 0.0:
                        v[i, 1] = 0.0
                else:
                    anc_t[i] = np.nan
        X[f] = x
    return X
