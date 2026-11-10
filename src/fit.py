"""
Fit the sponge model to the tracked videos, in two stages.

Stage A - dynamics (stiffness and damping), from the free wobble:
    After release the sponge top sways sideways and settles. wobble.py fits a
    damped sine to that motion in every clip, giving a natural frequency f_n
    and damping ratio zeta. The targets are the averages over the FIT clips
    (weighted by wobble amplitude). We then choose the model's overall
    stiffness k_axial and its mass-proportional damping alpha so that a
    virtual "pluck" of the model wobbles with the same f_n and zeta:
        f_n ~ sqrt(k_axial / mass)          -> rescale k_axial
        zeta_sway ~ alpha / (2 * omega_n)   -> adjust alpha
    (a quick fixed-point iteration).

Stage B - shape, from the trajectories:
    With stage A tied in, the remaining parameters are the shear/axial
    stiffness ratio (how the sponge bends vs. shears), the phone-foam
    friction mu and the material (spring) damping zeta. They are fitted on the dot trajectories: the simulation is
    driven ONLY by the tracked phone motion and must reproduce the dots.
    Loss per clip = mean sq. error while pressing + mean sq. error in the
    first FREE_WINDOW s after release (each averaged separately, so the long
    press does not drown out the short free-recovery window).

Stages A and B are alternated a few times (they interact slightly).
'test' clips (clips.json) are never used by the fit.

Usage:
    python src/fit.py
Outputs:
    results/fit_params.json
"""
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

import wobble
from data import DATA, ROOT, load_clip, sponge_dims
from sim import Params, SpongeModel
from sim3d import SpongeModel3D

# The results use the 3D model. The 2D side-view model (sim.SpongeModel) was
# the development model; set MODEL_DIM = 2 to fit it instead.
MODEL_DIM = 3

FREE_WINDOW = 0.25
W_FREE = 1.0
OUTER_ITERS = 2


def clip_names(role, trajectory=None):
    """Clips with this role. trajectory=True/False also filters on the
    clip's use_trajectory flag (default True)."""
    with open(os.path.join(DATA, "clips.json")) as f:
        cfg = json.load(f)
    return [k for k, v in cfg.items() if not k.startswith("_") and v.get("role") == role
            and (trajectory is None or v.get("use_trajectory", True) == trajectory)]


def make_model(bow=None):
    d = sponge_dims()
    if MODEL_DIM == 3:
        return SpongeModel3D(width=d["width_cm"], height=d["height_cm"],
                             depth=d["thickness_cm"], bow=bow)
    return SpongeModel(width=d["width_cm"], height=d["height_cm"], bow=bow)


_MODELS = {}


def model_for(clip):
    """Model whose rest shape is this clip's measured initial bow/lean."""
    if clip.name not in _MODELS:
        _MODELS[clip.name] = make_model(clip.rest_bow)
    return _MODELS[clip.name]


# ---------------------------------------------------------------- stage A ---
def model_wobble(model, p):
    t, x = model.pluck(p)
    return wobble.fit_wobble(t, x)


def match_dynamics(model, ratio, mu, zeta, f_target, zeta_target, k0=2e5, a0=15.0, iters=5):
    """Find k_axial and alpha so the model's pluck wobbles at f_target with
    damping ratio zeta_target (k_shear = ratio * k_axial; zeta, mu fixed)."""
    k, a = k0, a0
    for _ in range(iters):
        r = model_wobble(model, Params(k, ratio * k, zeta, mu, a))
        k *= (f_target / r["f_n_hz"]) ** 2                       # f ~ sqrt(k)
        a += 2 * (2 * np.pi * f_target) * (zeta_target - r["zeta"])  # zeta ~ alpha/(2w)
        a = float(max(a, 0.0))
    p = Params(k, ratio * k, zeta, mu, a)
    return p, model_wobble(model, p)


# ---------------------------------------------------------------- stage B ---
def errors(model, p, clip):
    """Per-frame, per-dot distance (cm) between simulation and tracking.
    (Uses the clip's own model, with its measured initial shape.)"""
    model = model_for(clip)
    emb = model.embed(clip.dots_rest)
    _, P = model.simulate_clip(p, clip, emb)
    return np.linalg.norm(P - clip.dots, axis=2)


def windows(clip):
    press = clip.t <= clip.release_t
    free = (clip.t > clip.release_t) & (clip.t <= clip.release_t + FREE_WINDOW)
    return press, free


def clip_loss(model, p, clip):
    e = errors(model, p, clip)
    if not np.all(np.isfinite(e)):
        return 1e3
    press, free = windows(clip)
    msq = lambda m: float((e[m] ** 2).mean()) if m.any() else 0.0
    return msq(clip.ok & press[:, None]) + W_FREE * msq(clip.ok & free[:, None])


def summarize(model, p, clip):
    e = errors(model, p, clip)
    ok = clip.ok
    press, free = windows(clip)
    rms = lambda m: float(np.sqrt((e[m] ** 2).mean())) if m.any() else float("nan")
    return {"rms_cm": rms(ok),
            "rms_cm_pressing": rms(ok & press[:, None]),
            "rms_cm_free_recovery": rms(ok & free[:, None]),
            "rms_cm_after_release": rms(ok & ~press[:, None]),
            "max_cm": float(e[ok].max())}


def main():
    t0 = time.time()
    model = make_model()
    wobble_clips = [load_clip(n) for n in clip_names("fit")]                 # stage A
    fit_clips = [load_clip(n) for n in clip_names("fit", trajectory=True)]   # stage B
    stress_clips = [load_clip(n) for n in clip_names("fit", trajectory=False)]
    test_clips = [load_clip(n) for n in clip_names("test")]
    print("wobble (stage A):", [c.name for c in wobble_clips],
          " trajectories (stage B):", [c.name for c in fit_clips],
          " stress test:", [c.name for c in stress_clips], " test:", [c.name for c in test_clips])

    # ---- targets from the video (fit clips only), weighted by wobble size ----
    meas = {c.name: wobble.measure_clip(c) for c in wobble_clips}
    w = np.array([meas[n]["amplitude_cm"] for n in meas])
    f_t = float(np.average([meas[n]["f_n_hz"] for n in meas], weights=w))
    z_t = float(np.average([meas[n]["zeta"] for n in meas], weights=w))
    per = ", ".join(f"{n} {m['f_n_hz']:.1f} Hz / {m['zeta']:.2f}" for n, m in meas.items())
    print(f"targets from video: f_n = {f_t:.2f} Hz, zeta = {z_t:.3f}  ({per})")

    ratio, mu, zeta = 0.5, 1.0, 0.2
    p = None
    for it in range(OUTER_ITERS):
        # Stage A
        p, mw = match_dynamics(model, ratio, mu, zeta, f_t, z_t,
                               *((p.k_axial, p.alpha) if p else ()))
        print(f"[iter {it}] stage A: k_axial {p.k_axial:.3g}, alpha {p.alpha:.2f} 1/s "
              f"-> model wobble {mw['f_n_hz']:.2f} Hz, zeta {mw['zeta']:.3f}")

        # Stage B: shear ratio, friction, material damping (stage A re-solved inside)
        def objective(z, p=p):
            r_, m_, zz = (float(v) for v in np.exp(z))
            q, _ = match_dynamics(model, r_, m_, zz, f_t, z_t,
                                  k0=p.k_axial, a0=p.alpha, iters=2)
            return float(np.mean([clip_loss(model, q, c) for c in fit_clips]))

        z0 = np.log([ratio, mu, zeta])
        if it == 0:   # coarse grid first: the loss is not convex
            grid = [(objective(np.log([r_, m_, zz])), np.log([r_, m_, zz]))
                    for r_ in (0.25, 0.5, 1.0) for m_ in (0.5, 1.0, 2.0)
                    for zz in (0.1, 0.4)]
            z0 = min(grid, key=lambda g: g[0])[1]
            print(f"[iter {it}] grid: best loss {min(g[0] for g in grid):.4f} at ratio "
                  f"{np.exp(z0[0]):.2f}, mu {np.exp(z0[1]):.2f}, zeta {np.exp(z0[2]):.2f} "
                  f"({time.time() - t0:.0f}s)")
        # explicit starting simplex: steps of x1.35 in every parameter (the
        # default step is proportional to the value, i.e. ~0 when log(x) = 0)
        simplex = np.vstack([z0] + [z0 + 0.3 * np.eye(3)[k] for k in range(3)])
        res = minimize(objective, z0, method="Nelder-Mead",
                       options={"xatol": 0.03, "fatol": 1e-4, "maxiter": 40,
                                "initial_simplex": simplex})
        ratio, mu, zeta = (float(v) for v in np.exp(res.x))
        print(f"[iter {it}] stage B: shear ratio {ratio:.3f}, friction mu {mu:.2f}, "
              f"material damping zeta {zeta:.3f}, loss {res.fun:.4f} cm^2  ({time.time() - t0:.0f}s)")

    p, mw = match_dynamics(model, ratio, mu, zeta, f_t, z_t, p.k_axial, p.alpha)
    test_meas = {c.name: wobble.measure_clip(c) for c in test_clips}
    out = {
        "model": f"{MODEL_DIM}D mass-spring grid",
        "params": {"k_axial": p.k_axial, "k_shear": p.k_shear, "zeta": p.zeta, "mu": p.mu,
                   "alpha": p.alpha},
        "units": "per unit total mass: k in cm/s^2 per unit strain; zeta (spring dashpot "
                 "ratio) and mu dimensionless; alpha in 1/s",
        "effective": {"wobble_frequency_hz": mw["f_n_hz"], "wobble_damping_ratio": mw["zeta"],
                      "shear_to_axial_ratio": ratio},
        "targets_from_fit_clips": {"f_n_hz": f_t, "zeta": z_t,
                                   "per_clip": {n: {"f_n_hz": m["f_n_hz"], "zeta": m["zeta"],
                                                    "amplitude_cm": m["amplitude_cm"]}
                                                for n, m in meas.items()}},
        "test_clips_measured_wobble": {n: {"f_n_hz": m["f_n_hz"], "zeta": m["zeta"],
                                           "amplitude_cm": m["amplitude_cm"]}
                                       for n, m in test_meas.items()},
        "loss": {"free_window_s": FREE_WINDOW, "w_free": W_FREE},
        "fit": {c.name: summarize(model, p, c) for c in fit_clips},
        "stress_test": {c.name: summarize(model, p, c) for c in stress_clips},
        "test": {c.name: summarize(model, p, c) for c in test_clips},
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "fit_params.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"final: {p}")
    print(f"model wobble {mw['f_n_hz']:.2f} Hz / zeta {mw['zeta']:.3f}; "
          + "; ".join(f"test {n} measured {m['f_n_hz']:.2f} Hz / zeta {m['zeta']:.3f}"
                      for n, m in test_meas.items()))
    for grp in ("fit", "stress_test", "test"):
        for n, s in out[grp].items():
            print(f"[{grp}] {n}: rms {s['rms_cm']:.2f} cm (pressing {s['rms_cm_pressing']:.2f}, "
                  f"free recovery {s['rms_cm_free_recovery']:.2f}, "
                  f"after release {s['rms_cm_after_release']:.2f}), max {s['max_cm']:.2f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
