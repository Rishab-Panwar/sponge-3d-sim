"""
Measure the sponge's free sideways wobble: natural frequency and damping ratio.

After the phone lets go, the top of the sponge sways sideways a few times and
settles. We fit a damped oscillation (plus a slow linear drift, which absorbs
the slow creep back to rest) to the top row's sideways position:

    x(t) = A * exp(-zeta * w_n * t) * cos(w_d * t + phi) + B + C * t
    w_d  = w_n * sqrt(1 - zeta^2)

and report f_n = w_n / (2*pi) [Hz] and zeta [-].

The same function measures the simulated sponge after a virtual "pluck", so
video and model are characterised in exactly the same way.

Usage:
    python src/wobble.py      # measure every clip, write results/wobble_measured.json
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

from data import DATA, ROOT, load_clip

WINDOW = (0.015, 0.30)   # seconds after release used for the fit


def damped(t, A, zeta, wn, phi, B, C):
    wd = wn * np.sqrt(max(1.0 - zeta ** 2, 1e-6))
    return A * np.exp(-zeta * wn * t) * np.cos(wd * t + phi) + B + C * t


def fit_wobble(t, x, f_guess=13.0):
    """Fit a damped oscillation to x(t) (t starts at 0). Returns a dict with
    f_n [Hz], zeta, the fitted curve and the rms residual; None if it fails."""
    best = None
    for f0 in (f_guess * 0.7, f_guess, f_guess * 1.4):
        for phi0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            p0 = [np.ptp(x) / 2 + 1e-3, 0.1, 2 * np.pi * f0, phi0, np.median(x), 0.0]
            try:
                p, _ = curve_fit(damped, t, x, p0=p0, maxfev=20000,
                                 bounds=([-10, 0.0, 2 * np.pi * 2, -2 * np.pi, -20, -50],
                                         [10, 0.9, 2 * np.pi * 60, 4 * np.pi, 20, 50]))
            except (RuntimeError, ValueError):
                continue
            r = np.sqrt(np.mean((damped(t, *p) - x) ** 2))
            if best is None or r < best[1]:
                best = (p, r)
    if best is None:
        return None
    p, r = best
    return {"f_n_hz": float(p[2] / (2 * np.pi)), "zeta": float(p[1]),
            "amplitude_cm": float(abs(p[0])), "rms_residual_cm": float(r), "coeffs": p.tolist()}


def measure_clip(clip):
    """Wobble of the real sponge after release (top row of dots, sideways)."""
    t = clip.t - clip.release_t
    sel = (t >= WINDOW[0]) & (t <= WINDOW[1])
    x = clip.dots[:, :2, 0].mean(1)          # top row = first two dots
    res = fit_wobble(t[sel] - WINDOW[0], x[sel])
    if res:
        res["t"] = (t[sel] - WINDOW[0]).tolist()
        res["x"] = x[sel].tolist()
    return res


def main():
    with open(os.path.join(DATA, "clips.json")) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    out = {}
    for name, c in cfg.items():
        res = measure_clip(load_clip(name))
        if res is None:
            print(f"{name}: fit failed")
            continue
        out[name] = {k: v for k, v in res.items() if k not in ("t", "x", "coeffs")}
        out[name]["role"] = c.get("role")
        print(f"{name} [{c.get('role')}]: f_n = {res['f_n_hz']:.1f} Hz, zeta = {res['zeta']:.3f}, "
              f"amplitude {res['amplitude_cm']:.2f} cm, residual {res['rms_residual_cm']:.3f} cm")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "wobble_measured.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot_fits(cfg)


def plot_fits(cfg):
    """results/plots/sway_fit.png: measured sway + damped-sine fit, every clip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(cfg)
    fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.8))
    axs = np.atleast_1d(axs)
    for ax, name in zip(axs, names):
        r = measure_clip(load_clip(name))
        if r is None:
            continue
        t = np.array(r["t"])
        ax.plot(t * 1000, r["x"], "k.", ms=3, label="video (top dots)")
        ax.plot(t * 1000, damped(t, *r["coeffs"]), "C3", lw=1.5, label="damped-sine fit")
        ax.set_title(f"{name} ({cfg[name].get('role')}): {r['f_n_hz']:.1f} Hz, "
                     f"zeta {r['zeta']:.2f}", fontsize=9)
        ax.set_xlabel("ms after release")
    axs[0].set_ylabel("sideways position [cm]")
    axs[0].legend(fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.join(ROOT, "results", "plots"), exist_ok=True)
    plt.savefig(os.path.join(ROOT, "results", "plots", "sway_fit.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
