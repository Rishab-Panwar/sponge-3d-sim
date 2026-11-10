"""
Export the fitted 3D simulation of every clip for the interactive viewer.

Writes results/viewer/data.js (loaded by results/viewer/index.html). Open
index.html in a browser: rotate/zoom with the mouse, play/scrub the time line,
switch clips, and compare real (orange) with simulated (blue) dots.

Usage:
    python src/export_viewer.py
"""
import json
import os

import numpy as np

from data import DATA, ROOT, load_clip
from fit import model_for
from render import load_params, surface_quads

STEP = 2      # export every 2nd frame (120 fps of real time)


def main():
    p = load_params()
    with open(os.path.join(DATA, "clips.json")) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    with open(os.path.join(ROOT, "results", "fit_params.json")) as f:
        fitres = json.load(f)
    ev_path = os.path.join(ROOT, "results", "evaluation.json")
    evaluation = json.load(open(ev_path)) if os.path.exists(ev_path) else {}
    out = {"params": fitres["params"], "effective": fitres.get("effective", {}), "clips": {}}
    for name, c in cfg.items():
        clip = load_clip(name)
        model = model_for(clip)
        emb = model.embed(clip.dots_rest)
        X, P = model.simulate_clip(p, clip, emb)
        fr = np.arange(0, len(clip.t), STEP)
        role = c.get("role")
        if role == "fit" and not c.get("use_trajectory", True):
            role = "stress test"
        # current scores (evaluate.py) take precedence over the ones saved by the fit
        stats = (evaluation.get(name) or fitres.get("test", {}).get(name)
                 or fitres.get("fit", {}).get(name) or fitres.get("stress_test", {}).get(name) or {})
        out["clips"][name] = {
            "role": role,
            "rms_cm": stats.get("rms_cm"),
            "t": np.round(clip.t[fr], 4).tolist(),
            "release_t": clip.release_t,
            "depth": model.depth,
            "quads": surface_quads(model).tolist(),
            "nodes": np.round(X[fr], 3).tolist(),
            "sim_dots": np.round(P[fr], 3).tolist(),
            "real_dots": np.round(clip.dots[fr], 3).tolist(),
            "plate": [[None if not np.isfinite(clip.plate_c[i]) else round(float(clip.plate_c[i]), 3),
                       round(float(np.nan_to_num(clip.plate_a[i])), 4)] for i in fr],
        }
        print(f"[{name}] {len(fr)} frames")
    d = os.path.join(ROOT, "results", "viewer")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "data.js"), "w") as f:
        f.write("window.SPONGE_DATA = ")
        json.dump(out, f, separators=(",", ":"))
        f.write(";\n")
    print("wrote results/viewer/data.js "
          f"({os.path.getsize(os.path.join(d, 'data.js')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
