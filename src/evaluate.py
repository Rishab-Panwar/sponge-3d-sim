"""
Evaluate the fitted model on any clip WITHOUT refitting.

Uses the saved parameters in results/fit_params.json. For each clip it
simulates the sponge from the clip's tracked phone motion (and its starting
shape before contact), then reports how far the simulated dots are from the
tracked dots, and compares the predicted sway with the sway measured in the clip.

New test video:
  1. put the video in data/raw/ (same setup: side view, yellow sponge, 5 x 2
     dark dots on the visible face, dark phone pressing from the top)
  2. add it to data/clips.json, e.g.
        "clip5": {"file": "raw/clip5.mp4", "capture_fps": 240, "role": "test"}
  3. python track.py clip5 --overlay      (check results/tracking/clip5_overlay.mp4)
     python evaluate.py clip5
     python render.py clip5               (demo video + plot)

Usage:
    python evaluate.py            # every clip in clips.json
    python evaluate.py clip5      # just these
Outputs:
    results/evaluation.json  (merged with earlier entries)
"""
import json
import os
import sys

import wobble
from data import DATA, ROOT, load_clip
from fit import make_model, summarize
from render import load_params


def main():
    names = sys.argv[1:]
    with open(os.path.join(DATA, "clips.json")) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    names = names or list(cfg)
    p = load_params()
    pred = wobble.fit_wobble(*make_model().pluck(p))
    print(f"fitted model: sway {pred['f_n_hz']:.2f} Hz, damping ratio {pred['zeta']:.3f}")
    path = os.path.join(ROOT, "results", "evaluation.json")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            out = json.load(f)
    out["_model_sway"] = {"f_n_hz": pred["f_n_hz"], "zeta": pred["zeta"]}
    for n in names:
        if n not in cfg:
            sys.exit(f"{n} is not in data/clips.json")
        clip = load_clip(n)
        s = summarize(None, p, clip)
        meas = wobble.measure_clip(clip)
        if meas:
            s["measured_sway_hz"] = meas["f_n_hz"]
            s["measured_damping_ratio"] = meas["zeta"]
        s["role"] = cfg[n].get("role")
        out[n] = s
        sway = (f"; measured sway {meas['f_n_hz']:.2f} Hz / {meas['zeta']:.3f} "
                f"(model {pred['f_n_hz']:.2f} / {pred['zeta']:.3f})") if meas else ""
        print(f"[{n}] ({s['role']}) dot error rms {s['rms_cm']:.2f} cm: pressing "
              f"{s['rms_cm_pressing']:.2f}, free recovery {s['rms_cm_free_recovery']:.2f}, "
              f"max {s['max_cm']:.2f}{sway}")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
