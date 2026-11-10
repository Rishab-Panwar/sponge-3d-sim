"""
Build the assets for the results web page (index.html at the repo root).

  results/web/<clip>.mp4              demo videos re-encoded to H.264 (browsers
                                      cannot play OpenCV's default mp4v codec)
  results/plots/errors_per_clip.png   dot error per clip (pressing / free recovery)
  results/plots/sway_comparison.png   measured vs predicted sway frequency + damping
  results/plots/model_2d_vs_3d.png    2D vs 3D model, same data
  results/plots/robustness.png        error under +-5 % parameter changes
  results/model_2d_vs_3d.json

Run after evaluate.py and render.py:
    python make_page.py
"""
import json
import os
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fit  # noqa: E402
import wobble  # noqa: E402
from data import DATA, ROOT, load_clip  # noqa: E402
from sim import Params  # noqa: E402

RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")
# palette of the results page: ink = fitting clips, orange = unseen test clips
C_FIT, C_TEST, C_STRESS = "#383b36", "#ff7900", "#b8b6ae"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.edgecolor": "#64665f",
                     "axes.labelcolor": "#17191d", "xtick.color": "#535650",
                     "ytick.color": "#535650", "axes.facecolor": "#f9f8f4",
                     "figure.facecolor": "#f9f8f4", "savefig.facecolor": "#f9f8f4"})


def role_color(role):
    return {"fit": C_FIT, "test": C_TEST}.get(role, C_STRESS)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def clip_cfg():
    with open(os.path.join(DATA, "clips.json")) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    for v in cfg.values():
        if v.get("role") == "fit" and not v.get("use_trajectory", True):
            v["role"] = "stress test"
    return cfg


def web_videos(cfg):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = os.path.join(RES, "web")
    os.makedirs(out, exist_ok=True)
    for n in cfg:
        src = os.path.join(RES, "demo", f"{n}.mp4")
        dst = os.path.join(out, f"{n}.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-vf", "scale=1100:-2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                        "-preset", "medium", "-movflags", "+faststart", "-an", dst], check=True)
        print(f"[web] {n}.mp4  {os.path.getsize(dst) / 1e6:.1f} MB")


def bar_errors(ev, cfg):
    names = list(cfg)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(names))
    w = 0.38
    pr = [ev[n]["rms_cm_pressing"] for n in names]
    fr = [ev[n]["rms_cm_free_recovery"] for n in names]
    cols = [role_color(cfg[n]["role"]) for n in names]
    ax.bar(x - w / 2, pr, w, color=cols, label="while pressing")
    ax.bar(x + w / 2, fr, w, color=cols, alpha=0.5, hatch="//", edgecolor="white",
           label="first 0.25 s after release")
    for i, n in enumerate(names):
        ax.text(i, max(pr[i], fr[i]) + 0.02, f"{ev[n]['rms_cm']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{n}\n({cfg[n]['role']})" for n in names])
    ax.set_ylabel("dot error, RMS [cm]")
    ax.set_title("Prediction error per clip (number = overall RMS); dots move up to ~4 cm", fontsize=10)
    ax.set_ylim(0, max(pr + fr) * 1.3)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "errors_per_clip.png"), dpi=130)
    plt.close(fig)


def sway(wm, cfg, p):
    pred = wobble.fit_wobble(*fit.make_model().pluck(p))
    names = [n for n in cfg if n in wm]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, lab, pv in [(axs[0], "f_n_hz", "sway frequency [Hz]", pred["f_n_hz"]),
                             (axs[1], "zeta", "damping ratio", pred["zeta"])]:
        vals = [wm[n][key] for n in names]
        ax.bar(range(len(names)), vals, color=[role_color(cfg[n]["role"]) for n in names])
        ax.axhline(pv, color="k", lw=1.5, ls="--", label=f"fitted model: {pv:.3g}")
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel(lab)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylim(0, max(wm[n]["f_n_hz"] for n in names) * 1.2)
    fig.suptitle("Measured sway after release (dark/grey = used to fit the sway, orange = unseen "
                 "test clips) vs the fitted model (dashed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "sway_comparison.png"), dpi=130)
    plt.close(fig)
    return pred


def two_d_vs_three_d(cfg, p3):
    p2 = Params(**load("fit_params_2d.json")["params"])
    out = {}
    for dim, p in [(3, p3), (2, p2)]:
        fit.MODEL_DIM = dim
        fit._MODELS.clear()
        out[f"{dim}D"] = {n: fit.summarize(None, p, load_clip(n))["rms_cm"] for n in cfg}
    fit.MODEL_DIM = 3
    fit._MODELS.clear()
    with open(os.path.join(RES, "model_2d_vs_3d.json"), "w") as f:
        json.dump(out, f, indent=2)
    names = list(cfg)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [out["2D"][n] for n in names], 0.4, color="#b8b6ae", label="2D side-view model")
    ax.bar(x + 0.2, [out["3D"][n] for n in names], 0.4, color="#ff7900", label="3D model (used)")
    ax.set_xticks(x, [f"{n}\n({cfg[n]['role']})" for n in names])
    ax.set_ylabel("dot error, RMS [cm]")
    ax.set_title("2D vs 3D model, same data and method", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "model_2d_vs_3d.png"), dpi=130)
    plt.close(fig)
    print("[2D vs 3D]", {k: {n: round(v, 2) for n, v in d.items()} for k, d in out.items()})


def robustness():
    rows = load("robustness_3d.json")
    labels = list(rows)
    vals = np.array([rows[k] for k in labels])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for j, (n, c) in enumerate([("clip1", C_FIT), ("clip2", C_FIT), ("clip4", C_TEST)]):
        ax.plot(range(len(labels)), vals[:, j], "o-", color=c, alpha=0.5 + 0.25 * (j == 2),
                label=n)
    ax.set_xticks(range(len(labels)), [l.replace(" x", "\nx") for l in labels], fontsize=7)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("dot error, RMS [cm]")
    ax.set_title("Robustness: each fitted parameter changed by ±5 %", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "robustness.png"), dpi=130)
    plt.close(fig)


def main():
    cfg = clip_cfg()
    os.makedirs(PLOTS, exist_ok=True)
    p = Params(**load("fit_params.json")["params"])
    ev = load("evaluation.json")
    wm = load("wobble_measured.json")
    bar_errors(ev, cfg)
    pred = sway(wm, cfg, p)
    print(f"[sway] model {pred['f_n_hz']:.2f} Hz / {pred['zeta']:.3f}")
    robustness()
    two_d_vs_three_d(cfg, p)
    web_videos(cfg)


if __name__ == "__main__":
    main()
