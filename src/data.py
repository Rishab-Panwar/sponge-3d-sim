"""
Load tracked clips and convert them from image pixels to a physical frame.

Physical frame (used by the simulator):
    x  [cm]  to the right, 0 = centre of the sponge's base
    y  [cm]  up,           0 = bottom of the sponge (table / tape)
    t  [s]   real time (frame index / capture fps)

Scale: each clip is calibrated separately, because the camera distance changed
between recordings. The ruler is the sponge's resting height in that clip
(mask height before the phone touches) = height_cm from data/sponge.json.
"""
import csv
import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt, savgol_filter

# Light smoothing of the tracked phone pose (frames). Removes frame-to-frame
# tracking jitter (which can flip the sponge's buckling direction in the
# simulation) but keeps the hand's real motion: 9 frames = 37 ms at 240 fps.
INPUT_SMOOTH = 9


def _smooth(a):
    g = np.isfinite(a)
    if g.sum() > INPUT_SMOOTH:
        a = a.copy()
        a[g] = savgol_filter(a[g], INPUT_SMOOTH, 2)
    return a

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def sponge_dims():
    with open(os.path.join(DATA, "sponge.json")) as f:
        return json.load(f)


@dataclass
class Clip:
    name: str
    fps: float
    t: np.ndarray            # (T,) seconds
    dots: np.ndarray         # (T, N, 2) tracked dot positions [cm]
    ok: np.ndarray           # (T, N) bool, dot verified on a detected marker
    dots_rest: np.ndarray    # (N, 2) dot positions in the first frame [cm]
    plate_c: np.ndarray      # (T,) presser plane: y = plate_c - plate_a * x  [cm]
    plate_a: np.ndarray      # (T,) (NaN where the phone is not visible)
    plate_x: np.ndarray      # (T,) sideways position of the phone [cm], relative
                             #      (NaN gaps filled by holding the last value)
    contact: np.ndarray      # (T,) bool, tracked contact (for plots / weighting)
    release_t: float
    touch_t: float
    px_per_cm: float
    origin_px: tuple         # (x0, y0) image pixel of the physical origin
    rest_bow: tuple          # (ys, us): centre-line sideways offset [cm] at heights ys,
                             #           averaged over the frames before contact
    video: str               # path of the source video
    cam_px: np.ndarray       # (T, 2) camera shift vs frame 0 [px]; add back to draw on the video


def load_clip(name):
    with open(os.path.join(DATA, "tracks", f"{name}_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(DATA, "tracks", f"{name}.csv")) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])
    n = meta["n_dots"]
    T = len(rows)

    # ---- scale: resting sponge height in this clip -----------------------
    touch = max(meta["touch_frame"], 1)
    rest = slice(0, touch)
    top = np.median(col("sponge_top_y")[rest])
    bottom = np.median(col("mask_bottom_y")[rest])
    s = (bottom - top) / sponge_dims()["height_cm"]          # px per cm

    # ---- origin: base of the sponge --------------------------------------
    # ---- camera-motion compensation (video stabilization) ----------------
    # The tracker measured how far the whole image moved relative to frame 0
    # (camera nudged during the press). Subtract it from everything tracked.
    cdx = medfilt(col("cam_dx"), 5) if "cam_dx" in rows[0] else np.zeros(T)
    cdy = medfilt(col("cam_dy"), 5) if "cam_dy" in rows[0] else np.zeros(T)
    dx = np.stack([col(f"d{i}_x") for i in range(n)], 1) - cdx[:, None]
    dy = np.stack([col(f"d{i}_y") for i in range(n)], 1) - cdy[:, None]
    ncols = meta["n_cols"]
    bottom_row = slice(n - ncols, n)
    x0 = dx[0, bottom_row].mean()
    y0 = bottom

    dots = np.stack([(dx - x0) / s, (y0 - dy) / s], -1)
    ok = np.stack([col(f"d{i}_ok") for i in range(n)], 1) > 0

    # ---- presser plane ---------------------------------------------------
    # image line y_px = a*x_px + b  ->  y_cm = c - a*x_cm
    a = col("phone_a")
    b = col("phone_b") + a * cdx - cdy        # line in stabilized coordinates
    vis = np.isfinite(a) & np.isfinite(b)
    if vis.sum() > 5:      # remove single-frame jitter
        a[vis] = medfilt(a[vis], 5)
        b[vis] = medfilt(b[vis], 5)
    # The visible phone edge sits a few px off the true contact plane
    # (perspective, phone thickness, fuzzy foam). The tracker measured that
    # offset (contact_level_px) while the phone was pressing; apply it.
    off = meta["contact_level_px"]
    c = (y0 - b - a * x0 - off) / s

    # Sideways phone motion, from its right end. Only changes matter (it is a
    # velocity input for the contact). Gaps (end not visible) hold the last
    # value, i.e. no sideways motion is assumed there.
    ex = col("phone_end_x") - cdx
    good = np.isfinite(ex)
    px = np.zeros(T)
    if good.sum() > 5:
        exs = ex.copy()
        exs[good] = medfilt(ex[good], 5)
        last = exs[good][0]
        for i in range(T):
            if good[i]:
                last = exs[i]
            px[i] = last
        px = (px - px[0]) / s

    # Initial shape: centre line of each dot row, averaged before contact.
    # The base (y = 0) is taped down at x = 0.
    ncols_ = meta["n_cols"]
    pre = dots[:touch].mean(0).reshape(-1, ncols_, 2)
    ys_ = pre[:, :, 1].mean(1)[::-1]                 # bottom -> top
    us_ = pre[:, :, 0].mean(1)[::-1]
    us_ = us_ - us_[0]                               # bottom row centred on the base
    rest_bow = (np.r_[0.0, ys_], np.r_[0.0, us_])

    c, a, px = _smooth(c), _smooth(a), _smooth(px)

    return Clip(name=name, fps=meta["fps"], t=col("t"), dots=dots, ok=ok,
                dots_rest=dots[0].copy(), plate_c=c, plate_a=a, plate_x=px,
                contact=col("contact") > 0, release_t=meta["release_t"],
                touch_t=meta["touch_t"], px_per_cm=s, origin_px=(float(x0), float(y0)), rest_bow=rest_bow,
                cam_px=np.stack([cdx, cdy], 1),
                video=os.path.join(DATA, _clip_cfg(name)["file"]))


def _clip_cfg(name):
    with open(os.path.join(DATA, "clips.json")) as f:
        return json.load(f)[name]


def to_px(clip, xy):
    """Physical (cm) -> original video pixels."""
    xy = np.asarray(xy, float)
    x0, y0 = clip.origin_px
    return np.stack([x0 + clip.px_per_cm * xy[..., 0], y0 - clip.px_per_cm * xy[..., 1]], -1)


if __name__ == "__main__":
    with open(os.path.join(DATA, "clips.json")) as f:
        names = [k for k in json.load(f) if not k.startswith("_")]
    for nm in names:
        c = load_clip(nm)
        w = c.dots_rest[:, 0].max() - c.dots_rest[:, 0].min()
        h = c.dots_rest[:, 1].max() - c.dots_rest[:, 1].min()
        print(f"{nm}: {c.px_per_cm:.1f} px/cm, {len(c.t)} frames, release {c.release_t:.3f}s, "
              f"dot grid {w:.2f} x {h:.2f} cm, top dots at y={c.dots_rest[:2, 1].round(2)}")
