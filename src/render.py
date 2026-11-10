"""
Make the demo: real video next to the fitted simulation, plus plots.

For each clip:
  results/demo/<clip>.mp4          side-by-side video (slow motion):
        left  : real video, with the SIMULATED sponge outline/mesh drawn on top
        right : the simulation alone, with the phone plate that drives it
        bottom: top-dot position over time, real (thick) vs simulated (thin grey)
  results/plots/<clip>_traces.png  all dots, real vs simulated, over time

The simulation is driven only by the tracked phone motion (its height, tilt
and sideways position). The dots are never fed to it; they are what it predicts.

Usage:
    python src/render.py              # every clip
    python src/render.py clip4        # just one
"""
import json
import os
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import DATA, ROOT, load_clip, to_px  # noqa: E402
from fit import model_for  # noqa: E402
from sim import Params, SpongeModel  # noqa: E402

OUT_H = 720          # height of each video panel
PLOT_H = 220         # height of the time-series strip


def load_params():
    with open(os.path.join(ROOT, "results", "fit_params.json")) as f:
        return Params(**json.load(f)["params"])


def outline_ids(model):
    nx, ny = model.nx, model.ny
    idx = lambda i, j: j * nx + i
    ids = [idx(i, 0) for i in range(nx)]
    ids += [idx(nx - 1, j) for j in range(1, ny)]
    ids += [idx(i, ny - 1) for i in range(nx - 2, -1, -1)]
    ids += [idx(0, j) for j in range(ny - 2, 0, -1)]
    return np.array(ids)


def draw_mesh(img, pts, model, color, fill=None, thick=1):
    poly = pts[outline_ids(model)].astype(np.int32)
    if fill is not None:
        over = img.copy()
        cv2.fillPoly(over, [poly], fill)
        cv2.addWeighted(over, 0.35, img, 0.65, 0, img)
    for i, j in model.springs:
        cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), color, 1, cv2.LINE_AA)
    cv2.polylines(img, [poly], True, color, thick + 1, cv2.LINE_AA)


# ---------------------------------------------------------------- 3D view --
def surface_quads(model):
    """Outer surface of the 3D grid as quads (node index lists)."""
    nx, ny, nz, idx = model.nx, model.ny, model.nz, model.idx
    q = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            for k in (0, nz - 1):                   # front / back faces
                q.append([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k)])
        for k in range(nz - 1):
            for i in (0, nx - 1):                   # left / right faces
                q.append([idx(i, j, k), idx(i, j, k + 1), idx(i, j + 1, k + 1), idx(i, j + 1, k)])
    for k in range(nz - 1):
        for i in range(nx - 1):
            for j in (0, ny - 1):                   # bottom / top faces
                q.append([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j, k + 1), idx(i, j, k + 1)])
    return np.array(q)


def view_matrix(az_deg=-35.0, el_deg=18.0):
    az, el = np.radians(az_deg), np.radians(el_deg)
    Ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return Rx @ Ry


def draw_3d(img, model, x, quads, R, to_scr, plate=None, dots=None):
    """Shaded 3D view (painter's algorithm, orthographic)."""
    xr = x @ R.T                                      # rotated: x right, y up, z toward viewer
    light = np.array([0.4, 0.7, 0.6]) / np.linalg.norm([0.4, 0.7, 0.6])
    ctr = x.mean(0)
    if plate is not None:                             # phone plate, drawn first (behind)
        c, a = plate
        X_ = np.array([-4.5, 4.5, 4.5, -4.5])
        Z_ = np.array([-model.depth / 2 - 0.8, -model.depth / 2 - 0.8,
                       model.depth / 2 + 0.8, model.depth / 2 + 0.8])
        P_ = np.stack([X_, c - a * X_ + 0.02, Z_], 1) @ R.T
        over = img.copy()
        cv2.fillPoly(over, [to_scr(P_).astype(np.int32)], (60, 60, 60))
        cv2.addWeighted(over, 0.45, img, 0.55, 0, img)
    depth = xr[quads][:, :, 2].mean(1)
    for qi in np.argsort(depth):                      # far -> near
        pts = x[quads[qi]]
        nrm = np.cross(pts[2] - pts[0], pts[3] - pts[1])
        nrm /= np.linalg.norm(nrm) + 1e-12
        if np.dot(nrm, pts.mean(0) - ctr) < 0:
            nrm = -nrm                                # outward
        shade = 0.35 + 0.65 * abs(float(np.dot(nrm, light)))
        col = tuple(int(c_ * shade) for c_ in (90, 215, 240))
        poly = to_scr(xr[quads[qi]]).astype(np.int32)
        cv2.fillPoly(img, [poly], col, cv2.LINE_AA)
        cv2.polylines(img, [poly], True, (40, 110, 150), 1, cv2.LINE_AA)
    if dots is not None:                              # dots on the front face
        for d in to_scr(dots @ R.T):
            cv2.circle(img, tuple(d.astype(int)), 4, (40, 40, 40), -1, cv2.LINE_AA)


def plot_strip(w, h, t, real, sim, f, release_t, label):
    """Time series of top-dot x and y (real solid, sim dashed) as an image."""
    img = np.full((h, w, 3), 255, np.uint8)
    t0, t1 = t[0], t[-1]
    pad = 40
    X = lambda tt: (pad + (tt - t0) / (t1 - t0) * (w - 2 * pad)).astype(int)
    for k, (col, name) in enumerate([((200, 120, 0), "x (sideways)"), ((0, 140, 255), "y (height)")]):
        vals = np.r_[real[:, k], sim[:, k]]
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        top = 10 + k * (h - 20) // 2
        hh = (h - 20) // 2 - 10
        Y = lambda v: (top + hh - (v - lo) / max(hi - lo, 1e-6) * hh).astype(int)
        pr = np.stack([X(t), Y(real[:, k])], 1)
        ps = np.stack([X(t), Y(sim[:, k])], 1)
        cv2.polylines(img, [pr], False, col, 3, cv2.LINE_AA)
        cv2.polylines(img, [ps], False, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"top dot {name}", (pad, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    xr = X(np.array([release_t]))[0]
    cv2.line(img, (xr, 0), (xr, h), (0, 0, 220), 1)
    cv2.putText(img, "release", (xr + 3, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    xf = X(np.array([t[f]]))[0]
    cv2.line(img, (xf, 0), (xf, h), (0, 0, 0), 1)
    cv2.putText(img, label, (w - 420, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


def render_clip(name, p):
    clip = load_clip(name)
    model = model_for(clip)
    emb = model.embed(clip.dots_rest)
    X, P = model.simulate_clip(p, clip, emb)
    T = len(clip.t)
    is3d = hasattr(model, "nz")
    if is3d:
        # front face (toward the camera) has the same layout as the 2D grid
        face = SpongeModel(width=model.width, height=model.height, nx=model.nx, ny=model.ny)
        front = lambda xf: xf[model.front][:, :2]
        quads = surface_quads(model)
        R = view_matrix()
        zf = model.depth / 2
        dots3d = lambda f: np.c_[P[f], np.full(len(P[f]), zf + 0.01)]
    else:
        face, front = model, (lambda xf: xf)

    # ---------- static plot of every dot ----------
    os.makedirs(os.path.join(ROOT, "results", "plots"), exist_ok=True)
    nd = clip.dots.shape[1]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for d in range(nd):
        for k, ax in enumerate(axs):
            ax.plot(clip.t, clip.dots[:, d, k], c=f"C{d % 10}", lw=1.2)
            ax.plot(clip.t, P[:, d, k], c=f"C{d % 10}", lw=1, ls="--")
    for ax, lab in zip(axs, ["x, sideways [cm]", "y, height [cm]"]):
        ax.axvline(clip.release_t, c="r", lw=0.8)
        ax.set_ylabel(lab)
    axs[0].set_title(f"{name}: all dots, real (solid) vs simulated (dashed); red line = release")
    axs[1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results", "plots", f"{name}_traces.png"), dpi=110)
    plt.close(fig)

    # ---------- video ----------
    s = clip.px_per_cm
    x0, y0 = clip.origin_px
    crop = (int(x0 - 5.5 * s), int(y0 - 11.5 * s), int(x0 + 5.5 * s), int(y0 + 0.8 * s))
    cw, chh = crop[2] - crop[0], crop[3] - crop[1]
    sc = OUT_H / chh
    pw = int(cw * sc)
    sim_w = pw
    out_dir = os.path.join(ROOT, "results", "demo")
    os.makedirs(out_dir, exist_ok=True)
    W = pw + sim_w
    vw = cv2.VideoWriter(os.path.join(out_dir, f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                         30, (W, OUT_H + PLOT_H))
    cap = cv2.VideoCapture(clip.video)
    real_top = clip.dots[:, :2].mean(1)
    sim_top = P[:, :2].mean(1)
    to_crop = lambda q: (q - np.array(crop[:2])) * sc
    # simulation panel: physical cm -> panel pixels
    sp = OUT_H / 12.3
    to_sim = lambda q: np.stack([sim_w / 2 + q[..., 0] * sp, OUT_H - 0.8 * sp - q[..., 1] * sp], -1)
    for f in range(T):
        ok, frame = cap.read()
        if not ok:
            break
        H0, W0 = frame.shape[:2]
        pad = cv2.copyMakeBorder(frame, max(0, -crop[1]), max(0, crop[3] - H0),
                                 max(0, -crop[0]), max(0, crop[2] - W0), cv2.BORDER_REPLICATE)
        ox, oy = max(0, -crop[0]), max(0, -crop[1])
        real = pad[crop[1] + oy:crop[3] + oy, crop[0] + ox:crop[2] + ox]
        real = cv2.resize(real, (pw, OUT_H), interpolation=cv2.INTER_AREA)
        cam = clip.cam_px[f]                  # camera shift: draw in the raw video frame
        draw_mesh(real, to_crop(to_px(clip, front(X[f])) + cam), face, (255, 200, 0))
        for d in range(nd):
            rp = to_crop(to_px(clip, clip.dots[f, d]) + cam).astype(int)
            spt = to_crop(to_px(clip, P[f, d]) + cam).astype(int)
            cv2.circle(real, tuple(rp), 6, (0, 140, 255), 2)
            cv2.drawMarker(real, tuple(spt), (255, 60, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(real, "REAL video + simulated front face" if is3d else "REAL video + simulated mesh",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(real, "o real dot   x simulated dot", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        simp = np.full((OUT_H, sim_w, 3), 245, np.uint8)
        if is3d:
            s3 = OUT_H / 15.0
            to_scr = lambda q: np.stack([sim_w / 2 + 0.6 * s3 + q[..., 0] * s3,
                                         OUT_H - 2.2 * s3 - q[..., 1] * s3], -1)
            gx = np.array([[-5, 0, -5], [5, 0, -5], [5, 0, 5], [-5, 0, 5]], float) @ R.T
            cv2.fillPoly(simp, [to_scr(gx).astype(np.int32)], (215, 215, 215))
            plate = (clip.plate_c[f], clip.plate_a[f]) if np.isfinite(clip.plate_c[f]) else None
            draw_3d(simp, model, X[f], quads, R, to_scr, plate, dots3d(f))
        else:
            gy = int(OUT_H - 0.8 * sp)
            cv2.line(simp, (0, gy), (sim_w, gy), (90, 90, 90), 2)
        if not is3d and np.isfinite(clip.plate_c[f]):
            xs_ = np.array([-6.0, 6.0])
            ys_ = clip.plate_c[f] - clip.plate_a[f] * xs_
            q = to_sim(np.stack([xs_, ys_], 1)).astype(int)
            cv2.line(simp, tuple(q[0]), tuple(q[1]), (40, 40, 40), 4)
        if not is3d:
            draw_mesh(simp, to_sim(X[f]), model, (0, 120, 200), fill=(80, 220, 240))
            for d in range(nd):
                cv2.drawMarker(simp, tuple(to_sim(P[f, d]).astype(int)), (255, 60, 0),
                               cv2.MARKER_TILTED_CROSS, 10, 2)
        phase = "pressing" if clip.t[f] <= clip.release_t else "FREE (phone released)"
        cv2.putText(simp, ("3D " if is3d else "") + "SIMULATION (input: phone motion only)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(simp, f"t = {clip.t[f]:.3f} s   {phase}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180) if "FREE" in phase else (0, 0, 0), 1)

        strip = plot_strip(W, PLOT_H, clip.t, real_top, sim_top, f, clip.release_t,
                           f"{name}: thick colour = real, thin grey = simulated  (8x slow motion)")
        vw.write(np.vstack([np.hstack([real, simp]), strip]))
    cap.release()
    vw.release()
    print(f"[{name}] wrote results/demo/{name}.mp4 and results/plots/{name}_traces.png")


def main():
    names = sys.argv[1:]
    if not names:
        with open(os.path.join(DATA, "clips.json")) as f:
            names = [k for k in json.load(f) if not k.startswith("_")]
    p = load_params()
    for n in names:
        render_clip(n, p)


if __name__ == "__main__":
    main()
