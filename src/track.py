"""
Track the sponge's marker dots and the presser (phone) in each clip.

For every frame we record:
  * the (x, y) pixel position of every marker dot, with a stable ID
  * the presser's bottom edge, fitted as a line y = a*x + b
  * the sponge's top edge and whether the presser is touching it
  * the camera's shift relative to the first frame (cam_dx, cam_dy), from
    the static strip around the taped base; data.py subtracts it

Usage:
    python src/track.py              # track every clip listed in data/clips.json
    python src/track.py old2 old4    # track only these clips
    python src/track.py --overlay    # also write an overlay video to check tracking

Outputs (per clip):
    data/tracks/<clip>.csv        one row per frame
    data/tracks/<clip>_meta.json  dot layout, image size, fps
    results/tracking/<clip>_overlay.mp4   (with --overlay)
"""
import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ---- colour thresholds (HSV, OpenCV ranges: H 0-179, S/V 0-255) ----------
SPONGE_H = (15, 40)       # yellow
SPONGE_S_MIN = 60
SPONGE_V_MIN = 90
PHONE_V_MAX = 75          # phone body is very dark
PROC_W = 480              # all detection runs at this frame width (px); results are
                          # scaled back to the original resolution
STAB_MIN_RESPONSE = 0.08   # phase-correlation confidence needed to trust a camera-shift estimate
CONTACT_TOL_FRAC = 0.08   # contact if gap <= (min gap of clip) + this * dot spacing


def sponge_mask(hsv, prev_center=None):
    """Binary mask of the sponge (dots filled in), largest yellow blob."""
    h, s, v = cv2.split(hsv)
    m = ((h >= SPONGE_H[0]) & (h <= SPONGE_H[1]) & (s >= SPONGE_S_MIN) & (v >= SPONGE_V_MIN))
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(m)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if prev_center is not None:
        # prefer large blobs close to where the sponge was last frame
        d = np.linalg.norm(cents[1:] - prev_center, axis=1)
        score = areas / (1.0 + d / 50.0)
    else:
        score = areas
    k = 1 + int(np.argmax(score))
    m = (lab == k).astype(np.uint8) * 255
    # close small gaps and fill holes (the dots) so the mask is solid
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(m)
    cv2.drawContours(solid, cnts, -1, 255, thickness=cv2.FILLED)
    return solid


def yellow_raw(hsv):
    h, s, v = cv2.split(hsv)
    return (h >= SPONGE_H[0]) & (h <= SPONGE_H[1]) & (s >= SPONGE_S_MIN) & (v >= SPONGE_V_MIN)


def detect_dots(gray, mask, hsv=None, dot_px=None, use_hull=False):
    """Marker dots = small, roughly round dark spots surrounded by foam.

    Uses a black-hat filter (local darkness compared with the neighbourhood),
    so dots are found even where the foam is shaded. Returns Nx2 centroids."""
    H, W = gray.shape
    k = int(dot_px * 2.5) | 1 if dot_px else max(9, (min(H, W) // 30) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Solid sponge region: a large closing fills the notches the shaded edge
    # and edge-dots cut into the colour mask.
    solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    if use_hull:   # straight sponge at rest: fill edge notches completely
        cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cv2.fillPoly(solid, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 255)
    inside = cv2.distanceTransform((solid > 0).astype(np.uint8), cv2.DIST_L2, 5)
    # Blur the darkness map at roughly the dot scale: round dots become strong
    # peaks, while thin dark streaks (the fuzzy shaded edge) are smeared out.
    sigma = (dot_px / 2.5) if dot_px else k / 5.0
    sm = cv2.GaussianBlur(bh.astype(np.float32), (0, 0), sigma)
    sm[solid == 0] = 0
    # local maxima, at least ~one dot apart
    nms = int(2 * round(1.6 * sigma * 2.5) + 1)
    peaks = (sm == cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms, nms))))
    peaks &= sm > 5.0
    py, px = np.nonzero(peaks)
    if px.size == 0:
        return np.zeros((0, 2))
    val = sm[py, px]
    depth = inside[py, px]
    ok = depth >= max(3.0, 1.2 * sigma)          # not sitting on the outline
    py, px, val = py[ok], px[ok], val[ok]
    if px.size == 0:
        return np.zeros((0, 2))
    ref = np.median(np.sort(val)[::-1][:8])      # typical strength of a real dot
    keep = val >= 0.4 * ref
    pts = []
    for x, y in zip(px[keep], py[keep]):
        # sub-pixel centre: darkness-weighted centroid in a small window
        r = int(max(3, round(1.5 * sigma)))
        y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
        w = sm[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        pts.append((float((w * gx).sum() / w.sum()), float((w * gy).sum() / w.sum())))
    return np.array(pts).reshape(-1, 2)


def order_grid(pts, n_rows, n_cols):
    """Known marker layout (n_rows x n_cols, evenly spaced rows): pick the rows
    that form the most evenly spaced grid, rejecting stray detections (e.g. the
    phone's shadow at the top or the tape edge at the bottom). Returns the
    ordered points (row-major, top->bottom, left->right) or None."""
    from itertools import combinations
    ys = np.sort(pts[:, 1])
    # cluster detections into candidate rows (dots ~10 px, rows ~100 px apart
    # at the fixed working width)
    cut = np.where(np.diff(ys) > 20)[0]
    rows_y = [g.mean() for g in np.split(ys, cut + 1)]
    if len(rows_y) < n_rows:
        return None
    best = None
    for combo in combinations(range(len(rows_y)), n_rows):
        ry = np.array([rows_y[i] for i in combo])
        gaps = np.diff(ry)
        score = gaps.std() / gaps.mean()
        # each row must contain at least n_cols detections
        counts = [np.sum(np.abs(pts[:, 1] - y) < 20) for y in ry]
        if min(counts) < n_cols:
            continue
        if best is None or score < best[0]:
            best = (score, ry)
    if best is None:
        return None
    ry = best[1]
    rows = [pts[np.abs(pts[:, 1] - y) < 20] for y in ry]
    # column positions from rows that have exactly n_cols points
    clean = [r[np.argsort(r[:, 0])] for r in rows if len(r) == n_cols]
    col_x = np.median(np.stack([r[:, 0] for r in clean]), 0) if clean else None
    out = []
    for r in rows:
        if len(r) > n_cols and col_x is not None:
            r = np.array([r[np.argmin(np.abs(r[:, 0] - cx))] for cx in col_x])
        out.append(r[np.argsort(r[:, 0])][:n_cols])
    return np.vstack(out)


def order_dots(pts, n_rows=None, n_cols=None):
    """Give dots stable IDs: sort into rows (top->bottom), then left->right.
    If the marker layout is known (data/sponge.json "dot_grid"), the most
    evenly spaced n_rows x n_cols grid is selected (order_grid). Otherwise:
    points that don't belong to a proper column of the grid (stray detections
    on the sponge edge) are dropped. Returns (ordered points, n_rows, n_cols)."""
    if n_rows and n_cols:
        g = order_grid(pts, n_rows, n_cols)
        if g is not None:
            return g, n_rows, n_cols
    def split_1d(v, frac=0.35):
        """Cluster 1-D values; split at gaps larger than frac * value range."""
        order = np.argsort(v)
        gaps = np.diff(v[order])
        cut = np.where(gaps > frac * max(np.ptp(v), 1e-6) / 2)[0]
        return np.split(order, cut + 1)

    # columns: cluster x positions, keep only well-populated columns
    cols = split_1d(pts[:, 0])
    biggest = max(len(c) for c in cols)
    cols = [c for c in cols if len(c) >= max(3, biggest // 2)]
    cols = sorted(cols, key=lambda c: pts[c, 0].mean())      # left -> right
    ncols = len(cols)
    # rows: the column with the fewest points (no strays) defines the row
    # heights; in every column keep only the point nearest each row height
    ref = min(cols, key=len)
    row_y = np.sort(pts[ref, 1])
    nrows = len(row_y)
    grid = np.zeros((nrows, ncols, 2))
    for j, c in enumerate(cols):
        cp = pts[c]
        for i, y in enumerate(row_y):
            grid[i, j] = cp[np.argmin(np.abs(cp[:, 1] - y))]
    return grid.reshape(-1, 2), nrows, ncols


def phone_line(hsv, sponge, x_lo, x_hi):
    """Fit the phone's bottom edge as y = a*x + b over columns x_lo..x_hi,
    and find the x of the phone's right end (its sideways position).
    Returns (a, b, end_x) or None if the phone isn't visible above the sponge.
    end_x is NaN if the phone's end touches the image border (not visible)."""
    v = hsv[..., 2]
    dark = ((v < PHONE_V_MAX) & (sponge == 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        return None
    # the phone is the biggest dark blob
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 2000:
        return None
    ph = lab == k
    xs, ys = [], []
    for x in range(max(0, x_lo), min(ph.shape[1], x_hi)):
        col = np.nonzero(ph[:, x])[0]
        if col.size:
            xs.append(x)
            ys.append(col.max())
    if len(xs) < 5:
        return None
    a, b = np.polyfit(xs, ys, 1)
    cols = np.nonzero(ph.any(axis=0))[0]
    end_x = float(cols.max()) if cols.max() < ph.shape[1] - 2 else np.nan
    return float(a), float(b), end_x


def track_clip(name, cfg, overlay=False):
    path = os.path.join(DATA, cfg["file"])
    fps = float(cfg["capture_fps"])
    cap = cv2.VideoCapture(path)
    W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Work at a fixed resolution so every pixel-size setting means the same
    # thing for any input video; positions are scaled back at the end.
    sc = W_full / PROC_W
    W, H = PROC_W, int(round(H_full / sc))

    def read():
        ok, f = cap.read()
        return ok, (cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) if ok else None)

    ok, frame = read()
    if not ok:
        raise RuntimeError(f"cannot read {path}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = sponge_mask(hsv)
    dots0 = detect_dots(gray, mask, hsv, use_hull=True)
    if len(dots0) < 4:
        raise RuntimeError(f"{name}: only {len(dots0)} dots found in first frame")
    with open(os.path.join(DATA, "sponge.json")) as f:
        grid = json.load(f).get("dot_grid", {})
    dots0, nrows, ncols = order_dots(dots0, grid.get("rows"), grid.get("cols"))
    ndots = len(dots0)
    ys, xs = np.nonzero(mask)
    sponge_x = (int(xs.min()), int(xs.max()))
    # Camera-motion reference: the static strip around the taped base (tape,
    # its shadow, the table) from just above the base to ~1.5 cm below it.
    base_y = int(ys.max())
    st0, st1 = max(0, base_y - 4), min(H, base_y + 70)
    stab_ref = gray[st0:st1].astype(np.float32)
    stab_win = cv2.createHanningWindow((W, st1 - st0), cv2.CV_32F)
    cam = (0.0, 0.0)
    print(f"[{name}] {W_full}x{H_full}, {nframes} frames @ {fps:.0f} fps, "
          f"{ndots} dots ({nrows} rows x {ncols} cols)")

    # median dot spacing is used as the "snap" radius for re-detection
    spacing = np.median(np.linalg.norm(np.diff(dots0, axis=0), axis=1))
    snap_r = 0.4 * spacing
    dot_px = None   # let the detector estimate dot size each frame

    writer = None
    if overlay:
        out_dir = os.path.join(ROOT, "results", "tracking")
        os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(os.path.join(out_dir, f"{name}_overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    rows = []
    pts = dots0.astype(np.float32).copy()
    valid = np.ones(ndots, bool)
    prev_gray = gray
    center = np.array([xs.mean(), ys.mean()])
    fi = 0
    while True:
        if fi > 0:
            ok, frame = read()
            if not ok:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = sponge_mask(hsv, center)
            # 1) predict every dot with optical flow
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts.reshape(-1, 1, 2), None, **lk)
            nxt = nxt.reshape(-1, 2)
            st = st.ravel().astype(bool)
            # 2) snap each prediction to the nearest detected dark blob
            det = detect_dots(gray, mask, hsv, dot_px=dot_px)
            used = set()
            for i in range(ndots):
                guess = nxt[i] if st[i] else pts[i]
                if len(det):
                    d = np.linalg.norm(det - guess, axis=1)
                    j = int(np.argmin(d))
                    if d[j] < snap_r and j not in used:
                        pts[i] = det[j]
                        used.add(j)
                        valid[i] = True
                        continue
                # no blob nearby: keep flow estimate but mark as unverified
                pts[i] = guess
                valid[i] = bool(st[i]) and mask[int(np.clip(guess[1], 0, H - 1)),
                                                int(np.clip(guess[0], 0, W - 1))] > 0
            ys_, xs_ = np.nonzero(mask)
            if xs_.size:
                center = np.array([xs_.mean(), ys_.mean()])
            prev_gray = gray

        if fi > 0:
            (cdx, cdy), resp = cv2.phaseCorrelate(stab_ref, gray[st0:st1].astype(np.float32), stab_win)
            if resp >= STAB_MIN_RESPONSE:      # unreliable (blur, hand in view): hold last value
                cam = (cdx, cdy)
        ys_, xs_ = np.nonzero(mask)
        top_y = float(ys_.min()) if ys_.size else np.nan
        top_x = float(xs_[ys_ == ys_.min()].mean()) if ys_.size else np.nan
        x_lo = int(min(sponge_x[0], np.nanmin(xs_) if xs_.size else sponge_x[0])) - 10
        x_hi = int(max(sponge_x[1], np.nanmax(xs_) if xs_.size else sponge_x[1])) + 10
        pl = phone_line(hsv, mask, x_lo, x_hi)
        if pl is not None:
            a, b, end_x = pl
            phone_at_top = a * top_x + b
        else:
            a = b = phone_at_top = end_x = np.nan
        gap = top_y - phone_at_top     # px between phone edge and sponge top

        # all positions are stored in ORIGINAL-video pixels
        row = {"frame": fi, "t": fi / fps, "sponge_top_x": top_x * sc, "sponge_top_y": top_y * sc,
               "phone_a": a, "phone_b": b * sc, "phone_y_at_top": phone_at_top * sc,
               "phone_end_x": end_x * sc,
               "gap_px": gap * sc,
               "mask_bottom_y": (float(ys_.max()) if ys_.size else np.nan) * sc,
               "mask_left_x": (float(xs_.min()) if xs_.size else np.nan) * sc,
               "mask_right_x": (float(xs_.max()) if xs_.size else np.nan) * sc,
               "cam_dx": cam[0] * sc, "cam_dy": cam[1] * sc}
        for i in range(ndots):
            row[f"d{i}_x"] = float(pts[i, 0]) * sc
            row[f"d{i}_y"] = float(pts[i, 1]) * sc
            row[f"d{i}_ok"] = int(valid[i])
        rows.append(row)

        if writer is not None:
            vis = frame.copy()
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, (255, 255, 0), 1)
            for i in range(ndots):
                col = (0, 255, 0) if valid[i] else (0, 0, 255)
                cv2.circle(vis, (int(pts[i, 0]), int(pts[i, 1])), 5, col, 1)
                cv2.putText(vis, str(i), (int(pts[i, 0]) + 6, int(pts[i, 1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            if pl is not None:
                cv2.line(vis, (0, int(b)), (W, int(a * W + b)), (255, 0, 255), 1)
            cv2.putText(vis, f"t={fi / fps:.3f}s gap={gap:.0f}px", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            writer.write(vis)
        fi += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Contact: the visible phone edge and the sponge's colour-mask top never
    # meet exactly (perspective, tilt, fuzzy foam edge), and the per-frame gap
    # is noisy (about ±5 px). But during contact it sits at a roughly constant
    # offset, and at release it jumps up sharply and stays up. So we detect two
    # events on a median-smoothed gap, touch-down and release, and call
    # everything in between contact.
    from scipy.signal import medfilt
    gaps = np.array([r["gap_px"] for r in rows], float)
    g = np.where(np.isfinite(gaps), gaps, 1e3)          # phone not visible = far away
    gs = medfilt(g, 9)
    contact_level = float(np.median(gs[gs <= np.percentile(gs, 50)]))
    thr = contact_level + CONTACT_TOL_FRAC * spacing * sc * 2.5
    near = np.nonzero(gs <= thr)[0]
    touch, release = (int(near[0]), int(near[-1])) if near.size else (-1, -1)
    for i, r in enumerate(rows):
        r["contact"] = int(touch <= i <= release) if touch >= 0 else 0

    out = os.path.join(DATA, "tracks")
    os.makedirs(out, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(out, f"{name}.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    meta = {"clip": name, "fps": fps, "width": W_full, "height": H_full, "n_frames": len(rows),
            "n_dots": ndots, "n_rows": nrows, "n_cols": ncols,
            "rest_dots_px": (dots0 * sc).tolist(), "dot_spacing_px": float(spacing * sc),
            "contact_gap_threshold_px": float(thr),
            "contact_level_px": float(contact_level),
            "touch_frame": touch, "release_frame": release,
            "touch_t": touch / fps, "release_t": release / fps}
    with open(os.path.join(out, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    ok_frac = np.mean([[r[f"d{i}_ok"] for i in range(ndots)] for r in rows])
    cfrac = np.mean([r["contact"] for r in rows])
    print(f"[{name}] wrote {len(rows)} rows; dots verified {100 * ok_frac:.1f}% of the time; "
          f"contact frames {touch}-{release} (release at t={release / fps:.3f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", help="clip names from data/clips.json (default: all)")
    ap.add_argument("--overlay", action="store_true", help="write overlay videos for checking")
    args = ap.parse_args()
    with open(os.path.join(DATA, "clips.json")) as f:
        clips = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    for name in args.clips or clips:
        track_clip(name, clips[name], overlay=args.overlay)


if __name__ == "__main__":
    main()
