"""
sem_tile_masking.py

Masks each raw SEM tile of a track *independently*, with no stitching step.
Each tile gets its own local threshold (Otsu / percentile / enhanced-image),
computed from that tile's own pixel statistics rather than a shared
panorama-wide threshold. This sidesteps the brightness-drift and
feature-matching problems that made stitching fragile, at the cost of only
giving you a width estimate per tile rather than one continuous profile.

Physical x-placement of tiles is approximate (nominal tile width x index),
since we're deliberately not doing the overlap-aware alignment that
sem_stitching.py does. Good enough for per-tile QC and a per-tile width
estimate; not a substitute for the stitched panorama if you need sub-tile
positional accuracy.

Usage
-----
    from sem_tile_masking import mask_all_tracks_per_tile, plot_track_tiles

    tile_results = mask_all_tracks_per_tile(
        TRACK_IDS, SEM_DIR, get_sem_tile_paths, load_sem_tile,
        SEM_TILE_WIDTH_MM, method="enhanced", save_dir=FIGURES_DIR,
    )

    for tid in TRACK_IDS:
        plot_track_tiles(tile_results[tid], save_dir=FIGURES_DIR)
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from sem_masking import (
    compute_enhanced_image,
    mask_enhanced_image,
    mask_panorama_horizontal_tophat,
    mask_panorama,
    _make_overlay,
    _to_gray,
)


@dataclass
class TileMaskResult:
    track_id: int
    tile_index: int          # 0-based, in physical x order (0 = 20mm side)
    tile_path: str
    image: np.ndarray        # raw grayscale tile
    mask: np.ndarray         # uint8, 255=track / 0=background -- the raw,
                              # per-column thresholded mask before path cleanup
    coverage_pct: float
    x_lo_mm: float
    x_hi_mm: float
    method: str
    smooth_mask: np.ndarray = None      # uint8, 255/0 -- the single continuous
                                         # laser-path band after noise removal
                                         # + boundary smoothing (None if
                                         # enforce_continuous_path=False)
    top_boundary: np.ndarray = None     # per-column row index of the path's
                                         # top edge, smoothed (px, float, NaN
                                         # where no path found)
    bottom_boundary: np.ndarray = None  # same, bottom edge


# ---------------------------------------------------------------------------
# Continuous-path extraction
#
# The raw per-column threshold mask can include stray bright specks, edge
# artifacts, or scan-line noise that have nothing to do with the laser path.
# The laser path itself is physically a single smooth, continuous band
# running along x. These functions enforce that structure:
#   1. bridge small gaps along x (morphological close, horizontal kernel)
#   2. discard everything except the single largest connected component
#      (removes stray blobs/specks not connected to the main path)
#   3. read off the top/bottom row of that component per column
#   4. smooth those two boundary curves (Savitzky-Golay, falls back to a
#      moving average if scipy isn't available) to remove pixel-level jitter
#   5. rebuild a clean band mask from the smoothed boundaries
# ---------------------------------------------------------------------------

def largest_connected_component(mask, connectivity=8):
    """Keep only the largest connected white region of a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    if num_labels <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def column_boundaries(mask):
    """
    Per-column top/bottom row index of a binary mask.

    Returns
    -------
    top, bottom : float arrays of length = mask.shape[1], NaN where the
                  column has no mask pixels.
    """
    h, w = mask.shape
    top = np.full(w, np.nan)
    bottom = np.full(w, np.nan)
    rows_any = mask > 0
    for j in range(w):
        rows = np.where(rows_any[:, j])[0]
        if rows.size:
            top[j] = rows[0]
            bottom[j] = rows[-1]
    return top, bottom


def _interpolate_nans(y):
    y = y.astype(float).copy()
    idx = np.arange(len(y))
    good = ~np.isnan(y)
    if good.sum() < 2:
        return y
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def smooth_boundary(y, window_px=51, polyorder=3):
    """
    Smooth a per-column boundary curve. Interpolates through any NaN gaps
    first (columns where the path wasn't detected), then applies a
    Savitzky-Golay filter (or a moving average if scipy is unavailable).
    """
    y = _interpolate_nans(y)
    n = len(y)
    window = min(window_px, n if n % 2 == 1 else n - 1)
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1

    if _HAVE_SCIPY:
        po = min(polyorder, window - 1)
        y_smooth = savgol_filter(y, window_length=window, polyorder=po)
    else:
        kernel = np.ones(window) / window
        y_smooth = np.convolve(y, kernel, mode="same")

    return y_smooth


def mask_from_boundaries(top, bottom, shape):
    """Rebuild a band mask from per-column top/bottom row boundaries."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for j in range(w):
        t, b = top[j], bottom[j]
        if np.isnan(t) or np.isnan(b):
            continue
        t_i = int(round(np.clip(t, 0, h - 1)))
        b_i = int(round(np.clip(b, 0, h - 1)))
        if b_i >= t_i:
            mask[t_i:b_i + 1, j] = 255
    return mask


def extract_continuous_track_path(raw_mask, smoothing_window_px=51, close_kernel_px=25):
    """
    Turn a raw per-column threshold mask into a single continuous, smoothed
    laser-path band.

    close_kernel_px      : width of the horizontal closing kernel used to
                            bridge small gaps in the path before component
                            selection. Increase if the path has real dropout
                            gaps that shouldn't fragment it into separate
                            components.
    smoothing_window_px  : window (in columns) for smoothing the top/bottom
                            boundary curves. Larger = smoother edges, but can
                            round off genuine width changes.

    Returns
    -------
    smooth_mask, top_boundary, bottom_boundary
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_px, 1))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    largest = largest_connected_component(closed)
    if not largest.any():
        # nothing survived component selection -- return an empty result
        # rather than silently falling back to the noisy raw mask
        empty = np.full(raw_mask.shape[1], np.nan)
        return np.zeros_like(raw_mask), empty, empty

    top, bottom = column_boundaries(largest)
    top_s = smooth_boundary(top, window_px=smoothing_window_px)
    bottom_s = smooth_boundary(bottom, window_px=smoothing_window_px)
    smooth_mask = mask_from_boundaries(top_s, bottom_s, raw_mask.shape)

    return smooth_mask, top_s, bottom_s


# ---------------------------------------------------------------------------
# Single-edge boundary detection
#
# For tiles where the laser track isn't a band with two edges inside the
# frame, but a single step-like boundary separating "track" (one side) from
# "substrate" (the other side), percentile thresholding on a locally
# enhanced image is the wrong tool -- it reacts to high-frequency texture
# noise everywhere, not the one large-scale transition that matters.
#
# This instead: heavily blurs the tile to suppress speckle/texture noise,
# finds the strongest vertical-gradient edge per column on that blurred
# image, then smooths the resulting boundary curve across columns. The
# output mask/mask-image keeps ORIGINAL pixel values on the track side and
# zeroes the other side -- no flat color fill, so contours/ridges/valleys
# and brightness are preserved.
# ---------------------------------------------------------------------------

def detect_edge_boundary(image, smooth_sigma=15, edge_margin_frac=0.05,
                          smoothing_window_px=101):
    """
    Find a single smooth boundary row per column marking the strongest
    large-scale vertical transition in the tile.

    smooth_sigma      : Gaussian blur sigma (px) applied before edge search.
                         Larger = more robust to texture noise, less able to
                         follow a sharply curving boundary. Start large
                         (~image_height / 10) and reduce if the boundary is
                         cut off or misses real curvature.
    edge_margin_frac   : fraction of image height excluded from the search
                         at the top/bottom (avoids locking onto tile-edge
                         artifacts rather than the real boundary).
    smoothing_window_px : Savitzky-Golay window (columns) for the final
                         boundary smoothing.

    Returns
    -------
    boundary_smoothed : float array (len = image width), row index per column
    strength           : float array, edge strength at that row (for QC --
                         low strength columns mean the boundary is uncertain
                         there)
    """
    img = image.astype(np.float32)
    h, w = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)

    margin = max(int(h * edge_margin_frac), 1)
    search = np.abs(grad_y[margin:h - margin, :])
    boundary_raw = margin + np.argmax(search, axis=0).astype(float)
    strength = search[np.argmax(search, axis=0), np.arange(w)]

    boundary_smoothed = smooth_boundary(boundary_raw, window_px=smoothing_window_px)
    return boundary_smoothed, strength


def mask_original_by_boundary(image, boundary, side="below"):
    """
    Keep original pixel values on one side of a per-column boundary curve,
    zero out the other side. Preserves texture/brightness of the kept
    region exactly -- no threshold, no color fill.

    side : 'below' keeps rows >= boundary[j] (bottom of frame) per column,
           'above' keeps rows < boundary[j] (top of frame) per column.

    Returns
    -------
    masked_image (same dtype as image, 0 outside track),
    mask (uint8, 255/0)
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for j in range(w):
        b = int(round(np.clip(boundary[j], 0, h - 1)))
        if side == "below":
            mask[b:, j] = 255
        elif side == "above":
            mask[:b, j] = 255
        else:
            raise ValueError("side must be 'below' or 'above'")

    masked_image = image.copy()
    masked_image[mask == 0] = 0
    return masked_image, mask


def preview_boundary_sides(image, smooth_sigma=15, edge_margin_frac=0.05,
                            smoothing_window_px=101, save_path=None):
    """
    Calibration helper: run detect_edge_boundary on one tile and show the
    original image with the detected boundary drawn, plus both possible
    masked-original outputs (keep-above vs keep-below) side by side, so you
    can tell which side is the actual laser track before running this
    across every tile/track.
    """
    boundary, strength = detect_edge_boundary(
        image, smooth_sigma=smooth_sigma, edge_margin_frac=edge_margin_frac,
        smoothing_window_px=smoothing_window_px,
    )
    above_img, _ = mask_original_by_boundary(image, boundary, side="above")
    below_img, _ = mask_original_by_boundary(image, boundary, side="below")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(image, cmap="gray")
    xs = np.arange(image.shape[1])
    axes[0].plot(xs, boundary, color="red", linewidth=1.5)
    axes[0].set_title("original + detected boundary")
    axes[0].axis("off")

    axes[1].imshow(above_img, cmap="gray")
    axes[1].set_title("side='above' (keeps top)")
    axes[1].axis("off")

    axes[2].imshow(below_img, cmap="gray")
    axes[2].set_title("side='below' (keeps bottom)")
    axes[2].axis("off")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  saved -> {save_path}")

    plt.show()
    return fig, boundary, strength


# ---------------------------------------------------------------------------
# Texture-roughness band detection
#
# On real tiles, the melt track shows up as a locally *smoother* patch
# against a rougher substrate/powder-bed texture -- a more reliable signal
# here than raw brightness gradient, which locks onto dust/particle specks
# instead of the real track edge. This detects the track as a band (top AND
# bottom boundary) using per-column normalized roughness, so it isn't
# thrown off by any left-to-right brightness/texture drift across the tile.
#
# Output is drawn as an OUTLINE on the untouched original image -- nothing
# is zeroed out or replaced. The boundary arrays are also returned so you
# can extract a masked copy separately if/when you actually want that.
# ---------------------------------------------------------------------------

def local_roughness(image, ksize=21):
    """Local intensity std-dev per pixel (texture roughness), via box filter."""
    img = image.astype(np.float32)
    mean = cv2.blur(img, (ksize, ksize))
    sqmean = cv2.blur(img * img, (ksize, ksize))
    return np.sqrt(np.clip(sqmean - mean * mean, 0, None))


def _dp_smooth_centerline(cost, max_jump=4, jump_penalty=5.0):
    """
    Dynamic-programming shortest-path through a per-pixel cost map (h x w),
    one row choice per column, penalizing row-to-row jumps quadratically.
    Finds the single lowest-cost path that's also smooth -- much more
    robust than thresholding pixels and hoping they stay one connected
    blob, which merges unrelated low-cost patches together.

    jump_penalty : higher = straighter/more rigid path, more resistant to
                   being pulled off by an unrelated nearby low-cost patch.
                   Lower = follows the cost map more literally, including
                   real local wandering.
    """
    h, w = cost.shape
    dp = np.full((h, w), np.inf)
    parent = np.zeros((h, w), dtype=np.int32)
    dp[:, 0] = cost[:, 0]
    parent[:, 0] = np.arange(h)
    offsets = np.arange(-max_jump, max_jump + 1)
    row_idx = np.arange(h)
    for j in range(1, w):
        candidates = np.full((len(offsets), h), np.inf)
        for k, off in enumerate(offsets):
            shifted = np.roll(dp[:, j - 1], off)
            if off > 0:
                shifted[:off] = np.inf
            elif off < 0:
                shifted[off:] = np.inf
            candidates[k] = shifted + jump_penalty * (off ** 2)
        best_k = np.argmin(candidates, axis=0)
        dp[:, j] = candidates[best_k, row_idx] + cost[:, j]
        parent[:, j] = row_idx - offsets[best_k]

    path = np.zeros(w, dtype=np.int32)
    path[-1] = int(np.argmin(dp[:, -1]))
    for j in range(w - 2, -1, -1):
        path[j] = parent[path[j + 1], j + 1]
    return path


def detect_smooth_band(image, roughness_ksize=21, centerline_jump_penalty=5.0,
                        centerline_max_jump=3, expand_margin=0.75,
                        max_half_width_px=120, smoothing_window_px=61):
    """
    Find the laser track as a smooth band using texture roughness:
      1. per-column z-score of local roughness (normalizes away any
         left-right drift in absolute roughness/brightness)
      2. a DP centerline through the z-score map -- the single smooth path
         of lowest (smoothest) cost, resistant to being pulled onto an
         unrelated nearby smooth patch the way naive thresholding is
      3. per-column expansion up/down from that centerline, stopping once
         roughness rises expand_margin above the centerline's own z-value
         there (relative, not an absolute cutoff -- using the column mean
         as a cutoff lets expansion run through ~50% of the column by
         definition, which is the runaway-width bug this replaces) and
         capped at max_half_width_px as a hard safety limit
      4. light smoothing of the resulting top/bottom curves, with top/
         bottom clamped so smoothing can't make them cross

    Returns
    -------
    top_boundary, bottom_boundary, center_boundary : float arrays (len =
        image width)
    """
    rough = local_roughness(image, ksize=roughness_ksize)
    col_mean = rough.mean(axis=0, keepdims=True)
    col_std = rough.std(axis=0, keepdims=True) + 1e-6
    z = (rough - col_mean) / col_std

    center = _dp_smooth_centerline(
        z, max_jump=centerline_max_jump, jump_penalty=centerline_jump_penalty
    )

    h, w = z.shape
    top = np.zeros(w, dtype=float)
    bottom = np.zeros(w, dtype=float)
    for j in range(w):
        c = center[j]
        thresh = z[c, j] + expand_margin
        t_limit = max(0, c - max_half_width_px)
        b_limit = min(h - 1, c + max_half_width_px)
        t = c
        while t > t_limit and z[t - 1, j] < thresh:
            t -= 1
        b = c
        while b < b_limit and z[b + 1, j] < thresh:
            b += 1
        top[j] = t
        bottom[j] = b

    top = smooth_boundary(top, window_px=smoothing_window_px)
    bottom = smooth_boundary(bottom, window_px=smoothing_window_px)
    center_smoothed = smooth_boundary(center.astype(float), window_px=smoothing_window_px)
    # smoothing top/bottom independently can occasionally let them cross on
    # a very thin/noisy band -- clamp back to a minimum 1px separation
    cross = top > bottom
    if cross.any():
        mid = (top[cross] + bottom[cross]) / 2
        top[cross] = mid - 0.5
        bottom[cross] = mid + 0.5

    return top, bottom, center_smoothed


def draw_band_outline(image, top_boundary, bottom_boundary, valid=None,
                       top_color=(0, 255, 0), bottom_color=(255, 0, 0),
                       thickness=2):
    """
    Draw the detected band's top/bottom boundary as an outline on top of
    the UNMODIFIED original image -- nothing is masked out or replaced, the
    full original image is preserved, only the boundary curves are added.

    Returns a BGR color image (uint8).
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    w = image.shape[1]
    for boundary, color in [(top_boundary, top_color), (bottom_boundary, bottom_color)]:
        pts = []
        for j in range(w):
            if valid is not None and not valid[j]:
                if len(pts) > 1:
                    cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, color, thickness)
                pts = []
                continue
            if np.isnan(boundary[j]):
                continue
            pts.append([j, int(round(boundary[j]))])
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, color, thickness)

    return vis


def preview_smooth_band(image, roughness_ksize=21, centerline_jump_penalty=5.0,
                         centerline_max_jump=3, expand_margin=0.75,
                         max_half_width_px=120, smoothing_window_px=61,
                         save_path=None):
    """
    Calibration helper: run detect_smooth_band on one tile and show the
    original image (fully intact) with the detected band outlined in
    green (top) / red (bottom) / yellow (centerline), next to the
    roughness map that drove the detection.
    """
    top, bottom, center = detect_smooth_band(
        image, roughness_ksize=roughness_ksize,
        centerline_jump_penalty=centerline_jump_penalty,
        centerline_max_jump=centerline_max_jump,
        expand_margin=expand_margin,
        max_half_width_px=max_half_width_px,
        smoothing_window_px=smoothing_window_px,
    )
    outline = draw_band_outline(image, top, bottom)
    xs = np.arange(image.shape[1])
    cv2.polylines(outline, [np.stack([xs, center.astype(np.int32)], axis=1)],
                  False, (0, 255, 255), 1)
    rough = local_roughness(image, ksize=roughness_ksize)

    width_px = bottom - top
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(outline, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"detected band (green=top, red=bottom, yellow=centerline)\n"
        f"width px: mean={width_px.mean():.0f} std={width_px.std():.0f}"
    )
    axes[0].axis("off")

    im = axes[1].imshow(rough, cmap="viridis")
    axes[1].plot(xs, center, color="yellow", linewidth=1)
    axes[1].set_title(f"local roughness (ksize={roughness_ksize}) + centerline")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.03)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  saved -> {save_path}")

    plt.show()
    return fig, top, bottom, center


def load_track_tiles(
    track_id,
    sem_dir,
    get_sem_tile_paths,
    load_sem_tile,
    tile_width_mm,
    flip_order=True,
    rotate_180=True,
    x_start_mm=20.0,
):
    """
    Load every raw tile for one track, in physical x order, tagged with an
    approximate x-extent per tile.

    flip_order   : per dataset_readme.txt, tile01 is captured on the 100mm
                   side and the highest-numbered tile is on the 20mm side --
                   the opposite of increasing physical x. Reverses the
                   natural-sort tile list so index 0 is the 20mm-side tile,
                   matching the convention used in sem_stitching.py.
    rotate_180   : applies the same pixel-orientation correction used before
                   stitching, so track position within each tile (top/bottom,
                   left/right) is consistent across tiles.
    x_start_mm   : physical x of the left edge of the first (20mm-side) tile.

    Returns
    -------
    list of dicts: {index, path, image, x_lo_mm, x_hi_mm}
    """
    try:
        paths = get_sem_tile_paths(sem_dir, track_id)
    except TypeError:
        paths = get_sem_tile_paths(track_id)

    if not paths:
        print(f"[load_track_tiles] track {track_id}: no tiles found")
        return []

    if flip_order:
        paths = list(reversed(paths))

    tiles = []
    for i, p in enumerate(paths):
        img = load_sem_tile(p)
        if rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        x_lo = x_start_mm + i * tile_width_mm
        x_hi = x_lo + tile_width_mm
        tiles.append({
            "index": i,
            "path": str(p),
            "image": img,
            "x_lo_mm": x_lo,
            "x_hi_mm": x_hi,
        })

    return tiles


def mask_tile(
    image,
    method="enhanced",
    threshold_percentile=95,
    enforce_continuous_path=True,
    smoothing_window_px=51,
    close_kernel_px=25,
    **kwargs,
):
    """
    Mask a single tile using its own statistics.

    method : 'enhanced'  - CLAHE + horizontal top-hat + blend, then
                            percentile threshold (mask_enhanced_image).
             'tophat'     - CLAHE + horizontal top-hat, percentile threshold
                            directly on the top-hat response
                            (mask_panorama_horizontal_tophat).
             'otsu'       - global Otsu split (mask_panorama); ignores
                            threshold_percentile.

    enforce_continuous_path : if True (default), post-process the raw
        threshold mask with extract_continuous_track_path -- keeps only the
        single largest connected band and smooths its top/bottom edges, so
        the result is one continuous laser path rather than raw thresholded
        blobs. Set False to get the unprocessed per-column mask instead.

    Returns
    -------
    raw_mask (uint8, 255/0), coverage_pct (float),
    smooth_mask (uint8 or None), top_boundary (float array or None),
    bottom_boundary (float array or None)
    """
    if method == "enhanced":
        enhanced, *_ = compute_enhanced_image(image, **kwargs)
        raw_mask, coverage_pct = mask_enhanced_image(
            enhanced, threshold_percentile=threshold_percentile
        )
    elif method == "tophat":
        raw_mask, coverage_pct, _, _ = mask_panorama_horizontal_tophat(
            image, threshold_percentile=threshold_percentile, **kwargs
        )
    elif method == "otsu":
        result = mask_panorama(image, method="otsu", **kwargs)
        raw_mask = (result.mask.astype(np.uint8)) * 255
        coverage_pct = 100 * result.coverage_frac
    else:
        raise ValueError("method must be 'enhanced', 'tophat', or 'otsu'")

    if not enforce_continuous_path:
        return raw_mask, coverage_pct, None, None, None

    smooth_mask, top_b, bottom_b = extract_continuous_track_path(
        raw_mask, smoothing_window_px=smoothing_window_px,
        close_kernel_px=close_kernel_px,
    )
    return raw_mask, coverage_pct, smooth_mask, top_b, bottom_b


def mask_track_tiles(
    track_id,
    tiles,
    method="enhanced",
    threshold_percentile=95,
    enforce_continuous_path=True,
    smoothing_window_px=51,
    close_kernel_px=25,
    save_dir=None,
    **kwargs,
):
    """
    Run mask_tile independently on every tile in `tiles` (from
    load_track_tiles), each with its own threshold, then (by default)
    collapse each tile's raw mask down to a single continuous, smoothed
    laser-path band.

    Returns
    -------
    list of TileMaskResult, in the same order as `tiles`
    """
    results = []
    for t in tiles:
        raw_mask, coverage_pct, smooth_mask, top_b, bottom_b = mask_tile(
            t["image"], method=method,
            threshold_percentile=threshold_percentile,
            enforce_continuous_path=enforce_continuous_path,
            smoothing_window_px=smoothing_window_px,
            close_kernel_px=close_kernel_px,
            **kwargs,
        )
        r = TileMaskResult(
            track_id=track_id,
            tile_index=t["index"],
            tile_path=t["path"],
            image=t["image"],
            mask=raw_mask,
            coverage_pct=coverage_pct,
            x_lo_mm=t["x_lo_mm"],
            x_hi_mm=t["x_hi_mm"],
            method=method,
            smooth_mask=smooth_mask,
            top_boundary=top_b,
            bottom_boundary=bottom_b,
        )
        results.append(r)

        if smooth_mask is not None:
            path_cov = 100 * (smooth_mask > 0).mean()
            print(f"Track {track_id}, tile {t['index']:02d} "
                  f"[{t['x_lo_mm']:.2f}-{t['x_hi_mm']:.2f} mm]: "
                  f"raw coverage={coverage_pct:.2f}%, "
                  f"path coverage={path_cov:.2f}%")
        else:
            print(f"Track {track_id}, tile {t['index']:02d} "
                  f"[{t['x_lo_mm']:.2f}-{t['x_hi_mm']:.2f} mm]: "
                  f"coverage={coverage_pct:.2f}%")

        if save_dir is not None:
            save_dir_p = Path(save_dir)
            save_dir_p.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(save_dir_p / f"track_{track_id}_tile_{t['index']:02d}_mask_raw.png"),
                raw_mask,
            )
            if smooth_mask is not None:
                cv2.imwrite(
                    str(save_dir_p / f"track_{track_id}_tile_{t['index']:02d}_mask_path.png"),
                    smooth_mask,
                )

    return results


def mask_all_tracks_per_tile(
    track_ids,
    sem_dir,
    get_sem_tile_paths,
    load_sem_tile,
    tile_width_mm,
    method="enhanced",
    threshold_percentile=95,
    enforce_continuous_path=True,
    smoothing_window_px=51,
    close_kernel_px=25,
    flip_order=True,
    rotate_180=True,
    x_start_mm=20.0,
    save_dir=None,
    **kwargs,
):
    """
    Convenience wrapper: load + mask every tile of every track, no stitching.

    Returns
    -------
    dict {track_id: list of TileMaskResult}
    """
    all_results = {}
    for tid in track_ids:
        print(f"\n--- Track {tid}: per-tile masking (method={method}) ---")
        tiles = load_track_tiles(
            tid, sem_dir, get_sem_tile_paths, load_sem_tile, tile_width_mm,
            flip_order=flip_order, rotate_180=rotate_180, x_start_mm=x_start_mm,
        )
        all_results[tid] = mask_track_tiles(
            tid, tiles, method=method,
            threshold_percentile=threshold_percentile,
            enforce_continuous_path=enforce_continuous_path,
            smoothing_window_px=smoothing_window_px,
            close_kernel_px=close_kernel_px,
            save_dir=save_dir,
            **kwargs,
        )
    return all_results


def plot_track_tiles(tile_results, ncols=6, save_dir=None):
    """
    Grid plot: every tile of a track with its mask overlaid in red, labeled
    with its approximate x-extent and coverage -- a quick visual QC pass
    across all tiles in a track without stitching them together.
    """
    n = len(tile_results)
    if n == 0:
        print("[plot_track_tiles] no tiles to plot")
        return None

    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    track_id = tile_results[0].track_id
    for ax, r in zip(axes, tile_results):
        display_mask = r.smooth_mask if r.smooth_mask is not None else r.mask
        overlay = _make_overlay(r.image, display_mask > 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ax.imshow(overlay_rgb)
        cov_label = (100 * (display_mask > 0).mean()) if r.smooth_mask is not None else r.coverage_pct
        ax.set_title(
            f"tile {r.tile_index:02d} [{r.x_lo_mm:.1f}-{r.x_hi_mm:.1f}mm]\n"
            f"cov={cov_label:.1f}%", fontsize=8,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Track {track_id}: per-tile masks ({tile_results[0].method})",
                 fontsize=12)

    if save_dir is not None:
        save_dir_p = Path(save_dir)
        save_dir_p.mkdir(parents=True, exist_ok=True)
        out_path = save_dir_p / f"track_{track_id}_tiles_masked_grid.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  saved -> {out_path}")

    plt.show()
    return fig


def tile_width_profile(tile_result, px_to_mm_y=None):
    """
    Derive a local width profile W(x) from a single tile, in the same spirit
    as sem_masking.mask_width_profile but scoped to one tile.

    Prefers the smoothed top/bottom boundaries (from enforce_continuous_path)
    when available -- these already reflect the single continuous laser path
    with jitter removed, so width_mm = (bottom - top) directly rather than
    re-scanning the mask. Falls back to scanning tile_result.mask if the
    tile wasn't run with enforce_continuous_path=True.

    px_to_mm_y : mm per pixel along the tile's vertical axis. If None,
                 assumes square pixels using the tile's own width_mm / width_px
                 (SEM tile width is fixed at tile_width_mm, so this is exact
                 as long as the tile has square pixels -- true for the Bruker
                 SEM images here).

    Returns
    -------
    dict with keys: x (mm), width_mm, coverage_frac_per_column
    """
    h, w = tile_result.image.shape[:2]
    tile_width_mm = tile_result.x_hi_mm - tile_result.x_lo_mm
    px_to_mm_x = tile_width_mm / w
    if px_to_mm_y is None:
        px_to_mm_y = px_to_mm_x

    x_mm = tile_result.x_lo_mm + (np.arange(w) + 0.5) * px_to_mm_x

    if tile_result.top_boundary is not None and tile_result.bottom_boundary is not None:
        top, bottom = tile_result.top_boundary, tile_result.bottom_boundary
        width_mm = (bottom - top) * px_to_mm_y
        coverage = (tile_result.smooth_mask > 0).mean(axis=0)
        return {"x": x_mm, "width_mm": width_mm, "coverage_frac_per_column": coverage}

    mask = tile_result.mask > 0
    width_mm = np.full(w, np.nan)
    coverage = mask.mean(axis=0)
    for j in range(w):
        col = mask[:, j]
        if not col.any():
            continue
        rows = np.where(col)[0]
        width_mm[j] = (rows[-1] - rows[0]) * px_to_mm_y

    return {"x": x_mm, "width_mm": width_mm, "coverage_frac_per_column": coverage}


def track_width_profile(tile_results, px_to_mm_y=None):
    """
    Concatenate per-tile width profiles for a whole track into one array,
    in physical x order. Since tiles are placed at nominal (non-overlap-
    corrected) spacing, treat this as an approximate profile -- there may be
    small x-discontinuities at tile boundaries vs. the true stitched
    geometry.

    Returns
    -------
    dict with keys: x (mm), width_mm, coverage_frac_per_column (all
    concatenated across tiles, sorted by tile_index)
    """
    ordered = sorted(tile_results, key=lambda r: r.tile_index)
    xs, widths, covs = [], [], []
    for r in ordered:
        p = tile_width_profile(r, px_to_mm_y=px_to_mm_y)
        xs.append(p["x"])
        widths.append(p["width_mm"])
        covs.append(p["coverage_frac_per_column"])

    return {
        "x": np.concatenate(xs),
        "width_mm": np.concatenate(widths),
        "coverage_frac_per_column": np.concatenate(covs),
    }


# ---------------------------------------------------------------------------
# Output formats for a detected band (top_boundary/bottom_boundary from
# detect_smooth_band). Pick whichever fits how you're using the result --
# they all come from the same detection, just displayed/exported differently.
# ---------------------------------------------------------------------------

def band_mask(image_shape, top_boundary, bottom_boundary):
    """Binary mask (uint8, 255=track/0=background) from band boundaries."""
    return mask_from_boundaries(top_boundary, bottom_boundary, image_shape)


def tint_band_overlay(image, top_boundary, bottom_boundary,
                       color=(0, 200, 0), alpha=0.35):
    """
    Original image, fully visible everywhere, with the detected track band
    semi-transparently tinted -- texture underneath stays visible, nothing
    is zeroed out or replaced. Good for a quick visual "is this right"
    check across many tiles.

    color : BGR tuple for the tint.
    alpha : 0=invisible tint (original untouched), 1=solid fill.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    mask = band_mask(image.shape[:2], top_boundary, bottom_boundary) > 0
    overlay = vis.copy()
    overlay[mask] = color
    tinted = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return tinted


def crop_band_only(image, top_boundary, bottom_boundary):
    """
    Original pixel values inside the detected band, zero everywhere else --
    a clean extraction of just the track region with contours/ridges/
    brightness fully preserved where kept. This is the one that actually
    discards the rest of the image; use tint or outline instead if you want
    everything visible.
    """
    mask = band_mask(image.shape[:2], top_boundary, bottom_boundary)
    out = image.copy()
    out[mask == 0] = 0
    return out, mask


def generate_band_outputs(image, roughness_ksize=21, centerline_jump_penalty=5.0,
                           centerline_max_jump=3, expand_margin=0.75,
                           max_half_width_px=120, smoothing_window_px=61):
    """
    Run detect_smooth_band once and return every output format together, so
    you can compare them side by side or pick per use case downstream.

    Returns
    -------
    dict with keys:
      top, bottom, center       : the raw boundary arrays (float, px)
      mask                      : uint8 binary mask (255=track)
      outline_image             : original + thin boundary lines (BGR)
      tint_image                : original + semi-transparent band tint (BGR)
      cropped_image, cropped_mask : original pixels inside band only,
                                     zero elsewhere + the mask used
    """
    top, bottom, center = detect_smooth_band(
        image, roughness_ksize=roughness_ksize,
        centerline_jump_penalty=centerline_jump_penalty,
        centerline_max_jump=centerline_max_jump,
        expand_margin=expand_margin,
        max_half_width_px=max_half_width_px,
        smoothing_window_px=smoothing_window_px,
    )
    mask = band_mask(image.shape[:2], top, bottom)
    outline_image = draw_band_outline(image, top, bottom)
    tint_image = tint_band_overlay(image, top, bottom)
    cropped_image, cropped_mask = crop_band_only(image, top, bottom)

    return {
        "top": top, "bottom": bottom, "center": center,
        "mask": mask,
        "outline_image": outline_image,
        "tint_image": tint_image,
        "cropped_image": cropped_image, "cropped_mask": cropped_mask,
    }


def preview_all_band_outputs(image, save_path=None, **kwargs):
    """
    Show all three output formats side by side on one tile: outline, tint,
    and crop-only. Same detection, three views -- use this to decide which
    format you actually want before running across every tile/track.
    """
    out = generate_band_outputs(image, **kwargs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(out["outline_image"], cv2.COLOR_BGR2RGB))
    axes[0].set_title("outline (lines on untouched original)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(out["tint_image"], cv2.COLOR_BGR2RGB))
    axes[1].set_title("tint (semi-transparent, texture visible)")
    axes[1].axis("off")

    axes[2].imshow(out["cropped_image"], cmap="gray")
    axes[2].set_title("cropped (track pixels only, rest blank)")
    axes[2].axis("off")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  saved -> {save_path}")

    plt.show()
    return fig, out
