import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sem_tile_masking import load_track_tiles, detect_smooth_band


# ---------------------------------------------------------------------------
# 1. Rectangular box mask
#
# Takes the same top/bottom boundary arrays detect_smooth_band already
# produces, then collapses them to a single box: the minimum of the top
# boundary and the maximum of the bottom boundary across every column, so
# the box is guaranteed to fully contain the path even where it drifts up
# or down across the tile.
# ---------------------------------------------------------------------------

def rectangular_box_from_boundaries(image_shape, top, bottom, box_mode="envelope",
                                     lower_pct=2.0, upper_pct=98.0):
    """
    Returns (y_min, y_max) for a single full-width box.

    box_mode="envelope" (default) uses the strict min(top) / max(bottom)
    across every column, so the box is mathematically guaranteed to fully
    contain the path everywhere along x, it can never cut into the true
    path, only ever match or exceed it. If the path's width varies a lot
    along x (a wide splash/blob in one spot, a narrow stretch elsewhere),
    this box has to be as tall as the widest point, so at narrower columns
    the extra rows inside the box are genuinely substrate, not path, this
    is expected, not a bug.

    box_mode="percentile" instead uses the lower_pct/upper_pct percentiles
    of top/bottom across columns, giving a tighter box that better matches
    the path's typical width, at the cost of possibly clipping a small
    fraction of columns where the path bulges out further than that
    percentile. Use this if you'd rather have a cleaner box than a
    guaranteed-complete one.
    """
    h, w = image_shape
    if box_mode == "envelope":
        y_min = np.nanmin(top)
        y_max = np.nanmax(bottom)
    elif box_mode == "percentile":
        y_min = np.nanpercentile(top, lower_pct)
        y_max = np.nanpercentile(bottom, upper_pct)
    else:
        raise ValueError("box_mode must be 'envelope' or 'percentile'")

    y_min = max(0, int(np.floor(y_min)))
    y_max = min(h - 1, int(np.ceil(y_max)))
    return y_min, y_max


def apply_rectangular_mask(image, y_min, y_max, mode="cover_track"):
    """
    mode="cover_track" (default): blacks out rows [y_min, y_max] (the
    track) and leaves everything outside the box untouched at its
    original pixel values, this is the one that covers the track.

    mode="keep_track": the old behavior, original pixel values inside
    [y_min, y_max], zero everywhere else, kept only in case you want it
    back.
    """
    h, w = image.shape[:2]
    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[y_min:y_max + 1, :] = 255

    out = image.copy()
    if mode == "cover_track":
        out[box_mask > 0] = 0
    elif mode == "keep_track":
        out[box_mask == 0] = 0
    else:
        raise ValueError("mode must be 'cover_track' or 'keep_track'")

    return out, box_mask


def generate_rectangular_outputs(image, box_mode="envelope", lower_pct=2.0,
                                  upper_pct=98.0, mask_mode="cover_track",
                                  **band_kwargs):
    """Runs detect_smooth_band, then reduces it to one rectangular box.
    See rectangular_box_from_boundaries for box_mode options, and
    apply_rectangular_mask for mask_mode options (default covers the
    track and leaves the rest of the tile untouched)."""
    top, bottom, center = detect_smooth_band(image, **band_kwargs)
    y_min, y_max = rectangular_box_from_boundaries(
        image.shape[:2], top, bottom, box_mode=box_mode,
        lower_pct=lower_pct, upper_pct=upper_pct,
    )
    masked_image, mask = apply_rectangular_mask(image, y_min, y_max, mode=mask_mode)

    # columns where the true band pokes outside the box, only possible in
    # percentile mode, always zero in envelope mode by construction
    clipped_top = np.nansum(top < y_min)
    clipped_bottom = np.nansum(bottom > y_max)
    w = image.shape[1]
    clipped_frac = float((clipped_top + clipped_bottom) / max(w, 1))

    return {
        "top": top, "bottom": bottom, "center": center,
        "y_min": y_min, "y_max": y_max,
        "mask": mask, "masked_image": masked_image,
        "box_mode": box_mode, "mask_mode": mask_mode, "clipped_frac": clipped_frac,
    }


# ---------------------------------------------------------------------------
# 2. Path exclusion
#
# Instead of keeping the path and dropping the substrate (crop_band_only)
# or keeping everything and just outlining the path (draw_band_outline),
# this drops the path itself and keeps the two substrate regions on either
# side of it: everything above the green top boundary, and everything
# below the blue bottom boundary.
# ---------------------------------------------------------------------------

def band_exclusion_masks(image_shape, top, bottom):
    """Per-column masks for the region strictly above the top boundary and
    the region strictly below the bottom boundary. The rows in between
    (the path itself) are excluded from both."""
    h, w = image_shape
    above_mask = np.zeros((h, w), dtype=np.uint8)
    below_mask = np.zeros((h, w), dtype=np.uint8)
    for j in range(w):
        t = int(round(np.clip(top[j], 0, h - 1)))
        b = int(round(np.clip(bottom[j], 0, h - 1)))
        above_mask[:t, j] = 255
        below_mask[b + 1:, j] = 255
    return above_mask, below_mask


def apply_exclusion(image, above_mask, below_mask):
    """Original pixel values kept in the above/below regions, path rows
    (and the opposite region) zeroed in each."""
    above_image = image.copy()
    above_image[above_mask == 0] = 0
    below_image = image.copy()
    below_image[below_mask == 0] = 0
    combined_mask = np.maximum(above_mask, below_mask)
    combined_image = image.copy()
    combined_image[combined_mask == 0] = 0
    return above_image, below_image, combined_image, combined_mask


def generate_exclusion_outputs(image, **band_kwargs):
    """Runs detect_smooth_band, then splits the tile into the two
    substrate regions on either side of the path, with the path itself
    dropped from both."""
    top, bottom, center = detect_smooth_band(image, **band_kwargs)
    above_mask, below_mask = band_exclusion_masks(image.shape[:2], top, bottom)
    above_image, below_image, combined_image, combined_mask = apply_exclusion(
        image, above_mask, below_mask
    )
    return {
        "top": top, "bottom": bottom, "center": center,
        "above_image": above_image, "below_image": below_image,
        "above_mask": above_mask, "below_mask": below_mask,
        "combined_excluded_image": combined_image, "combined_mask": combined_mask,
    }


# ---------------------------------------------------------------------------
# batch runners, mirror run_smooth_band_masking's grid + json pattern
# ---------------------------------------------------------------------------

def run_rectangular_box_masking(
    TRACK_IDS, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM,
    FIGURES_DIR, METADATA_DIR, n_show=6, ncols=6, box_mode="envelope",
    lower_pct=2.0, upper_pct=98.0, mask_mode="cover_track",
    show_boundary_lines=True, band_kwargs=None,
):
    """
    box_mode="envelope" (default) guarantees the box fully contains the
    path everywhere, box_mode="percentile" gives a tighter box that may
    clip rare bulges, see rectangular_box_from_boundaries for details.

    mask_mode="cover_track" (default) blacks out the box (the track) and
    leaves the rest of the tile at its original pixel values. Pass
    mask_mode="keep_track" for the reverse (crop to just the track).

    show_boundary_lines draws the green top / blue bottom boundary on top
    of the masked image, so you can see directly that the box always sits
    at or outside those lines, whatever texture shows up inside the box
    at a given column is genuinely between those two lines there.
    """
    band_kwargs = band_kwargs or {}
    FIGURES_DIR = Path(FIGURES_DIR)
    METADATA_DIR = Path(METADATA_DIR)
    summary = {}

    for track_id in TRACK_IDS:
        tiles = load_track_tiles(
            track_id, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM
        )
        if not tiles:
            continue

        n = len(tiles) if n_show is None else min(n_show, len(tiles))
        cols = min(ncols, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.4 * rows),
                                  constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        track_records = []
        for ax, t in zip(axes, tiles[:n]):
            out = generate_rectangular_outputs(
                t["image"], box_mode=box_mode, lower_pct=lower_pct,
                upper_pct=upper_pct, mask_mode=mask_mode, **band_kwargs,
            )
            h, w = t["image"].shape[:2]
            px_to_mm = SEM_TILE_WIDTH_MM / w

            ax.imshow(out["masked_image"], cmap="gray")
            if show_boundary_lines:
                xs = np.arange(w)
                ax.plot(xs, out["top"], color="lime", linewidth=1)
                ax.plot(xs, out["bottom"], color="blue", linewidth=1)

            y_min_mm = out["y_min"] * px_to_mm
            y_max_mm = out["y_max"] * px_to_mm
            ax.set_title(
                f"tile {t['index']:02d} [{t['x_lo_mm']:.1f}-{t['x_hi_mm']:.1f}mm]\n"
                f"box y=[{y_min_mm:.2f},{y_max_mm:.2f}]mm "
                f"clipped={out['clipped_frac']*100:.0f}%", fontsize=8,
            )
            ax.axis("off")

            track_records.append({
                "tile_index": t["index"], "tile_path": t["path"],
                "x_lo_mm": t["x_lo_mm"], "x_hi_mm": t["x_hi_mm"],
                "y_min_px": out["y_min"], "y_max_px": out["y_max"],
                "y_min_mm": float(y_min_mm), "y_max_mm": float(y_max_mm),
                "box_height_mm": float(y_max_mm - y_min_mm),
                "box_mode": box_mode, "clipped_frac": out["clipped_frac"],
            })

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(f"Track {track_id}: rectangular box mask", fontsize=12)
        fig.savefig(FIGURES_DIR / f"sem_rect_mask_track_{track_id}.png",
                    dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / f"sem_rect_mask_track_{track_id}.pdf",
                    bbox_inches="tight")
        plt.show()

        summary[str(track_id)] = track_records

    with open(METADATA_DIR / "sem_rect_mask_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_path_exclusion_masking(
    TRACK_IDS, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM,
    FIGURES_DIR, METADATA_DIR, n_show=6, band_kwargs=None,
):
    """Two rows per track: row 0 is the region above the path (kept down to
    the green boundary), row 1 is the region below the path (kept from
    just past the blue boundary), path itself excluded from both."""
    band_kwargs = band_kwargs or {}
    FIGURES_DIR = Path(FIGURES_DIR)
    METADATA_DIR = Path(METADATA_DIR)
    summary = {}

    for track_id in TRACK_IDS:
        tiles = load_track_tiles(
            track_id, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM
        )
        if not tiles:
            continue

        n = len(tiles) if n_show is None else min(n_show, len(tiles))
        fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.4), constrained_layout=True)
        axes = np.atleast_2d(axes)
        if axes.shape[0] == 1:
            axes = axes.reshape(2, -1) if n == 1 else axes

        track_records = []
        for col, t in enumerate(tiles[:n]):
            out = generate_exclusion_outputs(t["image"], **band_kwargs)
            h, w = t["image"].shape[:2]
            px_to_mm = SEM_TILE_WIDTH_MM / w

            ax_above = axes[0, col]
            ax_below = axes[1, col]
            ax_above.imshow(out["above_image"], cmap="gray")
            ax_above.set_title(
                f"tile {t['index']:02d} [{t['x_lo_mm']:.1f}-{t['x_hi_mm']:.1f}mm]\n"
                f"above path", fontsize=8,
            )
            ax_above.axis("off")

            ax_below.imshow(out["below_image"], cmap="gray")
            ax_below.set_title("below path", fontsize=8)
            ax_below.axis("off")

            above_frac = float((out["above_mask"] > 0).mean())
            below_frac = float((out["below_mask"] > 0).mean())
            track_records.append({
                "tile_index": t["index"], "tile_path": t["path"],
                "x_lo_mm": t["x_lo_mm"], "x_hi_mm": t["x_hi_mm"],
                "above_area_frac": above_frac, "below_area_frac": below_frac,
                "top_boundary_mm_mean": float(np.nanmean(out["top"]) * px_to_mm),
                "bottom_boundary_mm_mean": float(np.nanmean(out["bottom"]) * px_to_mm),
            })

        fig.suptitle(f"Track {track_id}: path excluded, substrate regions only", fontsize=12)
        fig.savefig(FIGURES_DIR / f"sem_path_excluded_track_{track_id}.png",
                    dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / f"sem_path_excluded_track_{track_id}.pdf",
                    bbox_inches="tight")
        plt.show()

        summary[str(track_id)] = track_records

    with open(METADATA_DIR / "sem_path_excluded_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
