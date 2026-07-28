import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sem_tile_masking import load_track_tiles, generate_band_outputs


def run_smooth_band_masking(
    TRACK_IDS, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM,
    FIGURES_DIR, METADATA_DIR, n_show=6, ncols=6, view="tint", band_kwargs=None,
):
    """
    Runs detect_smooth_band (via generate_band_outputs) on every tile of
    every track, saves one grid figure per track to FIGURES_DIR, and writes
    a per-tile width summary json to METADATA_DIR.

    n_show   : tiles to show per track. None shows every tile for that
               track, arranged over multiple rows if needed.
    view     : which generate_band_outputs image to plot, 'outline',
               'tint', or 'cropped'.
    band_kwargs : passed straight through to generate_band_outputs, e.g.
               dict(roughness_ksize=21, expand_margin=0.75).
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
            out = generate_band_outputs(t["image"], **band_kwargs)

            if view == "cropped":
                ax.imshow(out["cropped_image"], cmap="gray")
            else:
                key = f"{view}_image"
                ax.imshow(cv2.cvtColor(out[key], cv2.COLOR_BGR2RGB))

            h, w = t["image"].shape[:2]
            px_to_mm = SEM_TILE_WIDTH_MM / w  # square pixels, per tile_width_profile
            width_mm = (out["bottom"] - out["top"]) * px_to_mm

            ax.set_title(
                f"tile {t['index']:02d} [{t['x_lo_mm']:.1f}-{t['x_hi_mm']:.1f}mm]\n"
                f"width={width_mm.mean():.2f}\u00b1{width_mm.std():.2f} mm", fontsize=8,
            )
            ax.axis("off")

            track_records.append({
                "tile_index": t["index"],
                "tile_path": t["path"],
                "x_lo_mm": t["x_lo_mm"],
                "x_hi_mm": t["x_hi_mm"],
                "width_mm_mean": float(width_mm.mean()),
                "width_mm_std": float(width_mm.std()),
                "width_mm_min": float(width_mm.min()),
                "width_mm_max": float(width_mm.max()),
            })

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(f"Track {track_id}: smooth-band detection ({view})", fontsize=12)
        fig.savefig(FIGURES_DIR / f"sem_smooth_band_track_{track_id}.png",
                    dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / f"sem_smooth_band_track_{track_id}.pdf",
                    bbox_inches="tight")
        plt.show()

        summary[str(track_id)] = track_records

    with open(METADATA_DIR / "sem_smooth_band_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
