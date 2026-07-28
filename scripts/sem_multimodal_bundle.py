import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frame_extraction import (
    get_thermal_frame, get_height_slice, get_local_width_distribution,
    get_track_data, _width_distribution_from_arrays, _get_common_x_range,
    HEIGHT_CMAP, HEIGHTMAP_RANGES_UM,
)
from sem_tile_masking import load_track_tiles, generate_band_outputs
from sem_region_extraction import generate_rectangular_outputs, generate_exclusion_outputs


# ---------------------------------------------------------------------------
# SEM tile cache and x-lookup
#
# Thermal and height data are continuous in x (every scan-direction location
# has a nearest frame/column), SEM is not, it's one discrete ~6.41 mm tile
# per physical segment. This section is the bridge: given an arbitrary x,
# find which tile actually covers it, loading each track's tiles from disk
# only once no matter how many x locations get queried.
# ---------------------------------------------------------------------------

_SEM_TILE_CACHE = {}


def _track_num(track_id):
    """Accepts either 'Track_8' (frame_extraction's convention) or 8 (the
    SEM loader's convention, it builds a 'SEM_<id>' folder path) and always
    returns the int form."""
    if isinstance(track_id, int):
        return track_id
    return int(''.join(ch for ch in str(track_id) if ch.isdigit()))


def get_track_sem_tiles(track_id, SEM_DIR, get_sem_tile_paths, load_sem_tile,
                         SEM_TILE_WIDTH_MM, force_reload=False):
    """Loads (and caches) every SEM tile for a track, in physical x order,
    via load_track_tiles, which already handles the tile-order flip and
    180-degree rotation described in the dataset paper (highest-numbered
    tile is the 20 mm side)."""
    key = _track_num(track_id)
    if force_reload or key not in _SEM_TILE_CACHE:
        _SEM_TILE_CACHE[key] = load_track_tiles(
            key, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM
        )
    return _SEM_TILE_CACHE[key]


def clear_sem_tile_cache(track_id=None):
    if track_id is None:
        _SEM_TILE_CACHE.clear()
    else:
        _SEM_TILE_CACHE.pop(_track_num(track_id), None)


def get_track_sem_x_range(tiles):
    """(min, max) physical x actually covered by this track's SEM tiles."""
    return float(min(t["x_lo_mm"] for t in tiles)), float(max(t["x_hi_mm"] for t in tiles))


def _find_sem_tile_for_x(tiles, x_value):
    """Picks the tile whose [x_lo_mm, x_hi_mm) contains x_value. If x_value
    falls outside SEM coverage entirely, clips to the nearest edge tile and
    warns, same convention get_height_slice uses for its window clipping."""
    tiles_sorted = sorted(tiles, key=lambda t: t["index"])
    for t in tiles_sorted:
        if t["x_lo_mm"] <= x_value < t["x_hi_mm"]:
            return t
    if x_value < tiles_sorted[0]["x_lo_mm"]:
        print(f"Warning, x={x_value:.2f} mm is below the SEM coverage start "
              f"({tiles_sorted[0]['x_lo_mm']:.2f} mm), using tile 0.")
        return tiles_sorted[0]
    print(f"Warning, x={x_value:.2f} mm is beyond the SEM coverage end "
          f"({tiles_sorted[-1]['x_hi_mm']:.2f} mm), using the last tile.")
    return tiles_sorted[-1]


# ---------------------------------------------------------------------------
# 1. Masked SEM extraction at a physical location
# ---------------------------------------------------------------------------

def generate_masked_sem_outputs(image, method="box", box_mode="envelope", lower_pct=2.0,
                                 upper_pct=98.0, mask_mode="cover_track", view="tint",
                                 band_kwargs=None):
    """
    Dispatches to whichever SEM masking approach you want, on one tile image.

    method="box"       : the rectangular box (generate_rectangular_outputs),
                          mask_mode="cover_track" (default) blacks out the
                          track and keeps the substrate, mask_mode=
                          "keep_track" is the reverse.
    method="exclusion"  : drops the path entirely, keeps the two substrate
                          regions on either side (generate_exclusion_outputs).
    method="band"        : the raw detect_smooth_band rendering, view picks
                          'outline', 'tint', or 'cropped'.
    """
    band_kwargs = band_kwargs or {}
    if method == "box":
        out = generate_rectangular_outputs(
            image, box_mode=box_mode, lower_pct=lower_pct, upper_pct=upper_pct,
            mask_mode=mask_mode, **band_kwargs,
        )
        display_image = out["masked_image"]
    elif method == "exclusion":
        out = generate_exclusion_outputs(image, **band_kwargs)
        display_image = out["combined_excluded_image"]
    elif method == "band":
        out = generate_band_outputs(image, **band_kwargs)
        display_image = out["cropped_image"] if view == "cropped" else out[f"{view}_image"]
    else:
        raise ValueError("method must be 'box', 'exclusion', or 'band'")

    out = dict(out)
    out["method"] = method
    out["display_image"] = display_image
    return out


def get_masked_sem_at_x(
    x_value, track_id, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM,
    method="box", box_mode="envelope", lower_pct=2.0, upper_pct=98.0,
    mask_mode="cover_track", view="tint", band_kwargs=None, display=False,
):
    """
    Finds the SEM tile that physically covers x_value for this track, and
    returns it masked using the requested method. This is the piece that
    ties an arbitrary scan-direction x back to one of the 13-14 discrete
    SEM tiles for that track.

    method="box" and method="band" return a single masked image, under
    sem_masked_image. method="exclusion" returns two, sem_masked_image_above
    and sem_masked_image_below, since that method drops the path and keeps
    the two substrate regions on either side of it as separate images
    rather than one combined image. sem_masked_image is still included for
    exclusion too, as the combined (both regions in one image, path
    blanked) version, in case a single image is more convenient somewhere.
    """
    tiles = get_track_sem_tiles(
        track_id, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM
    )
    if not tiles:
        raise ValueError(f"No SEM tiles found for track {track_id}")

    tile = _find_sem_tile_for_x(tiles, x_value)
    out = generate_masked_sem_outputs(
        tile["image"], method=method, box_mode=box_mode, lower_pct=lower_pct,
        upper_pct=upper_pct, mask_mode=mask_mode, view=view, band_kwargs=band_kwargs,
    )

    result = {
        "sem_tile_index": tile["index"], "sem_tile_path": tile["path"],
        "sem_x_lo_mm": tile["x_lo_mm"], "sem_x_hi_mm": tile["x_hi_mm"],
        "sem_image": tile["image"], "sem_method": method,
        "sem_top_boundary": out.get("top"), "sem_bottom_boundary": out.get("bottom"),
    }

    if method == "exclusion":
        result["sem_masked_image_above"] = out["above_image"]
        result["sem_masked_image_below"] = out["below_image"]
        result["sem_masked_image"] = out["combined_excluded_image"]
    else:
        result["sem_masked_image"] = out["display_image"]

    if method == "box":
        result["sem_box_y_min_px"] = out["y_min"]
        result["sem_box_y_max_px"] = out["y_max"]

    if display:
        if method == "exclusion":
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
            axes[0].imshow(tile["image"], cmap="gray")
            axes[0].set_title(f"tile {tile['index']:02d} "
                               f"[{tile['x_lo_mm']:.1f}-{tile['x_hi_mm']:.1f}mm] original")
            axes[0].axis("off")
            axes[1].imshow(out["above_image"], cmap="gray")
            axes[1].set_title(f"above path, x = {x_value:.2f} mm")
            axes[1].axis("off")
            axes[2].imshow(out["below_image"], cmap="gray")
            axes[2].set_title(f"below path, x = {x_value:.2f} mm")
            axes[2].axis("off")
            plt.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
            axes[0].imshow(tile["image"], cmap="gray")
            axes[0].set_title(f"tile {tile['index']:02d} "
                               f"[{tile['x_lo_mm']:.1f}-{tile['x_hi_mm']:.1f}mm] original")
            axes[0].axis("off")
            axes[1].imshow(out["display_image"], cmap="gray")
            axes[1].set_title(f"masked ({method}), x = {x_value:.2f} mm")
            axes[1].axis("off")
            plt.show()

    return result


# ---------------------------------------------------------------------------
# 2. SEM-extended multimodal bundle functions
#
# Same names as frame_extraction's own get_multimodal_bundle_at_x and
# get_multimodal_bundle_for_track, redefined here to also pull in the
# matching SEM tile. Import this module after frame_extraction (or just use
# these directly) and these versions are the ones that get used.
# ---------------------------------------------------------------------------

def get_multimodal_bundle_at_x(
    x_value, track_id="Track_8", width_window_mm=1.0, display=False, track_data=None,
    include_sem=True, SEM_DIR=None, get_sem_tile_paths=None, load_sem_tile=None,
    SEM_TILE_WIDTH_MM=None, sem_method="box", sem_box_mode="envelope",
    sem_mask_mode="cover_track", sem_view="tint", sem_band_kwargs=None,
):
    """
    Thermal frame, 2D height slice, and width descriptors at one x location,
    same as before, plus the SEM tile covering that x, masked with
    sem_method. Set include_sem=False to skip SEM and match the original
    behavior exactly.
    """
    td = track_data if track_data is not None else get_track_data(track_id)

    x_thermal_actual, T = get_thermal_frame(x_value, track_id=track_id, display=display)
    x_height_actual, y, Z_crop = get_height_slice(
        x_value, track_id=track_id, window_mm=width_window_mm, mode="2d", display=display
    )
    dist = _width_distribution_from_arrays(
        td.x_height, td.y_height, td.Z_height, x_value, width_window_mm
    )

    bundle = {
        "x_requested_mm": float(x_value),
        "x_thermal_actual_mm": x_thermal_actual,
        "x_height_actual_mm": x_height_actual,
        "thermal_frame": T,
        "height_y_mm": y,
        "height_Z_crop": Z_crop,
        "width_mean_mm": dist["width_mean_mm"],
        "width_std_mm": dist["width_std_mm"],
        "width_ci95_low_mm": dist["width_ci95_low_mm"],
        "width_ci95_high_mm": dist["width_ci95_high_mm"],
        "y_left_mm": float(np.nanmean(dist["y_left_mm"])),
        "y_right_mm": float(np.nanmean(dist["y_right_mm"])),
        "n_valid_columns": dist["n_valid"],
        "widths_mm": dist["widths_mm"],
    }

    if include_sem:
        if None in (SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM):
            raise ValueError(
                "include_sem=True needs SEM_DIR, get_sem_tile_paths, load_sem_tile, "
                "and SEM_TILE_WIDTH_MM."
            )
        sem_result = get_masked_sem_at_x(
            x_value, track_id, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM,
            method=sem_method, box_mode=sem_box_mode, mask_mode=sem_mask_mode,
            view=sem_view, band_kwargs=sem_band_kwargs, display=display,
        )
        bundle.update(sem_result)

    if display:
        plt.figure(figsize=(6, 3))
        plt.hist(dist["widths_mm"][np.isfinite(dist["widths_mm"])], bins=15, alpha=0.7)
        plt.axvline(dist["width_mean_mm"], color="black", label=f"mean = {dist['width_mean_mm']:.4f} mm")
        plt.axvline(dist["width_ci95_low_mm"], color="gray", linestyle="--", label="±1.96 SD")
        plt.axvline(dist["width_ci95_high_mm"], color="gray", linestyle="--")
        plt.title(f"{track_id} local width distribution near x = {x_value:.2f} mm "
                  f"(n = {dist['n_valid']} columns)")
        plt.xlabel("width (mm)")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return bundle


def get_multimodal_bundle_for_track(
    track_id="Track_8", width_window_mm=1.0, step_mm=0.5, display=False,
    sample_locations=None, n_samples=None,
    include_sem=True, SEM_DIR=None, get_sem_tile_paths=None, load_sem_tile=None,
    SEM_TILE_WIDTH_MM=None, sem_method="box", sem_box_mode="envelope",
    sem_mask_mode="cover_track", sem_view="tint", sem_band_kwargs=None,
):
    """
    Width descriptors along the whole track's thermal/height common span,
    same as before, with two additions:

    - if include_sem is True, the common x range is further intersected
      with this track's actual SEM tile coverage, so sample locations never
      land somewhere SEM has no tile to offer.
    - n_samples: an alternative to naming sample_locations explicitly, give
      a count instead and this auto-picks that many evenly spaced x
      locations across the common range, then builds a full sample bundle
      (thermal + height + descriptors + masked SEM tile) for each, same as
      sample_locations would.
    """
    common_min, common_max = _get_common_x_range(track_id)

    if include_sem:
        if None in (SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM):
            raise ValueError(
                "include_sem=True needs SEM_DIR, get_sem_tile_paths, load_sem_tile, "
                "and SEM_TILE_WIDTH_MM."
            )
        sem_tiles = get_track_sem_tiles(
            track_id, SEM_DIR, get_sem_tile_paths, load_sem_tile, SEM_TILE_WIDTH_MM
        )
        sem_min, sem_max = get_track_sem_x_range(sem_tiles)
        common_min = max(common_min, sem_min)
        common_max = min(common_max, sem_max)
        if common_min >= common_max:
            raise ValueError(
                f"No x range shared by thermal, height, and SEM data for {track_id}."
            )

    x_locations = np.arange(common_min, common_max, step_mm)

    rows = []
    for x0 in x_locations:
        dist = get_local_width_distribution(track_id, x0, window_mm=width_window_mm, max_gap=3)
        rows.append({
            "x_mm": float(x0),
            "width_mean_mm": dist["width_mean_mm"],
            "width_std_mm": dist["width_std_mm"],
            "width_ci95_low_mm": dist["width_ci95_low_mm"],
            "width_ci95_high_mm": dist["width_ci95_high_mm"],
            "y_left_mm": float(np.nanmean(dist["y_left_mm"])),
            "y_right_mm": float(np.nanmean(dist["y_right_mm"])),
            "n_valid_columns": dist["n_valid"],
        })

    df = pd.DataFrame(rows)

    if display:
        x = np.load(f"{track_id}/{track_id}_height_x_mm.npy")
        y = np.load(f"{track_id}/{track_id}_height_y_mm.npy")
        Z = np.load(f"{track_id}/{track_id}_height_detrended.npy")
        id_num = _track_num(track_id)
        vmin_um, vmax_um = HEIGHTMAP_RANGES_UM.get(id_num, (None, None))

        fig, axes = plt.subplots(2, 1, figsize=(11, 7))
        extent = [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]
        axes[0].imshow(Z * 1000.0, cmap=HEIGHT_CMAP, aspect='auto',
                        extent=extent, vmin=vmin_um, vmax=vmax_um)
        axes[0].plot(df["x_mm"], df["y_left_mm"], color="red", linewidth=1.2, label="left boundary")
        axes[0].plot(df["x_mm"], df["y_right_mm"], color="black", linewidth=1.2, label="right boundary")
        axes[0].set_xlim(common_min, common_max)
        axes[0].set_title(f"{track_id} height map with extracted track boundary")
        axes[0].set_xlabel("actual x (mm)")
        axes[0].set_ylabel("cross-track y (mm)")
        axes[0].legend()

        axes[1].plot(df["x_mm"], df["width_mean_mm"], color="tab:orange", label="mean width")
        axes[1].fill_between(df["x_mm"],
                              df["width_mean_mm"] - 1.96 * df["width_std_mm"],
                              df["width_mean_mm"] + 1.96 * df["width_std_mm"],
                              alpha=0.15, label="±1.96 SD")
        axes[1].set_title(f"{track_id} local width, {width_window_mm} mm window, common span")
        axes[1].set_xlabel("actual x (mm)")
        axes[1].set_ylabel("width (mm)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    if sample_locations is None and n_samples is not None:
        sample_locations = np.linspace(common_min, common_max, n_samples).tolist()

    sample_bundles = {}
    if sample_locations:
        for x0 in sample_locations:
            sample_bundles[float(x0)] = get_multimodal_bundle_at_x(
                x0, track_id=track_id, width_window_mm=width_window_mm, display=True,
                include_sem=include_sem, SEM_DIR=SEM_DIR,
                get_sem_tile_paths=get_sem_tile_paths, load_sem_tile=load_sem_tile,
                SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM, sem_method=sem_method,
                sem_box_mode=sem_box_mode, sem_mask_mode=sem_mask_mode,
                sem_view=sem_view, sem_band_kwargs=sem_band_kwargs,
            )

    return {
        "track_id": track_id,
        "common_x_range_mm": (common_min, common_max),
        "descriptors": df,
        "sample_bundles": sample_bundles,
    }
