#!/usr/bin/env python
# coding: utf-8

# ### Thermal frame and height slice extraction.

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import shift as nd_shift
import pandas as pd

THERMAL_PIXEL_SIZE_MM = 0.014
THERMAL_CMAP = 'jet'
THERMAL_VMIN = 1000.0
THERMAL_VMAX = 2500.0
HEIGHT_CMAP = 'jet'
HEIGHTMAP_RANGES_UM = {8: (-50, 125), 10: (-45, 110), 14: (-45, 70), 21: (-30, 45)}
DEFAULT_HEIGHT_WINDOW_MM = 400 * THERMAL_PIXEL_SIZE_MM  # 5.6 mm, matches thermal FOV

# Base directories for the per-track .npy files. Every load in this module
# goes through _thermal_path/_height_path below rather than a bare relative
# path, since thermal and height data can live in two different folders
# (they no longer sit next to the notebook the way they used to). Set these
# from the notebook right after importing this module, e.g.:
#
#   import frame_extraction
#   frame_extraction.THERMAL_BASE_DIR = str(THERMAL_DIR)
#   frame_extraction.HEIGHT_BASE_DIR = str(HEIGHT_DIR)
#
# assuming the actual files sit at THERMAL_DIR/Track_8/Track_8_x_mm_center.npy
# and HEIGHT_DIR/Track_8/Track_8_height_x_mm.npy respectively. If your real
# folder structure differs (e.g. no per-track subfolder, or a different
# nesting), adjust _thermal_path/_height_path below to match, this is the
# one place that needs to agree with wherever the files actually are.
THERMAL_BASE_DIR = "."
HEIGHT_BASE_DIR = "."


def _thermal_path(track_id, filename):
    return f"{THERMAL_BASE_DIR}/{track_id}/{filename}"


def _height_path(track_id, filename):
    return f"{HEIGHT_BASE_DIR}/{track_id}/{filename}"


def _shift_columns(frame, shift_px, order=1, mode='nearest'):
    return nd_shift(frame, shift=(0.0, shift_px), order=order, mode=mode)

def get_thermal_frame(x_value, track_id="Track_8", display=False, tol=1e-6, restore_shape=True):
    x_loc = np.load(_thermal_path(track_id, f"{track_id}_x_mm_center.npy"))
    thermal_frames = np.load(_thermal_path(track_id, f"{track_id}_thermal_frames.npy"))

    if x_value < x_loc.min() or x_value > x_loc.max():
        raise ValueError(
            f"x_value {x_value} is outside the available range "
            f"[{x_loc.min():.3f}, {x_loc.max():.3f}] for {track_id}."
        )

    idx_nearest = int(np.argmin(np.abs(x_loc - x_value)))
    n_cols_original = thermal_frames.shape[2]

    if abs(x_loc[idx_nearest] - x_value) <= tol:
        x_out = float(x_loc[idx_nearest])
        T_out = thermal_frames[idx_nearest]

    else:
        if x_loc[idx_nearest] < x_value:
            i_left, i_right = idx_nearest, idx_nearest + 1
        else:
            i_left, i_right = idx_nearest - 1, idx_nearest

        x_left, x_right = float(x_loc[i_left]), float(x_loc[i_right])
        T_left, T_right = thermal_frames[i_left], thermal_frames[i_right]

        shift_left_px = (x_left - x_value) / THERMAL_PIXEL_SIZE_MM
        shift_right_px = (x_right - x_value) / THERMAL_PIXEL_SIZE_MM

        T_left_aligned = _shift_columns(T_left, shift_left_px)
        T_right_aligned = _shift_columns(T_right, shift_right_px)

        gap = x_right - x_left
        w_left = (x_right - x_value) / gap
        w_right = (x_value - x_left) / gap
        T_blend = w_left * T_left_aligned + w_right * T_right_aligned

        crop_start = int(np.ceil(abs(shift_right_px)))
        crop_end = int(np.ceil(abs(shift_left_px)))
        T_valid = T_blend[:, crop_start: T_blend.shape[1] - crop_end]

        if restore_shape:
            # pad the trimmed edges back with the nearest valid column,
            # so shape always matches the original frames, edges are just
            # flagged internally as replicated rather than blended
            T_out = np.pad(T_valid, ((0, 0), (crop_start, crop_end)), mode='edge')
        else:
            T_out = T_valid

        x_out = x_value

    if display:
        n_cols = T_out.shape[1]
        thermal_extent = [0, n_cols * THERMAL_PIXEL_SIZE_MM,
                           T_out.shape[0] * THERMAL_PIXEL_SIZE_MM, 0]
        plt.figure(figsize=(5, 5))
        plt.imshow(T_out, cmap=THERMAL_CMAP, vmin=THERMAL_VMIN, vmax=THERMAL_VMAX,
                   extent=thermal_extent)
        plt.title(f'{track_id} thermal frame at x = {x_out:.2f} mm')
        plt.xlabel('thermal local x (mm)')
        plt.ylabel('thermal local y (mm)')
        cb = plt.colorbar()
        cb.set_label('temperature / intensity')
        plt.show()

    return x_out, T_out

def get_thermal_frame_rigid(x_value, track_id="Track_8", display=False):
    TRACK_IDS = [8, 10, 14, 21]
    THERMAL_CMAP = 'jet'
    THERMAL_VMIN = 1000.0
    THERMAL_VMAX = 2500.0
    HEIGHT_CMAP = 'jet'
    HEIGHTMAP_RANGES_UM = {8: (-50, 125), 10: (-45, 110), 14: (-45, 70), 21: (-30, 45)}
    THERMAL_PIXEL_SIZE_MM = 0.014  # 14 µm/pixel
    SEM_TILE_WIDTH_MM = 6.41
    SELECTED_SLOPE_EFF = {10: 0.003562, 14: -0.002517, 21: -0.002448}
    SELECTED_STRENGTH = {10: 1.00, 14: 0.75, 21: 1.00}

    x_loc = np.load(_thermal_path(track_id, f"{track_id}_x_mm_center.npy"))
    thermal_frames = np.load(_thermal_path(track_id, f"{track_id}_thermal_frames.npy"))

    idx = np.argmin(np.abs(x_loc - x_value))
    x = x_loc[idx]
    T = thermal_frames[idx]

    if display:
        thermal_extent = [0,
                           T.shape[1] * THERMAL_PIXEL_SIZE_MM,
                           T.shape[0] * THERMAL_PIXEL_SIZE_MM,
                           0]
        plt.figure(figsize=(5, 5))
        plt.imshow(T, cmap=THERMAL_CMAP, vmin=THERMAL_VMIN, vmax=THERMAL_VMAX, extent=thermal_extent)
        plt.title(f'{track_id} thermal frame at x ≈ {x:.1f} mm')
        plt.xlabel('thermal local x (mm)')
        plt.ylabel('thermal local y (mm)')
        cb = plt.colorbar()
        cb.set_label('temperature / intensity')
        plt.show()

    return x, T

def plot_full_heightmap(track_id="Track_8", save_path=None, dpi=400,
                         figsize=(12.5, 4.1)):
    x = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))
    y = np.load(_height_path(track_id, f"{track_id}_height_y_mm.npy"))
    Z = np.load(_height_path(track_id, f"{track_id}_height_detrended.npy"))

    id_num = int(''.join(ch for ch in track_id if ch.isdigit()))
    vmin_um, vmax_um = HEIGHTMAP_RANGES_UM.get(id_num, (None, None))

    extent = [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(Z * 1000.0, cmap=HEIGHT_CMAP, aspect='auto',
               extent=extent, vmin=vmin_um, vmax=vmax_um,)
               #interpolation='nearest')
    plt.title(f'{track_id}: detrended height map')
    plt.xlabel('actual x (mm)')
    plt.ylabel('cross-track y (mm)')
    cb = plt.colorbar()
    cb.set_label('height (µm)')

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('Saved:', save_path)

    plt.show()

def get_height_slice(x_value, track_id="Track_8", window_mm=DEFAULT_HEIGHT_WINDOW_MM,
                      mode="2d", display=False):
    """
    mode="2d" (default) crops a window around x_value and returns the 2D height map,
    same as before.

    mode="1d" pulls the single nearest column at x_value and returns it as a
    1D height vs y profile.
    """
    x = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))
    y = np.load(_height_path(track_id, f"{track_id}_height_y_mm.npy"))
    Z = np.load(_height_path(track_id, f"{track_id}_height_detrended.npy"))

    if x_value < x.min() or x_value > x.max():
        raise ValueError(
            f"x_value {x_value} is outside the available range "
            f"[{x.min():.3f}, {x.max():.3f}] for {track_id}."
        )

    if mode == "1d":
        idx = int(np.argmin(np.abs(x - x_value)))
        x_actual = float(x[idx])
        z_profile = Z[:, idx]

        if display:
            plot_height_slice(x_actual, y, z_profile, track_id=track_id, mode="1d")

        return x_actual, y, z_profile

    elif mode == "2d":
        half = window_mm / 2.0
        lo, hi = x_value - half, x_value + half
        if lo < x.min() or hi > x.max():
            # window would run off the edge of the scan, clip and warn
            lo = max(lo, x.min())
            hi = min(hi, x.max())
            print(f"Warning, requested window clipped near the scan edge, "
                  f"actual window is [{lo:.3f}, {hi:.3f}] mm.")

        mask = (x >= lo) & (x <= hi)
        x_crop = x[mask]
        Z_crop = Z[:, mask]

        if display:
            plot_height_slice(x_value, y, Z_crop, track_id=track_id, mode="2d", x_crop=x_crop)

        return x_crop, y, Z_crop

    else:
        raise ValueError(f"mode must be '2d' or '1d', got {mode!r}")

def plot_height_slice(x_value, y, data, track_id="Track_8", mode="2d", x_crop=None):
    """
    Plots whichever slice type get_height_slice produced.

    mode="2d" expects data as a 2D array (Z_crop) and x_crop as the cropped x axis.
    mode="1d" expects data as a 1D array (the height profile at one x column).
    """
    id_num = int(''.join(ch for ch in track_id if ch.isdigit()))

    if mode == "2d":
        if x_crop is None:
            raise ValueError("x_crop is required for mode='2d' plotting.")
        vmin_um, vmax_um = HEIGHTMAP_RANGES_UM.get(id_num, (None, None))
        extent = [float(x_crop[0]), float(x_crop[-1]), float(y[-1]), float(y[0])]
        plt.figure(figsize=(8, 3))
        plt.imshow(data * 1000.0, cmap=HEIGHT_CMAP, aspect='auto',
                   extent=extent, vmin=vmin_um, vmax=vmax_um)
        plt.title(f'{track_id} height map slice, centered near x = {x_value:.2f} mm')
        plt.xlabel('actual x (mm)')
        plt.ylabel('cross-track y (mm)')
        cb = plt.colorbar()
        cb.set_label('height (µm)')
        plt.show()

    elif mode == "1d":
        plt.figure(figsize=(8, 3))
        plt.plot(y, data * 1000.0)
        plt.title(f'{track_id} height profile at x = {x_value:.2f} mm')
        plt.xlabel('cross-track y (mm)')
        plt.ylabel('height (µm)')
        plt.grid(True, alpha=0.3)
        plt.show()

    else:
        raise ValueError(f"mode must be '2d' or '1d', got {mode!r}")


# ### Local Geometry Descriptor Extraction

import nsf_fmrg_data
from nsf_fmrg_data import largest_true_run

def _valid_run_boundary(y, valid_mask, max_gap=3):
    """
    Finds the boundary of the laser track by locating the longest contiguous
    run of valid (non-NaN) samples along y, closing gaps of up to max_gap
    consecutive missing samples so that scattered dropout pixels inside the
    track don't split it into separate runs.

    Returns width_mm, y_left_mm, y_right_mm, left_clipped, right_clipped.
    The clipped flags are True if the run touches the edge of the y array,
    meaning the true edge might sit outside the scanned window.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n = len(valid_mask)

    closed = valid_mask.copy()
    i = 0
    while i < n:
        if not closed[i]:
            j = i
            while j < n and not closed[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < n:
                closed[i:j] = True
            i = j
        else:
            i += 1

    start, stop = largest_true_run(closed)
    if start is None:
        return np.nan, np.nan, np.nan, True, True

    y_left = float(y[start])
    y_right = float(y[stop - 1])
    left_clipped = (start == 0)
    right_clipped = (stop == n)
    return float(y_right - y_left), y_left, y_right, left_clipped, right_clipped


def _column_width(y, col, max_gap=3):
    return _valid_run_boundary(y, np.isfinite(col), max_gap=max_gap)

def get_local_width_distribution(track_id="Track_8", x_value=55.0, window_mm=DEFAULT_HEIGHT_WINDOW_MM,
                                  max_gap=3):
    """
    Computes the per-column track width for every x column inside a window
    centered at x_value, using the valid-run boundary method. Returns the
    raw per-column measurements plus summary statistics.
    """
    x = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))
    y = np.load(_height_path(track_id, f"{track_id}_height_y_mm.npy"))
    Z = np.load(_height_path(track_id, f"{track_id}_height_detrended.npy"))

    half = window_mm / 2.0
    lo, hi = max(x_value - half, x.min()), min(x_value + half, x.max())
    win_mask = (x >= lo) & (x <= hi)
    x_window = x[win_mask]
    Z_window = Z[:, win_mask]

    widths, y_lefts, y_rights, left_clips, right_clips = [], [], [], [], []
    for col in Z_window.T:
        w, yl, yr, lc, rc = _column_width(y, col, max_gap=max_gap)
        widths.append(w)
        y_lefts.append(yl)
        y_rights.append(yr)
        left_clips.append(lc)
        right_clips.append(rc)

    widths = np.array(widths)
    valid = np.isfinite(widths)

    mean_width = np.nanmean(widths)
    std_width = np.nanstd(widths)
    ci_low = mean_width - 1.96 * std_width
    ci_high = mean_width + 1.96 * std_width

    return {
        "x_window_mm": x_window,
        "widths_mm": widths,
        "y_left_mm": np.array(y_lefts),
        "y_right_mm": np.array(y_rights),
        "left_clipped": np.array(left_clips),
        "right_clipped": np.array(right_clips),
        "n_valid": int(valid.sum()),
        "width_mean_mm": float(mean_width),
        "width_std_mm": float(std_width),
        "width_ci95_low_mm": float(ci_low),
        "width_ci95_high_mm": float(ci_high),
    }


def get_local_width_distribution_std(track_id="Track_8", x_value=55.0, window_mm=DEFAULT_HEIGHT_WINDOW_MM,
                                  max_gap=3):
    """
    Computes the per-column track width for every x column inside a window
    centered at x_value, using the valid-run boundary method. Returns the
    raw per-column measurements plus summary statistics.

    width_std_mm is the spread of the individual column widths.
    width_sem_mm is the standard deviation of the mean, std divided by
    sqrt(n_valid), and the 95 percent interval is built from that, so it
    reflects how precisely the mean width is known rather than how much
    width varies column to column.
    """
    x = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))
    y = np.load(_height_path(track_id, f"{track_id}_height_y_mm.npy"))
    Z = np.load(_height_path(track_id, f"{track_id}_height_detrended.npy"))

    half = window_mm / 2.0
    lo, hi = max(x_value - half, x.min()), min(x_value + half, x.max())
    win_mask = (x >= lo) & (x <= hi)
    x_window = x[win_mask]
    Z_window = Z[:, win_mask]

    widths, y_lefts, y_rights, left_clips, right_clips = [], [], [], [], []
    for col in Z_window.T:
        w, yl, yr, lc, rc = _column_width(y, col, max_gap=max_gap)
        widths.append(w)
        y_lefts.append(yl)
        y_rights.append(yr)
        left_clips.append(lc)
        right_clips.append(rc)

    widths = np.array(widths)
    valid = np.isfinite(widths)
    n_valid = int(valid.sum())

    mean_width = np.nanmean(widths)
    std_width = np.nanstd(widths)
    sem_width = std_width / np.sqrt(n_valid) if n_valid > 0 else np.nan
    ci_low = mean_width - 1.96 * sem_width
    ci_high = mean_width + 1.96 * sem_width

    return {
        "x_window_mm": x_window,
        "widths_mm": widths,
        "y_left_mm": np.array(y_lefts),
        "y_right_mm": np.array(y_rights),
        "left_clipped": np.array(left_clips),
        "right_clipped": np.array(right_clips),
        "n_valid": n_valid,
        "width_mean_mm": float(mean_width),
        "width_std_mm": float(std_width),
        "width_sem_mm": float(sem_width),
        "width_ci95_low_mm": float(ci_low),
        "width_ci95_high_mm": float(ci_high),
    }

def extract_geometry_descriptors(track_id="Track_8", x_value=None, window_mm=DEFAULT_HEIGHT_WINDOW_MM,
                                  mode="both", step_mm=0.5, max_gap=3):
    """
    Extracts local track width and left/right boundary locations from the height map,
    using the valid-run boundary method (the track is defined by where the profilometer
    captured continuous valid data, not by a height threshold).

    x_value, a single float extracts at one location. None extracts along the entire
    track at step_mm spacing.

    mode is '1d' (single nearest column), '2d' (mean, std, and 95 percent range from
    the distribution of per-column widths across a window_mm window), or 'both'
    (averages the 1d point estimate with the 2d mean).

    Returns a dict for a single location, or a pandas DataFrame for the whole track.
    """
    x = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))
    y = np.load(_height_path(track_id, f"{track_id}_height_y_mm.npy"))
    Z = np.load(_height_path(track_id, f"{track_id}_height_detrended.npy"))

    def descriptors_at(x0):
        idx = int(np.argmin(np.abs(x - x0)))
        x_actual = float(x[idx])
        out = {"x_mm": x_actual}

        if mode in ("1d", "both"):
            w1, yl1, yr1, lc1, rc1 = _column_width(y, Z[:, idx], max_gap=max_gap)
            out["width_1d_mm"] = w1
            out["y_left_1d_mm"] = yl1
            out["y_right_1d_mm"] = yr1
            out["left_clipped_1d"] = lc1
            out["right_clipped_1d"] = rc1

        if mode in ("2d", "both"):
            dist = get_local_width_distribution(track_id, x0, window_mm=window_mm, max_gap=max_gap)
            out["width_2d_mean_mm"] = dist["width_mean_mm"]
            out["width_2d_std_mm"] = dist["width_std_mm"]
            out["width_2d_ci95_low_mm"] = dist["width_ci95_low_mm"]
            out["width_2d_ci95_high_mm"] = dist["width_ci95_high_mm"]
            out["width_2d_n_valid"] = dist["n_valid"]
            out["y_left_2d_mm"] = float(np.nanmean(dist["y_left_mm"]))
            out["y_right_2d_mm"] = float(np.nanmean(dist["y_right_mm"]))

        if mode == "both":
            out["width_mm"] = np.nanmean([out["width_1d_mm"], out["width_2d_mean_mm"]])
            out["y_left_mm"] = np.nanmean([out["y_left_1d_mm"], out["y_left_2d_mm"]])
            out["y_right_mm"] = np.nanmean([out["y_right_1d_mm"], out["y_right_2d_mm"]])
        elif mode == "1d":
            out["width_mm"] = out["width_1d_mm"]
            out["y_left_mm"] = out["y_left_1d_mm"]
            out["y_right_mm"] = out["y_right_1d_mm"]
        elif mode == "2d":
            out["width_mm"] = out["width_2d_mean_mm"]
            out["y_left_mm"] = out["y_left_2d_mm"]
            out["y_right_mm"] = out["y_right_2d_mm"]

        return out

    if x_value is not None:
        return descriptors_at(x_value)

    x_locations = np.arange(x.min(), x.max(), step_mm)
    rows = [descriptors_at(xv) for xv in x_locations]
    return pd.DataFrame(rows)

def compare_extraction_modes(track_id="Track_8", window_mm=DEFAULT_HEIGHT_WINDOW_MM, step_mm=0.5, max_gap=3):
    df_1d = extract_geometry_descriptors(track_id, mode="1d", step_mm=step_mm, max_gap=max_gap)
    df_2d = extract_geometry_descriptors(track_id, mode="2d", window_mm=window_mm, step_mm=step_mm, max_gap=max_gap)

    width_1d = df_1d["width_mm"].to_numpy()
    width_2d = df_2d["width_2d_mean_mm"].to_numpy()
    width_avg = np.nanmean(np.vstack([width_1d, width_2d]), axis=0)
    x_mm = df_1d["x_mm"].to_numpy()

    valid = np.isfinite(width_1d) & np.isfinite(width_2d)

    mad_1d_2d = np.nanmean(np.abs(width_1d[valid] - width_2d[valid]))
    corr_1d_2d = np.corrcoef(width_1d[valid], width_2d[valid])[0, 1]

    print(f"1D vs 2D mean absolute difference is {mad_1d_2d:.4f} mm")
    print(f"1D vs 2D correlation is {corr_1d_2d:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    axes[0].plot(x_mm, width_1d, label="1D (single column)", alpha=0.8)
    axes[0].plot(x_mm, width_2d, label=f"2D mean over {window_mm} mm window", alpha=0.8)
    axes[0].fill_between(x_mm, df_2d["width_2d_ci95_low_mm"], df_2d["width_2d_ci95_high_mm"],
                          alpha=0.15, label="2D 95 percent range")
    axes[0].plot(x_mm, width_avg, label="average", linestyle="--", color="black")
    axes[0].set_xlabel("actual x (mm)")
    axes[0].set_ylabel("track width (mm)")
    axes[0].set_title(f"{track_id} local width, three extraction modes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    diff = width_1d - width_2d
    mean_val = (width_1d + width_2d) / 2.0
    bias = np.nanmean(diff[valid])
    sd = np.nanstd(diff[valid])

    axes[1].scatter(mean_val[valid], diff[valid], s=8, alpha=0.5)
    axes[1].axhline(bias, color="black", label=f"bias = {bias:.4f} mm")
    axes[1].axhline(bias + 1.96 * sd, color="gray", linestyle="--", label="±1.96 SD")
    axes[1].axhline(bias - 1.96 * sd, color="gray", linestyle="--")
    axes[1].set_xlabel("mean width of 1D and 2D (mm)")
    axes[1].set_ylabel("1D minus 2D (mm)")
    axes[1].set_title("Bland-Altman agreement plot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "x_mm": x_mm,
        "width_1d_mm": width_1d,
        "width_2d_mean_mm": width_2d,
        "width_2d_std_mm": df_2d["width_2d_std_mm"],
        "width_avg_mm": width_avg,
    })



# ### Joint data extraction

def _get_common_x_range(track_id="Track_8"):
    """
    Finds the x range that both the thermal data and the height data actually
    cover for this track, so a downstream loop never asks either loader for
    an x location that only exists in the other modality.
    """
    x_thermal = np.load(_thermal_path(track_id, f"{track_id}_x_mm_center.npy"))
    x_height = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))

    common_min = max(x_thermal.min(), x_height.min())
    common_max = min(x_thermal.max(), x_height.max())

    if common_min >= common_max:
        raise ValueError(
            f"No overlapping x range between thermal and height data for {track_id}. "
            f"Thermal spans [{x_thermal.min():.3f}, {x_thermal.max():.3f}], "
            f"height spans [{x_height.min():.3f}, {x_height.max():.3f}]."
        )

    return float(common_min), float(common_max)


def get_multimodal_bundle_at_x(x_value, track_id="Track_8", width_window_mm=1.0, display=False):
    """
    Extracts the thermal frame, the 2D height map slice, and the 2D width
    descriptors, all at a single x location.

    width_window_mm controls the height slice and the descriptor window, and
    defaults to 1 mm rather than the 5.6 mm thermal field of view default,
    since averaging over the full thermal footprint is too wide for a single
    point estimate of local width.

    display, if True, shows the thermal frame, the height map slice, and a
    histogram of the per-column widths inside the window with the mean, std,
    and 95 percent range marked.
    """
    x_thermal_actual, T = get_thermal_frame(x_value, track_id=track_id, display=display)

    x_height_actual, y, Z_crop = get_height_slice(
        x_value, track_id=track_id, window_mm=width_window_mm, mode="2d", display=display
    )

    dist = get_local_width_distribution(
        track_id, x_value, window_mm=width_window_mm, max_gap=3
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


def get_multimodal_bundle_for_track(track_id="Track_8", width_window_mm=1.0, step_mm=0.5,
                                     display=False, sample_locations=None):
    """
    Extracts the 2D width descriptors along the entire track, restricted to the
    x range where both thermal and height data are available. Does not load
    every thermal frame and height crop for every location by default, since
    that is heavy on memory for a full track and rarely useful all at once.

    display, if True, shows the full height map with the extracted left and
    right boundary overlaid, plus the local width versus x curve with the std
    band.

    sample_locations, an optional list of x values, will additionally call
    get_multimodal_bundle_at_x with display=True for just those points, so
    you can inspect the thermal frame and height slice at a few spots of
    interest without rendering the entire track.
    """
    common_min, common_max = _get_common_x_range(track_id)
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
        x = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))
        y = np.load(_height_path(track_id, f"{track_id}_height_y_mm.npy"))
        Z = np.load(_height_path(track_id, f"{track_id}_height_detrended.npy"))
        id_num = int(''.join(ch for ch in track_id if ch.isdigit()))
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
        axes[1].set_title(f"{track_id} local width, {width_window_mm} mm window, common thermal/height span")
        axes[1].set_xlabel("actual x (mm)")
        axes[1].set_ylabel("width (mm)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    sample_bundles = {}
    if sample_locations:
        for x0 in sample_locations:
            sample_bundles[float(x0)] = get_multimodal_bundle_at_x(
                x0, track_id=track_id, width_window_mm=width_window_mm, display=True
            )

    return {
        "track_id": track_id,
        "common_x_range_mm": (common_min, common_max),
        "descriptors": df,
        "sample_bundles": sample_bundles,
    }

_TRACK_DATA_CACHE = {}

class TrackData:
    """
    Loads all raw arrays for one track exactly once. Precomputes the common
    x range shared by the thermal and height modalities, so downstream loops
    never ask either modality for an x location the other one doesn't cover.
    """
    def __init__(self, track_id):
        self.track_id = track_id
        self.x_thermal = np.load(_thermal_path(track_id, f"{track_id}_x_mm_center.npy"))
        self.thermal_frames = np.load(_thermal_path(track_id, f"{track_id}_thermal_frames.npy"))
        self.x_height = np.load(_height_path(track_id, f"{track_id}_height_x_mm.npy"))
        self.y_height = np.load(_height_path(track_id, f"{track_id}_height_y_mm.npy"))
        self.Z_height = np.load(_height_path(track_id, f"{track_id}_height_detrended.npy"))

        self.common_x_min = float(max(self.x_thermal.min(), self.x_height.min()))
        self.common_x_max = float(min(self.x_thermal.max(), self.x_height.max()))
        if self.common_x_min >= self.common_x_max:
            raise ValueError(
                f"No overlapping x range between thermal and height data for {track_id}."
            )

    def nearest_thermal_index(self, x_value):
        return int(np.argmin(np.abs(self.x_thermal - x_value)))


def get_track_data(track_id="Track_8", force_reload=False):
    """
    Returns a cached TrackData for this track, loading it from disk only the
    first time it's requested. Pass force_reload=True if the underlying
    files have changed and you need a fresh read.
    """
    if force_reload or track_id not in _TRACK_DATA_CACHE:
        _TRACK_DATA_CACHE[track_id] = TrackData(track_id)
    return _TRACK_DATA_CACHE[track_id]


def clear_track_data_cache(track_id=None):
    """
    Frees cached arrays. Pass a track_id to drop just that one, or nothing
    to clear everything, useful once you're done building the training set
    and want the thermal arrays out of memory before training.
    """
    if track_id is None:
        _TRACK_DATA_CACHE.clear()
    else:
        _TRACK_DATA_CACHE.pop(track_id, None)


def _width_distribution_from_arrays(x_height, y_height, Z_height, x_value, window_mm, max_gap=3):
    half = window_mm / 2.0
    lo, hi = max(x_value - half, x_height.min()), min(x_value + half, x_height.max())
    win_mask = (x_height >= lo) & (x_height <= hi)
    x_window = x_height[win_mask]
    Z_window = Z_height[:, win_mask]

    widths, y_lefts, y_rights, left_clips, right_clips = [], [], [], [], []
    for col in Z_window.T:
        w, yl, yr, lc, rc = _column_width(y_height, col, max_gap=max_gap)
        widths.append(w)
        y_lefts.append(yl)
        y_rights.append(yr)
        left_clips.append(lc)
        right_clips.append(rc)

    widths = np.array(widths)
    valid = np.isfinite(widths)
    n_valid = int(valid.sum())

    mean_width = np.nanmean(widths)
    std_width = np.nanstd(widths)
    ci_low = mean_width - 1.96 * std_width
    ci_high = mean_width + 1.96 * std_width

    return {
        "x_window_mm": x_window,
        "widths_mm": widths,
        "y_left_mm": np.array(y_lefts),
        "y_right_mm": np.array(y_rights),
        "left_clipped": np.array(left_clips),
        "right_clipped": np.array(right_clips),
        "n_valid": n_valid,
        "width_mean_mm": float(mean_width),
        "width_std_mm": float(std_width),
        "width_ci95_low_mm": float(ci_low),
        "width_ci95_high_mm": float(ci_high),
    }


def get_local_width_distribution(track_id="Track_8", x_value=55.0, window_mm=DEFAULT_HEIGHT_WINDOW_MM,
                                  max_gap=3):
    """
    Disk-loading convenience wrapper, kept for single ad hoc queries.
    Internally goes through the cached TrackData so repeated calls on the
    same track_id still avoid re-reading the npy files.
    """
    td = get_track_data(track_id)
    return _width_distribution_from_arrays(td.x_height, td.y_height, td.Z_height, x_value, window_mm, max_gap)


def build_training_dataset_for_track(track_id="Track_8", width_window_mm=1.0, step_mm=0.5,
                                      max_gap=3, include_thermal=True):
    """
    Builds one training set entry per x location across the common thermal
    and height span, for a single track. Loads all raw arrays exactly once
    through the TrackData cache, then loops entirely in memory.

    Returns a dict with a metadata DataFrame (one row per x location, with
    the width descriptors and boundary locations) and, if include_thermal,
    a stacked thermal_frames array aligned to the same row order, ready to
    hand to a model as X_thermal alongside the metadata as labels/features.
    """
    td = get_track_data(track_id)
    x_locations = np.arange(td.common_x_min, td.common_x_max, step_mm)

    records = []
    thermal_stack = [] if include_thermal else None

    for x0 in x_locations:
        dist = _width_distribution_from_arrays(
            td.x_height, td.y_height, td.Z_height, x0, width_window_mm, max_gap
        )
        records.append({
            "track_id": track_id,
            "x_mm": float(x0),
            "width_mean_mm": dist["width_mean_mm"],
            "width_std_mm": dist["width_std_mm"],
            "width_ci95_low_mm": dist["width_ci95_low_mm"],
            "width_ci95_high_mm": dist["width_ci95_high_mm"],
            "y_left_mm": float(np.nanmean(dist["y_left_mm"])),
            "y_right_mm": float(np.nanmean(dist["y_right_mm"])),
            "n_valid_columns": dist["n_valid"],
        })
        if include_thermal:
            idx_t = td.nearest_thermal_index(x0)
            thermal_stack.append(td.thermal_frames[idx_t])

    metadata = pd.DataFrame(records)
    result = {"metadata": metadata}
    if include_thermal:
        result["thermal_frames"] = np.stack(thermal_stack, axis=0)

    return result


def build_training_dataset(track_ids=("Track_8", "Track_10", "Track_14", "Track_21"),
                            width_window_mm=1.0, step_mm=0.5, max_gap=3, include_thermal=True):
    """
    Runs build_training_dataset_for_track across multiple tracks and
    concatenates the results, with track_id already carried as a column
    in metadata so rows stay traceable back to their source track.
    """
    all_metadata = []
    all_thermal = [] if include_thermal else None

    for tid in track_ids:
        result = build_training_dataset_for_track(
            tid, width_window_mm=width_window_mm, step_mm=step_mm,
            max_gap=max_gap, include_thermal=include_thermal
        )
        all_metadata.append(result["metadata"])
        if include_thermal:
            all_thermal.append(result["thermal_frames"])

    metadata = pd.concat(all_metadata, ignore_index=True)
    output = {"metadata": metadata}
    if include_thermal:
        output["thermal_frames"] = np.concatenate(all_thermal, axis=0)

    return output

def get_multimodal_bundle_at_x(x_value, track_id="Track_8", width_window_mm=1.0, display=False, track_data=None):
    td = track_data if track_data is not None else get_track_data(track_id)

    x_thermal_actual, T = get_thermal_frame(x_value, track_id=track_id, display=display)
    x_height_actual, y, Z_crop = get_height_slice(
        x_value, track_id=track_id, window_mm=width_window_mm, mode="2d", display=display
    )
    dist = _width_distribution_from_arrays(td.x_height, td.y_height, td.Z_height, x_value, width_window_mm)

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