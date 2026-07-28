"""
height_map_descriptors.py

Extracts areal roughness descriptors (Sa, Sq, Sz) and per-x-position
track-geometry descriptors (local width, left/right boundary position)
from a Bruker/Wyko ASCII height map, following the conventions used in
`load_wyko_asc` / `robust_plane_detrend` from the challenge starter notebook.

Expected inputs
----------------
Z      : 2D array, shape (n_y, n_x), height values in mm (NaN = dropout)
x      : 1D array, shape (n_x,), actual x position in mm
y       : 1D array, shape (n_y,), cross-track y position in mm

Typical usage
-------------
    from height_map_descriptors import (
        areal_roughness,
        local_width_profile,
        column_dropout_diagnostic,
    )

    height = load_wyko_asc(HEIGHT_DIR, track_id, crop_to_common=True)
    Z_detrended, coef = robust_plane_detrend(
        height['Z_mm'], height['x_actual_mm'], height['y_mm']
    )

    Sa, Sq, Sz = areal_roughness(Z_detrended)

    profile = local_width_profile(
        Z_detrended, height['x_actual_mm'], height['y_mm']
    )
"""

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------
# 1. Global areal roughness descriptors (Sa, Sq, Sz)
# ------------------------------------------------------------------

def areal_roughness(Z):
    """
    Sa (arithmetic mean areal roughness), Sq (RMS areal roughness),
    and Sz (peak-to-valley height) for a detrended height map.

    NaNs (dropout pixels) are excluded from every statistic.

    Parameters
    ----------
    Z : (n_y, n_x) array, detrended height values (mm or um, any consistent unit)

    Returns
    -------
    Sa, Sq, Sz : floats, in the same units as Z
    """
    Z = np.asarray(Z, dtype=float)
    valid = np.isfinite(Z)
    if not valid.any():
        return np.nan, np.nan, np.nan

    z = Z[valid]
    Sa = np.mean(np.abs(z))
    Sq = np.sqrt(np.mean(z**2))
    Sz = z.max() - z.min()
    return float(Sa), float(Sq), float(Sz)


def rolling_areal_roughness(Z, x, window_mm=2.0, step_mm=0.5, min_valid_frac=0.3):
    """
    Sa/Sq/Sz computed in overlapping windows along x, so you get a
    profile Sa(x), Sq(x), Sz(x) instead of one number for the whole track.

    A window is skipped (returned as NaN) if fewer than `min_valid_frac`
    of its pixels are finite, so dropout-heavy regions don't silently
    produce misleading roughness values.

    Parameters
    ----------
    Z              : (n_y, n_x) detrended height map
    x              : (n_x,) actual x position, mm
    window_mm      : width of the rolling window, mm
    step_mm        : step between window centers, mm
    min_valid_frac : minimum finite-pixel fraction required to compute stats

    Returns
    -------
    dict with keys: x_centers, Sa, Sq, Sz, valid_frac  (all 1D arrays)
    """
    Z = np.asarray(Z, dtype=float)
    x = np.asarray(x, dtype=float)

    x_start, x_end = x[0], x[-1]
    centers = np.arange(x_start + window_mm / 2, x_end - window_mm / 2, step_mm)

    Sa_arr, Sq_arr, Sz_arr, frac_arr = [], [], [], []
    for xc in centers:
        cols = (x >= xc - window_mm / 2) & (x <= xc + window_mm / 2)
        block = Z[:, cols]
        frac = np.isfinite(block).mean() if block.size else 0.0
        frac_arr.append(frac)
        if frac < min_valid_frac:
            Sa_arr.append(np.nan)
            Sq_arr.append(np.nan)
            Sz_arr.append(np.nan)
            continue
        Sa, Sq, Sz = areal_roughness(block)
        Sa_arr.append(Sa)
        Sq_arr.append(Sq)
        Sz_arr.append(Sz)

    return {
        "x_centers": centers,
        "Sa": np.array(Sa_arr),
        "Sq": np.array(Sq_arr),
        "Sz": np.array(Sz_arr),
        "valid_frac": np.array(frac_arr),
    }


# ------------------------------------------------------------------
# 2. Per-column dropout diagnostic
# ------------------------------------------------------------------
# This is the diagnostic recommended before trusting any width number:
# check whether NaN dropout is patterned by row (cross-track position)
# or scattered randomly. A column can look "valid" by fraction alone
# while still missing exactly the rows needed to find a clean edge.

def column_dropout_diagnostic(Z, x, y):
    """
    Per-x-column dropout statistics, to check whether missing data is
    concentrated in specific cross-track (y) rows rather than randomly
    scattered — the pattern that produces sawtooth width artifacts.

    Returns
    -------
    dict with keys:
        x               : (n_x,) x positions, mm
        valid_frac      : (n_x,) finite fraction per column
        first_valid_row : (n_x,) index of first finite row (NaN-safe int, -1 if none)
        last_valid_row  : (n_x,) index of last finite row
        n_gaps          : (n_x,) number of separate finite "islands" per column
                          (>1 means the valid region is split by an internal
                          NaN gap, a common cause of spurious width jumps)
    """
    Z = np.asarray(Z, dtype=float)
    n_y, n_x = Z.shape
    valid = np.isfinite(Z)

    valid_frac = valid.mean(axis=0)
    first_valid_row = np.full(n_x, -1, dtype=int)
    last_valid_row = np.full(n_x, -1, dtype=int)
    n_gaps = np.zeros(n_x, dtype=int)

    for j in range(n_x):
        rows = np.where(valid[:, j])[0]
        if rows.size == 0:
            continue
        first_valid_row[j] = rows[0]
        last_valid_row[j] = rows[-1]
        # count separate runs of consecutive finite rows
        breaks = np.sum(np.diff(rows) > 1)
        n_gaps[j] = breaks + 1

    return {
        "x": np.asarray(x, dtype=float),
        "valid_frac": valid_frac,
        "first_valid_row": first_valid_row,
        "last_valid_row": last_valid_row,
        "n_gaps": n_gaps,
    }


# ------------------------------------------------------------------
# 3. Local width and left/right boundary position along x
# ------------------------------------------------------------------

@dataclass
class WidthProfile:
    x: np.ndarray            # x positions, mm
    left: np.ndarray         # left boundary, y (mm), NaN if undetected
    right: np.ndarray        # right boundary, y (mm), NaN if undetected
    width: np.ndarray        # right - left, mm, NaN if undetected
    valid_frac: np.ndarray   # finite-pixel fraction used per column
    dropped: np.ndarray      # bool, True where the column was rejected
    method: str
    threshold_used: np.ndarray  # per-column threshold value (height units)


def local_width_profile(
    Z,
    x,
    y,
    method="mad",
    mad_k=3.0,
    abs_threshold=None,
    min_valid_frac=0.5,
    min_run_px=3,
    smooth_window=None,
):
    """
    Local track width W(x) = x_right - x_left ... here defined along the
    cross-track direction y for each fixed x (i.e. W(x) is the track
    footprint width measured across y at that x position), plus the
    left/right boundary positions L(x), R(x).

    Boundary detection per column
    ------------------------------
    For each x column:
      1. Take the finite height values along y.
      2. Skip the column (mark NaN) if the finite fraction is below
         `min_valid_frac` — this is the main defense against the
         dropout-driven sawtooth artifact, since a column with too few
         real pixels cannot give a reliable edge.
      3. Threshold the column: a pixel is "in the track" if
         |z| > threshold, where threshold is either
           - 'mad': median absolute deviation based, threshold =
             mad_k * 1.4826 * MAD(z), which is robust to outliers and to
             the shape of the height distribution, or
           - 'abs': a fixed value `abs_threshold` (height units), or
           - 'otsu': automatic bimodal threshold via Otsu's method.
      4. Keep only the largest contiguous run of in-track pixels of at
         least `min_run_px` samples (this rejects isolated speckle hits
         that would otherwise create a fake early/late edge).
      5. left = y at the start of that run, right = y at the end.

    Parameters
    ----------
    Z              : (n_y, n_x) detrended height map, mm (or consistent units)
    x              : (n_x,) actual x position, mm
    y              : (n_y,) cross-track y position, mm
    method         : 'mad', 'abs', or 'otsu'
    mad_k          : multiplier on the robust MAD-based threshold (method='mad')
    abs_threshold  : fixed threshold value (method='abs')
    min_valid_frac : minimum finite fraction required to attempt detection
    min_run_px     : minimum length (in samples) of the in-track run to accept
    smooth_window  : optional odd int; if given, a centered moving-average
                     is applied to the resulting width profile only (not to
                     the raw data), useful for comparing raw vs smoothed
                     width without re-deriving boundaries

    Returns
    -------
    WidthProfile
    """
    Z = np.asarray(Z, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_y, n_x = Z.shape

    left = np.full(n_x, np.nan)
    right = np.full(n_x, np.nan)
    width = np.full(n_x, np.nan)
    valid_frac = np.zeros(n_x)
    dropped = np.zeros(n_x, dtype=bool)
    thr_used = np.full(n_x, np.nan)

    for j in range(n_x):
        col = Z[:, j]
        finite = np.isfinite(col)
        frac = finite.mean()
        valid_frac[j] = frac

        if frac < min_valid_frac:
            dropped[j] = True
            continue

        z = col.copy()
        z_valid = z[finite]

        if method == "mad":
            med = np.median(z_valid)
            mad = np.median(np.abs(z_valid - med))
            thr = mad_k * 1.4826 * mad
        elif method == "abs":
            thr = abs_threshold
        elif method == "otsu":
            thr = _otsu_threshold(np.abs(z_valid))
        else:
            raise ValueError("method must be 'mad', 'abs', or 'otsu'")

        thr_used[j] = thr

        in_track = np.zeros(n_y, dtype=bool)
        in_track[finite] = np.abs(z_valid) > thr

        run = _longest_true_run(in_track, min_run_px)
        if run is None:
            dropped[j] = True
            continue

        i0, i1 = run  # inclusive indices
        left[j] = y[i0]
        right[j] = y[i1]
        width[j] = y[i1] - y[i0]

    profile = WidthProfile(
        x=x, left=left, right=right, width=width,
        valid_frac=valid_frac, dropped=dropped,
        method=method, threshold_used=thr_used,
    )

    if smooth_window:
        profile.width = _moving_average_nan(profile.width, smooth_window)

    return profile


def _longest_true_run(mask, min_run_px):
    """Return (start_idx, end_idx) of the longest run of True values in
    `mask` with length >= min_run_px, or None if no run qualifies."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None

    best = None
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        if prev - run_start + 1 >= min_run_px:
            if best is None or (prev - run_start) > (best[1] - best[0]):
                best = (run_start, prev)
        run_start = i
        prev = i
    if prev - run_start + 1 >= min_run_px:
        if best is None or (prev - run_start) > (best[1] - best[0]):
            best = (run_start, prev)

    return best


def _otsu_threshold(values):
    """Simple Otsu threshold on a 1D array of non-negative values."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / np.maximum(weight2, 1)
    inter_class_var = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(inter_class_var)
    return bin_centers[idx]


def _moving_average_nan(arr, window):
    """Centered moving average that ignores NaNs, window must be odd."""
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = arr.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = arr[lo:hi]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out


# ------------------------------------------------------------------
# 3b. Restrict boundary/width results to the laser-on window
# ------------------------------------------------------------------

def laser_on_x_extent(thermal_result, mm_per_frame):
    """
    Convert a thermal result's laser-on frame indices ('on_start', 'on_stop')
    into a physical x extent (mm), using the extracted window's known
    x_mm_center mapping as the reference.

    Parameters
    ----------
    thermal_result : dict returned by extract_final_thermal_frames, must
                      contain 'x_mm_center', 'start_idx', 'on_start', 'on_stop'
    mm_per_frame    : physical scan distance per raw thermal frame, mm
                      (e.g. THERMAL_MM_PER_FRAME = SCAN_SPEED_MM_PER_S / THERMAL_FPS)

    Returns
    -------
    (x_on_start, x_on_stop) : floats, mm. x_on_start may be < the extracted
    window's first x if the laser turned on before the 20-100 mm crop began,
    and likewise x_on_stop may exceed the extracted window's last x.
    """
    x_mm_center = np.asarray(thermal_result["x_mm_center"], dtype=float)
    start_idx = thermal_result["start_idx"]
    on_start = thermal_result["on_start"]
    on_stop = thermal_result["on_stop"]

    x0 = x_mm_center[0]
    x_on_start = x0 + (on_start - start_idx) * mm_per_frame
    x_on_stop = x0 + (on_stop - start_idx) * mm_per_frame
    return float(x_on_start), float(x_on_stop)


def restrict_profile_to_x_range(profile, x_lo, x_hi, clip_to_available=True):
    """
    Mask a WidthProfile so that L(x), R(x), width(x) are NaN outside
    [x_lo, x_hi] (e.g. the laser-on window). Values inside the range are
    left untouched, including any that were already NaN from dropout.

    If the requested range extends beyond the data actually available in
    `profile.x` and `clip_to_available` is True, the range is silently
    clipped to what's available and a note is printed -- this happens
    whenever the laser-on window is wider than the 20-100 mm common crop,
    which is expected and not an error.

    Returns
    -------
    A new WidthProfile with the same x array, masked left/right/width.
    """
    x = profile.x
    lo, hi = x_lo, x_hi
    if clip_to_available:
        data_lo, data_hi = float(x.min()), float(x.max())
        clipped = False
        if lo < data_lo:
            lo = data_lo
            clipped = True
        if hi > data_hi:
            hi = data_hi
            clipped = True
        if clipped:
            print(
                f"[restrict_profile_to_x_range] requested [{x_lo:.2f}, {x_hi:.2f}] mm "
                f"clipped to available data range [{data_lo:.2f}, {data_hi:.2f}] mm"
            )

    in_range = (x >= lo) & (x <= hi)

    new_left = np.where(in_range, profile.left, np.nan)
    new_right = np.where(in_range, profile.right, np.nan)
    new_width = np.where(in_range, profile.width, np.nan)

    return WidthProfile(
        x=x,
        left=new_left,
        right=new_right,
        width=new_width,
        valid_frac=profile.valid_frac,
        dropped=profile.dropped | ~in_range,
        method=profile.method,
        threshold_used=profile.threshold_used,
    )


# ------------------------------------------------------------------
# 4. Plotting helper (optional, requires matplotlib)
# ------------------------------------------------------------------

def plot_width_profile(profile, track_id=None, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    else:
        fig = ax[0].figure

    ax[0].plot(profile.x, profile.width, lw=1.0)
    ax[0].set_ylabel("width W(x) (mm)")
    title = "Local width profile" if track_id is None else f"Track {track_id}: local width profile"
    ax[0].set_title(title)

    ax[1].plot(profile.x, profile.left, lw=1.0, label="left boundary L(x)")
    ax[1].plot(profile.x, profile.right, lw=1.0, label="right boundary R(x)")
    ax[1].set_ylabel("y (mm)")
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].plot(profile.x, profile.valid_frac, lw=1.0, color="gray")
    ax[2].axhline(0.5, ls="--", color="k", lw=0.8)
    ax[2].fill_between(profile.x, 0, 1, where=profile.dropped, color="red", alpha=0.15,
                        label="dropped (low valid_frac)")
    ax[2].set_ylabel("valid fraction")
    ax[2].set_xlabel("actual x (mm)")
    ax[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# 5. Example usage (against the notebook's existing loaded objects)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Illustrative only — assumes `load_wyko_asc` / `robust_plane_detrend`
    # are already imported and HEIGHT_DIR / track_id are already defined,
    # exactly as in cells 25-27 of the starter notebook.
    #
    # height = load_wyko_asc(HEIGHT_DIR, track_id, crop_to_common=True)
    # Z_detrended, coef = robust_plane_detrend(
    #     height['Z_mm'], height['x_actual_mm'], height['y_mm']
    # )
    #
    # Sa, Sq, Sz = areal_roughness(Z_detrended)
    # print(f"Sa={Sa*1000:.2f} um, Sq={Sq*1000:.2f} um, Sz={Sz*1000:.2f} um")
    #
    # diag = column_dropout_diagnostic(Z_detrended, height['x_actual_mm'], height['y_mm'])
    # # inspect diag['n_gaps'] and diag['valid_frac'] before trusting width(x)
    #
    # profile = local_width_profile(
    #     Z_detrended, height['x_actual_mm'], height['y_mm'],
    #     method='mad', mad_k=3.0, min_valid_frac=0.5, min_run_px=3,
    # )
    # plot_width_profile(profile, track_id=track_id)
    pass
