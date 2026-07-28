"""
Builds a tabular feature dataset for the width/boundary prediction task,
one row per sampled x location, with a compact set of physically motivated
features from the thermal frame and SEM crop rather than raw pixel arrays.

This is deliberately not the same data pipeline as multimodal_dataset.py.
That one streams resized images for the image-to-image U-Net. This one
produces a small dense feature table meant for GPR / NGBoost, where n is
around 300 and the input should be a handful of meaningful numbers per
sample rather than a 128 by 128 array.

Reuses the sampling, caching, and track-data helpers from
multimodal_dataset.py rather than duplicating them.
"""

import numpy as np
import pandas as pd
from scipy import ndimage

from multimodal_dataset import (
    _sample_x_locations,
    _x_fraction,
    cached_get_sem_tile_paths,
    cached_load_sem_tile,
)
from frame_extraction import get_track_data, get_thermal_frame
from sem_multimodal_bundle import get_multimodal_bundle_at_x


THERMAL_PIXEL_SIZE_MM = 0.014  # 14 micron/pixel, per the challenge paper
THERMAL_BACKGROUND_THRESHOLD = 1200  # a bit above the ~1000 floor on the fixed colorbar
THERMAL_MM_PER_FRAME = 0.2  # confirmed in the paper's Eq. 2: 10mm/s scan speed / 50fps
SCAN_SPEED_MM_S = 10.0  # fixed across every track, per the paper

# NOTE on this mapping: the paper states four laser powers (200, 300, 350, 400 W)
# and the four released track IDs (8, 10, 14, 21) in the same parallel sentence
# structure, which strongly suggests a direct 1-to-1 correspondence in that order,
# but it is never given as an explicit table. Treat this as an inferred working
# assumption, not a confirmed fact, worth double-checking against any process log
# or metadata file if one exists. It's also worth noting this sits in some tension
# with what the feature coverage checks found earlier, Track_8's melt_peak values
# cluster near the top of the whole dataset's range, which reads more like a high
# power track than the lowest one. If you can confirm the real mapping, update
# this dict directly.
TRACK_LASER_POWER_W = {
    "Track_8": 200.0,
    "Track_10": 300.0,
    "Track_14": 350.0,
    "Track_21": 400.0,
}


def linear_energy_density(track_id):
    """Laser power divided by scan speed (J/mm), a standard process-physics
    quantity, using TRACK_LASER_POWER_W above. Returns NaN for an unknown
    track_id rather than raising, so a bad/missing entry shows up as a
    missing value in the feature table instead of crashing extraction."""
    power = TRACK_LASER_POWER_W.get(track_id)
    if power is None:
        return float("nan")
    return power / SCAN_SPEED_MM_S


# ---------------------------------------------------------------------
# thermal features
# ---------------------------------------------------------------------

def extract_thermal_features(T, threshold=THERMAL_BACKGROUND_THRESHOLD,
                              pixel_size_mm=THERMAL_PIXEL_SIZE_MM):
    """
    Melt-pool descriptors from a single thermal frame: size, peak and mean
    intensity, elongation/aspect ratio, asymmetry of the hot region, and a
    cooling-tail ratio capturing the smaller secondary warm blob that
    trails the main melt pool in these frames.
    """
    T = np.asarray(T, dtype=np.float32)
    hot = T > threshold

    if not hot.any():
        return {
            "melt_area_mm2": 0.0,
            "melt_peak": float(T.max()),
            "melt_mean": float(T.mean()),
            "melt_aspect_ratio": 1.0,
            "melt_asymmetry": 0.0,
            "melt_tail_ratio": 0.0,
        }

    labeled, n_labels = ndimage.label(hot)
    sizes = ndimage.sum(hot, labeled, index=range(1, n_labels + 1))
    order = np.argsort(sizes)[::-1]  # largest region first
    main_label = 1 + int(order[0])
    main_mask = labeled == main_label

    ys, xs = np.nonzero(main_mask)
    area_px = main_mask.sum()

    feats = {
        "melt_area_mm2": float(area_px * pixel_size_mm ** 2),
        "melt_peak": float(T[main_mask].max()),
        "melt_mean": float(T[main_mask].mean()),
    }

    # elongation of the hot region, via the covariance of pixel coordinates
    # weighted by intensity above threshold
    weights = np.clip(T[main_mask] - threshold, 1e-3, None)
    cy = np.average(ys, weights=weights)
    cx = np.average(xs, weights=weights)
    cov = np.cov(np.stack([ys - cy, xs - cx]), aweights=weights)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 1e-6, None)
    feats["melt_aspect_ratio"] = float(np.sqrt(eigvals[-1] / eigvals[0]))

    # asymmetry, distance between the intensity centroid and the plain
    # geometric centroid of the same region, normalized by an equivalent
    # radius so it's comparable across differently sized melt pools
    geo_cy, geo_cx = ys.mean(), xs.mean()
    offset = np.hypot(cy - geo_cy, cx - geo_cx)
    equiv_radius = np.sqrt(area_px / np.pi)
    feats["melt_asymmetry"] = float(offset / equiv_radius) if equiv_radius > 0 else 0.0

    # cooling tail, ratio of the second largest hot region's peak intensity
    # to the main region's peak, 0 if there's no separate secondary region
    if n_labels > 1:
        second_label = 1 + int(order[1])
        second_mask = labeled == second_label
        feats["melt_tail_ratio"] = float(T[second_mask].max() / feats["melt_peak"])
    else:
        feats["melt_tail_ratio"] = 0.0

    return feats


# ---------------------------------------------------------------------
# SEM features
# ---------------------------------------------------------------------

def extract_sem_features(sem_img, black_row_threshold=1.0):
    """
    Substrate texture descriptors from the box-masked SEM crop. Rows that
    fall inside the black masked band (mean intensity near zero) are
    excluded first, so the roughness statistics describe the surrounding
    substrate only, never the masked-out track region.
    """
    sem_img = np.asarray(sem_img, dtype=np.float32)
    row_means = sem_img.mean(axis=1)
    masked_rows = row_means < black_row_threshold
    substrate = sem_img[~masked_rows] if (~masked_rows).any() else sem_img

    gy, gx = np.gradient(substrate)
    grad_mag = np.hypot(gy, gx)

    return {
        "sem_mean": float(substrate.mean()),
        "sem_std": float(substrate.std()),
        "sem_texture_mean": float(grad_mag.mean()),
        "sem_texture_std": float(grad_mag.std()),
        # peak local roughness. NOTE: an earlier version of this used the
        # fraction of pixels above the sample's own 90th percentile
        # gradient, which sits at ~0.1 for nearly any image by
        # construction and carries no real signal, this replaces it.
        "sem_texture_max": float(grad_mag.max()),
    }


def crop_sem_local_window(sem_masked_image, sem_x_lo_mm, sem_x_hi_mm, x_value, window_mm):
    """
    Crops the SEM tile down to a narrow physical window centered on
    x_value, using the tile's known [sem_x_lo_mm, sem_x_hi_mm) range.
    Column position within a tile maps linearly onto that range, since
    load_track_tiles' 180-degree rotation is specifically what makes tiles
    stitch together consistently left to right in physical x, so a linear
    column-to-mm mapping is valid within a single tile.

    Clips to the tile's actual columns if the requested window would run
    past this tile's edge, since a location near a tile boundary can't
    borrow columns from the neighboring tile here.
    """
    w = sem_masked_image.shape[1]
    tile_span_mm = sem_x_hi_mm - sem_x_lo_mm
    if tile_span_mm <= 0:
        return sem_masked_image

    col_center = (x_value - sem_x_lo_mm) / tile_span_mm * w
    half_window_px = (window_mm / 2.0) / tile_span_mm * w

    col_start = int(np.clip(np.floor(col_center - half_window_px), 0, w))
    col_end = int(np.clip(np.ceil(col_center + half_window_px), 0, w))
    if col_end <= col_start:
        return sem_masked_image  # degenerate window, fall back to the full tile

    return sem_masked_image[:, col_start:col_end]


# ---------------------------------------------------------------------
# multi-frame thermal features: cooling rate and spatial gradient
# ---------------------------------------------------------------------

def get_thermal_stack(x_value, track_id, track_data=None, n_frames=5, frame_spacing=1,
                       mm_per_frame=THERMAL_MM_PER_FRAME):
    """
    Returns an (n_frames, H, W) array of thermal frames leading up to and
    including x_value, oldest first, so stack[-1] is the frame nearest
    x_value itself, matching what a single-frame extraction would have
    pulled.

    This is what unlocks a real cooling-rate feature, G and R from the
    literature review are fundamentally rate/gradient quantities, a single
    static frame can't carry that information no matter how it's processed.

    Tries get_thermal_frame(x_value, track_id=..., track_data=track_data)
    first, matching the track_data-aware calling convention the rest of
    this codebase uses, falls back to the plain (x_value, track_id) form
    if that raises. If get_thermal_frame's real signature differs from
    both of these, this is the one place to adjust it.
    """
    offsets = [-(n_frames - 1 - i) * frame_spacing * mm_per_frame for i in range(n_frames)]
    frames = []
    for off in offsets:
        x_try = x_value + off
        try:
            _, T = get_thermal_frame(x_try, track_id=track_id, track_data=track_data, display=False)
        except TypeError:
            _, T = get_thermal_frame(x_try, track_id=track_id, display=False)
        except Exception:
            # x_try likely ran off the start of the track's common range,
            # repeat the requested frame rather than fail the whole sample
            try:
                _, T = get_thermal_frame(x_value, track_id=track_id, track_data=track_data, display=False)
            except TypeError:
                _, T = get_thermal_frame(x_value, track_id=track_id, display=False)
        frames.append(np.asarray(T, dtype=np.float32))
    return np.stack(frames)


def extract_cooling_features(stack, threshold=THERMAL_BACKGROUND_THRESHOLD,
                              pixel_size_mm=THERMAL_PIXEL_SIZE_MM,
                              mm_per_frame=THERMAL_MM_PER_FRAME, frame_spacing=1,
                              scan_speed_mm_s=SCAN_SPEED_MM_S):
    """
    Two physics-motivated features from a thermal frame stack, corresponding
    to R (cooling rate) and G (thermal gradient) from the literature.

    cooling_rate_proxy: slope of peak frame intensity over time (converting
    the fixed 0.2mm/frame spacing to real time via the known 10mm/s scan
    speed), sign-flipped so a positive value means cooling. This is a proxy
    for R, real thermal gradient/cooling rate work (5-20 K/um, 1-40 K/us in
    the literature) requires calibrated absolute temperature, which this
    sensor's raw intensity units aren't guaranteed to be, but the relative
    trend across frames is still a genuine physical signal.

    thermal_gradient_proxy: mean spatial gradient magnitude at the melt
    pool's boundary in the most recent frame, in intensity units per mm,
    a proxy for G.
    """
    n = stack.shape[0]
    peak_per_frame = np.array([float(f.max()) for f in stack])
    time_per_frame_s = (mm_per_frame * frame_spacing) / scan_speed_mm_s
    t = np.arange(n) * time_per_frame_s

    if n >= 2 and np.ptp(t) > 0:
        slope, _ = np.polyfit(t, peak_per_frame, 1)
    else:
        slope = 0.0
    cooling_rate_proxy = float(-slope)

    latest = stack[-1]
    hot = latest > threshold
    if hot.any():
        gy, gx = np.gradient(latest, pixel_size_mm)
        grad_mag = np.hypot(gy, gx)
        boundary = hot ^ ndimage.binary_erosion(hot)
        if boundary.any():
            thermal_gradient_proxy = float(grad_mag[boundary].mean())
        else:
            thermal_gradient_proxy = float(grad_mag[hot].mean())
    else:
        thermal_gradient_proxy = 0.0

    return {
        "cooling_rate_proxy": cooling_rate_proxy,
        "thermal_gradient_proxy": thermal_gradient_proxy,
    }


def _restrict_to_band(Z_crop, y_mm, y_left_mm, y_right_mm):
    """Restricts the height crop to the rows already established as the
    track (between y_left_mm and y_right_mm), the same boundary used for
    width, so every descriptor derived below is defined consistently with
    it rather than by a separate segmentation rule."""
    y = np.asarray(y_mm, dtype=np.float64)
    mask = (y >= y_left_mm) & (y <= y_right_mm)
    if mask.sum() < 5:
        mask = np.ones_like(y, dtype=bool)  # band too narrow/degenerate, fall back to everything
    return np.asarray(Z_crop)[mask, :], y[mask]


def extract_contour_descriptors(Z_crop, y_mm):
    """
    Cross-track height-profile shape descriptors, computed within the
    already-established track band (call _restrict_to_band first).

    Columns in Z_crop are averaged to get one representative 1D profile
    z(y) at this x location, robust to the small per-column noise already
    checked elsewhere in this project (see the nearby-pair discrepancy
    diagnostic).

    centerline_y_mm: y position of the deepest point, an independent
      estimate of the track's actual centerline, not assumed to sit at the
      midpoint of y_left/y_right.
    peak_depth_mm: how deep the track cuts at its deepest point.
    profile_skewness: shape asymmetry of the cross-section, left vs right.
    profile_waviness_mm: RMS residual after removing a smooth quadratic
      trend, a roughness/waviness descriptor distinct from width or depth.

    Units: same as Z_crop itself (this project keeps height in mm
    throughout), not converted to microns, to stay consistent with
    width_mean_mm and the rest of this table.
    """
    Z = np.asarray(Z_crop, dtype=np.float64)
    y = np.asarray(y_mm, dtype=np.float64)
    profile = np.nanmean(Z, axis=1)
    valid = ~np.isnan(profile)

    if valid.sum() < 5:
        return {
            "centerline_y_mm": np.nan, "peak_depth_mm": np.nan,
            "profile_skewness": np.nan, "profile_waviness_mm": np.nan,
        }

    y_v = y[valid]
    z_v = profile[valid]

    idx_min = int(np.argmin(z_v))
    centerline_y_mm = float(y_v[idx_min])
    peak_depth_mm = float(z_v[idx_min])

    mean_z, std_z = z_v.mean(), z_v.std()
    skewness = float(np.mean(((z_v - mean_z) / std_z) ** 3)) if std_z > 1e-9 else 0.0

    if len(y_v) >= 4:
        coeffs = np.polyfit(y_v, z_v, 2)
        residual = z_v - np.polyval(coeffs, y_v)
        waviness = float(np.sqrt(np.mean(residual ** 2)))
    else:
        waviness = 0.0

    return {
        "centerline_y_mm": centerline_y_mm,
        "peak_depth_mm": peak_depth_mm,
        "profile_skewness": skewness,
        "profile_waviness_mm": waviness,
    }


# ---------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------

def _extract_one_row(
    track_id, x_val, track_data, rng, width_window_mm,
    SEM_DIR, SEM_TILE_WIDTH_MM, get_sem_tile_paths_fn, load_sem_tile_fn,
    sem_method, sem_box_mode, sem_mask_mode, sem_window_mm, max_retries,
    n_thermal_frames=5, thermal_frame_spacing=1,
):
    """Builds one feature row, retrying with a fresh x location a few times
    if this spot has no valid width/boundary reading (e.g. a location with
    entirely missing profilometer coverage) or the bundle extraction fails
    outright."""
    for _ in range(max_retries):
        try:
            bundle = get_multimodal_bundle_at_x(
                x_val,
                track_id=track_id,
                width_window_mm=width_window_mm,
                display=False,
                track_data=track_data,
                include_sem=True,
                SEM_DIR=SEM_DIR,
                get_sem_tile_paths=get_sem_tile_paths_fn,
                load_sem_tile=load_sem_tile_fn,
                SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
                sem_method=sem_method,
                sem_box_mode=sem_box_mode,
                sem_mask_mode=sem_mask_mode,
            )
            y_left = bundle.get("y_left_mm", np.nan)
            y_right = bundle.get("y_right_mm", np.nan)
            width_mean = bundle.get("width_mean_mm", np.nan)
            if any(np.isnan(v) for v in (y_left, y_right, width_mean)):
                raise ValueError("no valid width/boundary reading at this location")

            thermal_feats = extract_thermal_features(bundle["thermal_frame"])

            # multi-frame stack for cooling rate / spatial gradient, see
            # get_thermal_stack's docstring for why a single static frame
            # can't carry this information
            if n_thermal_frames and n_thermal_frames > 1:
                stack = get_thermal_stack(
                    x_val, track_id, track_data=track_data,
                    n_frames=n_thermal_frames, frame_spacing=thermal_frame_spacing,
                )
                cooling_feats = extract_cooling_features(stack, frame_spacing=thermal_frame_spacing)
            else:
                cooling_feats = {"cooling_rate_proxy": np.nan, "thermal_gradient_proxy": np.nan}

            sem_crop = bundle["sem_masked_image"]
            if sem_window_mm is not None and "sem_x_lo_mm" in bundle and "sem_x_hi_mm" in bundle:
                sem_crop = crop_sem_local_window(
                    sem_crop, bundle["sem_x_lo_mm"], bundle["sem_x_hi_mm"], x_val, sem_window_mm
                )
            sem_feats = extract_sem_features(sem_crop)

            # contour/profile-shape descriptors, from height data already
            # fetched for the width/boundary targets above, no extra I/O
            Z_band, y_band = _restrict_to_band(
                bundle["height_Z_crop"], bundle["height_y_mm"], y_left, y_right
            )
            contour_feats = extract_contour_descriptors(Z_band, y_band)
            if any(np.isnan(v) for v in contour_feats.values()):
                # the width/boundary check above only guarantees enough
                # valid height data existed *somewhere* in this crop, not
                # specifically within the restricted y_left..y_right band,
                # given how much of this dataset's height maps are missing
                # (roughly 37-55% NaN per track), that band can still come
                # up short of the 5-point minimum extract_contour_descriptors
                # needs, retry at a new location rather than passing NaN
                # descriptors downstream into a GP fit
                raise ValueError("insufficient valid height data in the track band for contour descriptors")

            x_mm = bundle["x_requested_mm"]

            return {
                "track_id": track_id,
                "x_mm": x_mm,
                "x_frac": _x_fraction(track_id, x_mm),
                **thermal_feats,
                **cooling_feats,
                **sem_feats,
                "linear_energy_density": linear_energy_density(track_id),
                "y_left_mm": y_left,
                "y_right_mm": y_right,
                "width_mean_mm": width_mean,
                "width_std_mm": bundle.get("width_std_mm", np.nan),
                **contour_feats,
            }
        except Exception:
            x_val = float(_sample_x_locations(track_id, 1, rng)[0])
    return None


def build_feature_dataset(
    track_ids,
    n_per_track,
    width_window_mm=1.0,
    sem_window_mm=1.5,
    seed=0,
    SEM_DIR=None,
    SEM_TILE_WIDTH_MM=None,
    get_sem_tile_paths_fn=cached_get_sem_tile_paths,
    load_sem_tile_fn=cached_load_sem_tile,
    sem_method="box",
    sem_box_mode="envelope",
    sem_mask_mode="cover_track",
    max_retries_per_sample=5,
    n_thermal_frames=5,
    thermal_frame_spacing=1,
):
    """
    Samples n_per_track locations from each track in track_ids and returns
    a pandas DataFrame with one row per sample, feature columns first,
    then y_left_mm, y_right_mm, width_mean_mm, width_std_mm as targets.

    sem_window_mm crops the SEM tile down to this physical width (in mm)
    centered on each sample's x location before computing texture
    features, rather than using the whole ~6.41mm tile. Set to None to use
    the full tile (the old behavior).

    n_thermal_frames pulls this many consecutive thermal frames leading up
    to each sample location (native spacing 0.2mm/frame times
    thermal_frame_spacing) instead of a single static frame, used to
    compute cooling_rate_proxy and thermal_gradient_proxy. Set to 1 (or 0)
    to fall back to single-frame behavior only.

    seed makes this reproducible, unlike the epoch-varying sampling in
    multimodal_dataset.py, since a GPR/NGBoost fit works off one fixed
    table rather than a new draw every epoch.
    """
    rng = np.random.default_rng(seed)
    track_cache = {tid: get_track_data(tid) for tid in set(track_ids)}

    rows = []
    for track_id in track_ids:
        for x_val in _sample_x_locations(track_id, n_per_track, rng):
            row = _extract_one_row(
                track_id, float(x_val), track_cache[track_id], rng, width_window_mm,
                SEM_DIR, SEM_TILE_WIDTH_MM, get_sem_tile_paths_fn, load_sem_tile_fn,
                sem_method, sem_box_mode, sem_mask_mode, sem_window_mm, max_retries_per_sample,
                n_thermal_frames=n_thermal_frames, thermal_frame_spacing=thermal_frame_spacing,
            )
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    train_df = build_feature_dataset(
        track_ids=["Track_8", "Track_10", "Track_14"],
        n_per_track=100,
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        seed=0,
    )
    test_df = build_feature_dataset(
        track_ids=["Track_21"],
        n_per_track=100,
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        seed=42,
    )

    print("train shape", train_df.shape)
    print(train_df.head())
    print("test shape", test_df.shape)

    train_df.to_csv("train_features.csv", index=False)
    test_df.to_csv("test_features.csv", index=False)