"""
sem_masking.py

Segments the laser track from the surrounding substrate in a stitched SEM
panorama (see sem_stitching.py), producing a binary mask per track. This
gives an independent, SEM-derived width signal that can be cross-checked
against the Bruker height-map-derived width from height_map_descriptors.py.

Usage
-----
    from sem_stitching import stitch_all_tracks
    from sem_masking import mask_all_tracks, plot_mask_overlay, mask_width_profile

    sem_results = stitch_all_tracks(TRACK_IDS, SEM_DIR, get_sem_tile_paths,
                                     load_sem_tile, SEM_TILE_WIDTH_MM,
                                     save_dir=FIGURES_DIR)

    masks = mask_all_tracks(sem_results, save_dir=FIGURES_DIR)

    for tid, m in masks.items():
        plot_mask_overlay(sem_results[tid], m)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt


def mask_panorama_to_x_range(stitch_result, x_lo, x_hi, mode="crop", clip_to_available=True):
    """
    Restrict a stitched SEM panorama to a physical x-range -- typically the
    laser-on-to-off window from laser_on_x_extent() in
    height_map_descriptors.py -- so the SEM view covers the same physical
    extent as the Bruker/thermal analysis, rather than the full stitched
    tile sequence (which may include tiles captured before laser-on or
    after laser-off).

    This is a spatial crop/mask driven by known metadata, distinct from
    mask_panorama()'s intensity-based track/substrate segmentation.

    Parameters
    ----------
    stitch_result       : StitchResult from sem_stitching.stitch_track
    x_lo, x_hi           : physical x bounds to keep, mm (e.g. from
                            height_map_descriptors.laser_on_x_extent)
    mode                  : 'crop' returns a smaller panorama containing only
                            the in-range columns (recommended -- output size
                            shrinks to match). 'blackout' keeps the original
                            panorama size and zeroes out-of-range columns
                            (useful if you need consistent image dimensions
                            across tracks with different laser-on extents).
    clip_to_available     : if the requested range extends beyond the
                            panorama's actual x-extent, clip to what's
                            available and print a note rather than error

    Returns
    -------
    A new object with the same fields as StitchResult (panorama, x_start_mm,
    width_mm, height_mm, track_id, method, n_tiles_used, n_tiles_total,
    tile_paths) so it's a drop-in replacement anywhere a StitchResult is
    used downstream (plot_panorama, mask_panorama, mask_width_profile, etc).
    """
    from copy import copy

    x0, w_mm = stitch_result.x_start_mm, stitch_result.width_mm
    w_px = stitch_result.panorama.shape[1]
    px_to_mm = w_mm / w_px

    lo, hi = x_lo, x_hi
    data_lo, data_hi = x0, x0 + w_mm
    if clip_to_available:
        clipped = False
        if lo < data_lo:
            lo = data_lo
            clipped = True
        if hi > data_hi:
            hi = data_hi
            clipped = True
        if clipped:
            print(f"[mask_panorama_to_x_range] track {stitch_result.track_id}: "
                  f"requested [{x_lo:.2f}, {x_hi:.2f}] mm clipped to available "
                  f"panorama range [{data_lo:.2f}, {data_hi:.2f}] mm")

    col_lo = int(round((lo - x0) / px_to_mm))
    col_hi = int(round((hi - x0) / px_to_mm))
    col_lo = max(0, col_lo)
    col_hi = min(w_px, col_hi)

    result = copy(stitch_result)

    if mode == "crop":
        result.panorama = stitch_result.panorama[:, col_lo:col_hi]
        result.x_start_mm = x0 + col_lo * px_to_mm
        result.width_mm = (col_hi - col_lo) * px_to_mm
    elif mode == "blackout":
        panorama = stitch_result.panorama.copy()
        panorama[:, :col_lo] = 0
        panorama[:, col_hi:] = 0
        result.panorama = panorama
        # x_start_mm / width_mm stay the same in blackout mode -- the
        # panorama's physical extent is unchanged, just parts are zeroed
    else:
        raise ValueError("mode must be 'crop' or 'blackout'")

    return result


def mask_all_tracks_to_laser_on(stitch_results, thermal_results, mm_per_frame,
                                 mode="crop", save_dir=None):
    """
    Convenience wrapper: for every track, look up its laser-on x-extent
    from thermal_results (via height_map_descriptors.laser_on_x_extent)
    and restrict the SEM panorama to that window.

    Parameters
    ----------
    stitch_results  : dict {track_id: StitchResult}
    thermal_results : dict {track_id: thermal extraction result dict},
                       each containing 'x_mm_center', 'start_idx',
                       'on_start', 'on_stop'
    mm_per_frame     : THERMAL_MM_PER_FRAME
    mode              : 'crop' or 'blackout', passed to mask_panorama_to_x_range
    save_dir          : if given, saves each restricted panorama as
                        sem_panorama_laser_on_track_<id>.png

    Returns
    -------
    dict {track_id: restricted StitchResult-like object}
    """
    from height_map_descriptors import laser_on_x_extent

    restricted = {}
    for tid, result in stitch_results.items():
        x_lo, x_hi = laser_on_x_extent(thermal_results[tid], mm_per_frame)
        panorama_lo = result.x_start_mm
        panorama_hi = result.x_start_mm + result.width_mm

        fully_covered = (x_lo <= panorama_lo) and (x_hi >= panorama_hi)

        r = mask_panorama_to_x_range(result, x_lo, x_hi, mode=mode)
        restricted[tid] = r

        if fully_covered:
            print(f"Track {tid}: laser-on x-range=[{x_lo:.2f}, {x_hi:.2f}] mm fully "
                  f"covers the panorama's visible range [{panorama_lo:.2f}, {panorama_hi:.2f}] mm "
                  f"-> no laser-off region to remove, panorama unchanged")
        else:
            print(f"Track {tid}: laser-on x-range=[{x_lo:.2f}, {x_hi:.2f}] mm -> "
                  f"panorama trimmed to [{r.x_start_mm:.2f}, {r.x_start_mm + r.width_mm:.2f}] mm")

        if save_dir is not None:
            out_path = Path(save_dir) / f"sem_panorama_laser_on_track_{tid}.png"
            cv2.imwrite(str(out_path), r.panorama)

    return restricted


def _percentile_threshold_and_close(response_img, threshold_percentile, close_kernel):
    """
    Shared thresholding step: binarize `response_img` at its
    `threshold_percentile`, then morphologically close with `close_kernel`
    to bridge fragments. Used by both mask_panorama_horizontal_tophat and
    mask_enhanced_image so the two masking approaches stay consistent.

    Uses response_img >= threshold_value directly (not cv2.threshold's
    THRESH_BINARY, which applies a strict > comparison). This matters
    whenever more than (100 - threshold_percentile)% of pixels sit exactly
    at a saturated/plateau value (e.g. an enhancement step that clips to
    255) -- with a strict >, the threshold value itself equals the
    saturation value and nothing passes, silently producing an empty mask.
    """
    response_img = response_img if response_img.dtype == np.uint8 else response_img.astype(np.uint8)

    saturated_frac = np.mean(response_img == response_img.max())
    if saturated_frac > (100 - threshold_percentile) / 100.0:
        print(f"[_percentile_threshold_and_close] warning: {saturated_frac*100:.1f}% of pixels "
              f"are saturated at the max value ({response_img.max()}), which is >= the "
              f"{100 - threshold_percentile:.1f}% top slice requested by threshold_percentile="
              f"{threshold_percentile}. The mask will include all saturated pixels regardless "
              f"of the percentile setting -- consider reducing enhancement_strength if this "
              f"wasn't intended.")

    thr_value = np.percentile(response_img, threshold_percentile)
    mask = np.where(response_img >= thr_value, 255, 0).astype(np.uint8)

    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, connect_kernel)
    return mask


def compute_enhanced_image(
    panorama,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    tophat_kernel=(101, 3),
    enhancement_strength=0.6,
):
    """
    CLAHE-equalize the panorama, extract the horizontal top-hat response,
    and blend the two into a single "enhanced" image where horizontal
    track-like structure is boosted on top of the base contrast-corrected
    grayscale -- matches the enhancement pipeline you're already using
    (gray_eq + enhancement_strength * horizontal_norm, clipped to uint8).

    Returns
    -------
    enhanced, gray, gray_eq, horizontal_norm
        enhanced        : uint8, the blended enhancement image
        gray             : uint8, original grayscale (pre-CLAHE)
        gray_eq          : uint8, CLAHE-equalized grayscale
        horizontal_norm  : float32, normalized (0-255) top-hat response
    """
    gray = _to_gray(panorama)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    gray_eq = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tophat_kernel)
    horizontal = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel)
    horizontal_norm = cv2.normalize(horizontal, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    gray_f = gray_eq.astype(np.float32)
    enhanced = gray_f + enhancement_strength * horizontal_norm
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced, gray, gray_eq, horizontal_norm


def mask_enhanced_image(enhanced, threshold_percentile=95, close_kernel=(51, 5)):
    """
    Threshold the blended "enhanced" image (from compute_enhanced_image)
    at threshold_percentile and close to bridge fragments -- masks the
    enhanced image directly, as opposed to mask_panorama_horizontal_tophat
    which masks the raw top-hat response before any blending.

    Returns
    -------
    mask, coverage_pct
    """
    mask = _percentile_threshold_and_close(enhanced, threshold_percentile, close_kernel)
    coverage_pct = 100 * np.mean(mask > 0)
    return mask, coverage_pct


def mask_all_tracks_enhanced(
    sem_results,
    track_ids=None,
    save_dir=None,
    threshold_percentile=95,
    close_kernel=(51, 5),
    enhancement_strength=0.6,
    **enhance_kwargs,
):
    """
    Compute the enhanced image and its mask for every track, plotting all
    four stages (original, horizontal response, enhanced, mask) and
    optionally saving both the enhanced image and the mask per track.

    Parameters
    ----------
    sem_results             : dict {track_id: StitchResult}
    track_ids                : list of track ids; defaults to all keys in sem_results
    save_dir                  : if given, saves track_<id>_enhanced.png,
                               track_<id>_enhanced_mask.png, and
                               track_<id>_enhanced_summary.png per track
    threshold_percentile      : percentile cutoff on the enhanced image
    close_kernel               : morphological closing kernel
    enhancement_strength       : blend weight, passed to compute_enhanced_image
    **enhance_kwargs            : any other compute_enhanced_image params
                               (clahe_clip, clahe_tile, tophat_kernel)

    Returns
    -------
    dict {track_id: {'enhanced': ndarray, 'mask': ndarray, 'coverage_pct': float}}
    """
    if track_ids is None:
        track_ids = list(sem_results.keys())

    results = {}
    for tid in track_ids:
        img = sem_results[tid].panorama
        enhanced, gray, gray_eq, horizontal_norm = compute_enhanced_image(
            img, enhancement_strength=enhancement_strength, **enhance_kwargs
        )
        mask, coverage_pct = mask_enhanced_image(
            enhanced, threshold_percentile=threshold_percentile, close_kernel=close_kernel
        )
        print(f"Track {tid}: coverage = {coverage_pct:.2f}%")
        results[tid] = {"enhanced": enhanced, "mask": mask, "coverage_pct": coverage_pct}

        fig = plt.figure(figsize=(24, 5))
        plt.subplot(141)
        plt.imshow(gray, cmap="gray")
        plt.title(f"Track {tid}: Original")
        plt.subplot(142)
        plt.imshow(horizontal_norm, cmap="gray")
        plt.title("Horizontal Response")
        plt.subplot(143)
        plt.imshow(enhanced, cmap="gray")
        plt.title("Enhanced SEM")
        plt.subplot(144)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Mask ({coverage_pct:.2f}%)")
        plt.tight_layout()

        if save_dir is not None:
            save_dir_p = Path(save_dir)
            cv2.imwrite(str(save_dir_p / f"track_{tid}_enhanced.png"), enhanced)
            cv2.imwrite(str(save_dir_p / f"track_{tid}_enhanced_mask.png"), mask)
            plt.savefig(save_dir_p / f"track_{tid}_enhanced_summary.png", dpi=300, bbox_inches="tight")
            print(f"  saved enhanced -> {save_dir_p / f'track_{tid}_enhanced.png'}")
            print(f"  saved mask -> {save_dir_p / f'track_{tid}_enhanced_mask.png'}")

        plt.show()

    return results


def mask_panorama_horizontal_tophat(
    panorama,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    tophat_kernel=(101, 3),
    threshold_percentile=95,
    close_kernel=(51, 5),
):
    """
    Segment a horizontally-running track from a stitched SEM panorama using
    CLAHE contrast enhancement + horizontal top-hat filtering + percentile
    thresholding + morphological closing to bridge fragments.

    This targets the track as a linear horizontal structure rather than
    relying on a global intensity split (unlike mask_panorama's Otsu
    approach), which tends to work better on panoramas where the track has
    subtle or locally-varying contrast against the substrate.

    Parameters
    ----------
    panorama                : BGR or grayscale stitched SEM image
    clahe_clip, clahe_tile    : CLAHE contrast enhancement parameters
    tophat_kernel             : (width, height) structuring element for the
                                top-hat filter; wide and short favors long
                                horizontal features like a scan track
    threshold_percentile      : percentile of the top-hat response used as
                                the binary cutoff -- higher keeps only the
                                strongest responses (less coverage, higher
                                precision); lower keeps more (more coverage,
                                more false positives). This is the knob to
                                sweep -- see sweep_tophat_threshold.
    close_kernel              : structuring element for the closing step
                                that bridges gaps along x between fragments

    Returns
    -------
    mask, coverage_pct, gray_enhanced, horizontal_response
        mask                  : uint8 array, 255=track / 0=background
        coverage_pct          : float, percent of panorama pixels masked
        gray_enhanced          : CLAHE-enhanced grayscale image (for plotting)
        horizontal_response    : top-hat filter response (for plotting)
    """
    gray = _to_gray(panorama)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    gray_enhanced = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tophat_kernel)
    horizontal = cv2.morphologyEx(gray_enhanced, cv2.MORPH_TOPHAT, kernel)

    mask = _percentile_threshold_and_close(horizontal, threshold_percentile, close_kernel)

    coverage_pct = 100 * np.mean(mask > 0)
    return mask, coverage_pct, gray_enhanced, horizontal


def mask_all_tracks_horizontal_tophat(
    sem_results,
    track_ids=None,
    save_dir=None,
    threshold_percentile=95,
    **tophat_kwargs,
):
    """
    Run mask_panorama_horizontal_tophat on every track's stitched panorama,
    save the summary figure and binary mask per track, and return coverage
    stats for all of them.

    Parameters
    ----------
    sem_results            : dict {track_id: StitchResult}, e.g. from
                              stitch_all_tracks or mask_all_tracks_to_laser_on
    track_ids               : list of track ids to process; defaults to all
                              keys in sem_results
    save_dir                 : if given, saves
                              track_<id>_horizontal_mask.png and
                              track_<id>_horizontal_summary.png per track
    threshold_percentile     : passed to mask_panorama_horizontal_tophat
    **tophat_kwargs          : any other mask_panorama_horizontal_tophat params

    Returns
    -------
    dict {track_id: {'mask': ndarray, 'coverage_pct': float}}
    """
    if track_ids is None:
        track_ids = list(sem_results.keys())

    results = {}
    for tid in track_ids:
        img = sem_results[tid].panorama
        mask, coverage_pct, gray_enhanced, horizontal = mask_panorama_horizontal_tophat(
            img, threshold_percentile=threshold_percentile, **tophat_kwargs
        )
        print(f"Track {tid}: coverage = {coverage_pct:.2f}%")
        results[tid] = {"mask": mask, "coverage_pct": coverage_pct}

        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.imshow(gray_enhanced, cmap="gray")
        plt.title(f"Track {tid}: SEM Panorama (CLAHE)")
        plt.subplot(132)
        plt.imshow(horizontal, cmap="gray")
        plt.title("Horizontal Response")
        plt.subplot(133)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Track Candidate ({coverage_pct:.2f}%)")
        plt.tight_layout()

        if save_dir is not None:
            save_dir_p = Path(save_dir)
            mask_path = save_dir_p / f"track_{tid}_horizontal_mask.png"
            summary_path = save_dir_p / f"track_{tid}_horizontal_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches="tight")
            cv2.imwrite(str(mask_path), mask)
            print(f"  saved mask -> {mask_path}")
            print(f"  saved summary -> {summary_path}")

        plt.show()

    return results


def sweep_tophat_threshold(
    sem_results,
    track_id,
    bruker_width_profile=None,
    percentiles=(80, 85, 88, 90, 92, 95, 97, 99),
    **tophat_kwargs,
):
    """
    Sweep threshold_percentile for one track and report coverage at each
    value, so you can see how sensitive the mask is to this parameter
    rather than trusting a single arbitrary choice.

    If bruker_width_profile is given (a WidthProfile from
    height_map_descriptors.local_width_profile for the SAME track), each
    percentile's mean SEM-derived track width (estimated from coverage
    fraction x panorama height) is compared against the Bruker ground
    truth mean width, and the percentile with the closest match is
    reported as the recommended choice -- this is what "best coverage"
    should mean here: not maximum area, but area consistent with the
    independently-measured ground truth.

    Returns
    -------
    list of dicts, one per percentile, each with keys:
        percentile, coverage_pct, mean_width_mm (if StitchResult available),
        width_error_mm (if bruker_width_profile given)
    """
    img = sem_results[track_id].panorama
    result = sem_results[track_id]
    px_to_mm_y = result.height_mm / img.shape[0] if hasattr(result, "height_mm") else None

    bruker_mean_width = None
    if bruker_width_profile is not None:
        bruker_mean_width = float(np.nanmean(bruker_width_profile.width))

    rows = []
    for p in percentiles:
        mask, coverage_pct, _, _ = mask_panorama_horizontal_tophat(
            img, threshold_percentile=p, **tophat_kwargs
        )
        row = {"percentile": p, "coverage_pct": coverage_pct}

        if px_to_mm_y is not None:
            # coverage_pct is already the correct 0/255-safe fraction (see
            # mask_panorama_horizontal_tophat); reuse it rather than
            # re-averaging the raw mask, which is on a 0/255 scale and
            # would inflate this estimate by 255x if averaged directly
            mean_width_mm = (coverage_pct / 100.0) * img.shape[0] * px_to_mm_y
            row["mean_width_mm"] = mean_width_mm
            if bruker_mean_width is not None:
                row["width_error_mm"] = abs(mean_width_mm - bruker_mean_width)

        rows.append(row)
        msg = f"  percentile={p}: coverage={coverage_pct:.2f}%"
        if "mean_width_mm" in row:
            msg += f", est. mean width={row['mean_width_mm']:.3f} mm"
        if "width_error_mm" in row:
            msg += f", |error vs Bruker|={row['width_error_mm']:.3f} mm"
        print(msg)

    if bruker_mean_width is not None:
        best = min(rows, key=lambda r: r["width_error_mm"])
        print(f"\nBruker ground-truth mean width: {bruker_mean_width:.3f} mm")
        print(f"Recommended threshold_percentile={best['percentile']} "
              f"(coverage={best['coverage_pct']:.2f}%, "
              f"est. width={best['mean_width_mm']:.3f} mm, "
              f"error={best['width_error_mm']:.3f} mm)")

    return rows


@dataclass
class MaskResult:
    track_id: int
    mask: np.ndarray            # bool array, shape (H, W), True = track region
    method: str                  # 'otsu' or 'adaptive'
    inverted: bool                # whether the threshold sense was flipped
    coverage_frac: float          # fraction of panorama pixels classified as track
    largest_component_frac: float  # fraction of masked pixels retained after
                                   # keeping only the largest connected component


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def mask_panorama(
    panorama,
    method="otsu",
    blur_ksize=5,
    invert="auto",
    keep_largest_component=True,
    close_kernel=15,
    open_kernel=5,
    fill_holes=True,
    adaptive_block_size=101,
    adaptive_C=2,
):
    """
    Segment the laser track from the substrate in one stitched SEM panorama.

    Pipeline
    --------
    1. Convert to grayscale, Gaussian blur to reduce SEM speckle noise.
    2. Threshold:
         - 'otsu'     : single global threshold (cv2.THRESH_OTSU). Works well
                        if illumination is fairly uniform across the panorama.
         - 'adaptive' : local Gaussian-weighted threshold per neighborhood
                        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C). Use this if
                        stitched tiles show visible brightness banding
                        (common when tiles were captured with slightly
                        different SEM exposure settings).
    3. Auto-detect which side of the threshold is the track: assumes the
       track occupies less area than the substrate (a reasonable prior for
       a single scan line on a much wider baseplate) and picks whichever
       binary label is the minority class. Override with invert=True/False
       if this guess is wrong for your images.
    4. Morphological closing (kernel `close_kernel`) to bridge small gaps
       in the track region, then opening (kernel `open_kernel`) to remove
       speckle noise. These use separate kernel sizes deliberately:
       closing can be larger since bridging gaps is safe, but opening
       must stay SMALLER than your narrowest expected track width in
       pixels, or it will erode thin sections of a genuine track down to
       nothing and fragment it. If your track narrows to e.g. 15px
       anywhere, open_kernel must be well under that (the default of 5
       is a conservative starting point; increase it only if you know
       your track never gets that thin).
    5. Optionally keep only the largest connected component, since the
       track should be a single contiguous region running the length of
       the panorama; this discards speckle noise misclassified elsewhere.
    6. Optionally fill interior holes in the retained component.

    Returns
    -------
    MaskResult
    """
    gray = _to_gray(panorama)
    if blur_ksize:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            adaptive_block_size, adaptive_C,
        )
    else:
        raise ValueError("method must be 'otsu' or 'adaptive'")

    mask = binary > 0

    if invert == "auto":
        # assume the track is the minority class (smaller area than substrate)
        frac_true = mask.mean()
        did_invert = frac_true > 0.5
        if did_invert:
            mask = ~mask
    else:
        did_invert = bool(invert)
        if did_invert:
            mask = ~mask

    mask_u8 = (mask.astype(np.uint8)) * 255
    if close_kernel and close_kernel > 1:
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_k)
    if open_kernel and open_kernel > 1:
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_k)
    mask = mask_u8 > 0

    coverage_frac = float(mask.mean())
    largest_component_frac = 1.0

    if keep_largest_component and mask.any():
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if n_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
            largest_label = 1 + int(np.argmax(areas))
            new_mask = labels == largest_label
            largest_component_frac = float(new_mask.sum()) / max(mask.sum(), 1)
            mask = new_mask

    if fill_holes and mask.any():
        mask_u8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask_u8)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled > 0

    return MaskResult(
        track_id=None,  # filled in by mask_all_tracks
        mask=mask,
        method=method,
        inverted=did_invert,
        coverage_frac=coverage_frac,
        largest_component_frac=largest_component_frac,
    )


def mask_all_tracks(stitch_results, save_dir=None, **mask_kwargs):
    """
    Run mask_panorama on every track's stitched SEM panorama.

    Parameters
    ----------
    stitch_results : dict {track_id: StitchResult}, as returned by
                      sem_stitching.stitch_all_tracks
    save_dir        : if given, saves each mask as a PNG
                      (mask_track_<id>.png, 255=track / 0=background) and
                      an overlay visualization (mask_overlay_track_<id>.png)
    **mask_kwargs   : passed through to mask_panorama

    Returns
    -------
    dict {track_id: MaskResult}
    """
    masks = {}
    for tid, result in stitch_results.items():
        m = mask_panorama(result.panorama, **mask_kwargs)
        m.track_id = tid
        masks[tid] = m
        print(f"Track {tid}: method={m.method}, inverted={m.inverted}, "
              f"coverage={m.coverage_frac*100:.1f}%, "
              f"largest_component_frac={m.largest_component_frac*100:.1f}%")

        if save_dir is not None:
            mask_path = Path(save_dir) / f"mask_track_{tid}.png"
            cv2.imwrite(str(mask_path), (m.mask.astype(np.uint8)) * 255)

            overlay = _make_overlay(result.panorama, m.mask)
            overlay_path = Path(save_dir) / f"mask_overlay_track_{tid}.png"
            cv2.imwrite(str(overlay_path), overlay)

    return masks


def _make_overlay(panorama, mask, color=(0, 0, 255), alpha=0.4):
    """BGR image with the track region tinted `color` (default red) at `alpha` opacity."""
    base = panorama if panorama.ndim == 3 else cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[mask] = (
        (1 - alpha) * base[mask].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def plot_mask_overlay(stitch_result, mask_result, ax=None):
    """Plot the panorama with the detected track mask tinted red, x-axis in physical mm."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.figure

    overlay = _make_overlay(stitch_result.panorama, mask_result.mask)
    img_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb, extent=[stitch_result.x_start_mm,
                                stitch_result.x_start_mm + stitch_result.width_mm,
                                stitch_result.height_mm, 0])
    ax.set_title(f"Track {stitch_result.track_id}: SEM mask overlay "
                 f"(coverage={mask_result.coverage_frac*100:.1f}%)")
    ax.set_xlabel("actual x (mm)")
    ax.set_ylabel("y (mm, approx.)")
    plt.tight_layout()
    return fig, ax


def mask_width_profile(mask_result, stitch_result, min_valid_frac=0.3):
    """
    Derive a local width profile W(x) directly from the SEM mask, in the
    same physical x coordinate system as the Bruker-derived width profile
    (height_map_descriptors.local_width_profile), so the two can be
    plotted or compared directly.

    For each column of the mask, width = (last True row - first True row)
    of the track region in that column, in mm. Columns with no track
    pixels are NaN.

    Returns
    -------
    dict with keys: x (mm), width_mm, coverage_frac_per_column
    """
    mask = mask_result.mask
    h, w = mask.shape
    px_to_mm_x = stitch_result.width_mm / w
    px_to_mm_y = stitch_result.height_mm / h

    x_mm = stitch_result.x_start_mm + (np.arange(w) + 0.5) * px_to_mm_x
    width_mm = np.full(w, np.nan)
    coverage = mask.mean(axis=0)

    for j in range(w):
        col = mask[:, j]
        if not col.any():
            continue
        rows = np.where(col)[0]
        width_mm[j] = (rows[-1] - rows[0]) * px_to_mm_y

    return {"x": x_mm, "width_mm": width_mm, "coverage_frac_per_column": coverage}
