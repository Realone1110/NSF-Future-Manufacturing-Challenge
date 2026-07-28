import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from nsf_fmrg_data import get_sem_tile_paths, load_sem_tile, largest_true_run

# --------------------------------------------------------------------------
# core detection primitives
# --------------------------------------------------------------------------

def local_variance(img, window=15):
    """Moving-window variance, used as a texture/roughness measure."""
    img = img.astype(np.float64)
    mean = ndimage.uniform_filter(img, size=window)
    mean_sq = ndimage.uniform_filter(img ** 2, size=window)
    var = mean_sq - mean ** 2
    var[var < 0] = 0
    return var


def otsu_threshold(values):
    """Simple Otsu threshold, no skimage dependency needed."""
    values = values[np.isfinite(values)]
    hist, bin_edges = np.histogram(values, bins=256)
    hist = hist.astype(float)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    w1_safe = np.where(w1 == 0, 1, w1)
    w2_safe = np.where(w2 == 0, 1, w2)
    m1 = np.cumsum(hist * bin_mids) / w1_safe
    m2 = (np.cumsum((hist * bin_mids)[::-1])[::-1]) / w2_safe
    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return bin_mids[np.argmax(between)]


def detect_track_band(img, window=15, min_run_frac=0.10, edge_margin_frac=0.02,
                       morph_size=5, smooth_cols=25, tol_frac=0.05,
                       min_inlier_frac=0.25):
    """
    Finds the smooth track band in a single SEM tile.

    Per column, the largest contiguous low-variance run gives a candidate
    y_top/y_bottom. A handful of columns nearly always land on a stray flat
    patch inside the rough region, so instead of a straight min/max over
    every column, this smooths the column-wise boundaries with a median
    filter and only keeps columns that agree with that trend within
    tol_frac of the tile height. The final rectangle comes from the min/max
    over those inlier columns only. If too few columns survive that check,
    it falls back to a 5th/95th percentile range over all valid columns
    rather than failing outright.

    Returns a dict with the per-column boundaries, an inlier mask, the
    aggregated rectangle (y_min, y_max), and the intermediate variance map,
    or None if no band could be found at all.
    """
    h, w = img.shape
    var = local_variance(img, window=window)
    thresh = otsu_threshold(var)
    smooth_mask = var < thresh

    if morph_size:
        struct = np.ones((morph_size, morph_size))
        smooth_mask = ndimage.binary_opening(smooth_mask, structure=struct)
        smooth_mask = ndimage.binary_closing(smooth_mask, structure=struct)

    min_run = int(min_run_frac * h)
    edge_margin = max(1, int(edge_margin_frac * h))

    y_top = np.full(w, np.nan)
    y_bottom = np.full(w, np.nan)

    for x in range(w):
        start, stop = largest_true_run(smooth_mask[:, x])
        if start is None or (stop - start) < min_run:
            continue
        # a genuine track band shouldn't touch the very top/bottom of the tile
        if start <= edge_margin or stop >= h - edge_margin:
            continue
        y_top[x] = start
        y_bottom[x] = stop

    valid = ~np.isnan(y_top)
    if valid.sum() == 0:
        return None

    fill_top = np.where(valid, y_top, np.nanmedian(y_top[valid]))
    fill_bot = np.where(valid, y_bottom, np.nanmedian(y_bottom[valid]))
    top_trend = ndimage.median_filter(fill_top, size=smooth_cols)
    bot_trend = ndimage.median_filter(fill_bot, size=smooth_cols)

    tol_px = max(3, int(tol_frac * h))
    inlier = valid & (np.abs(y_top - top_trend) <= tol_px) & (np.abs(y_bottom - bot_trend) <= tol_px)

    if inlier.sum() < min_inlier_frac * w:
        # not enough columns agree with the trend, fall back to a trimmed
        # range so a few noisy columns can't still blow the box wide open
        y_min = int(np.percentile(y_top[valid], 5))
        y_max = int(np.percentile(y_bottom[valid], 95))
    else:
        y_min = int(np.min(y_top[inlier]))
        y_max = int(np.max(y_bottom[inlier]))

    return {
        'y_top': y_top, 'y_bottom': y_bottom, 'valid': valid, 'inlier': inlier,
        'y_min': y_min, 'y_max': y_max,
        'variance': var, 'threshold': float(thresh),
        'coverage': float(valid.mean()), 'inlier_frac': float(inlier.mean()),
    }


def mask_track_tile(img, band):
    """Returns (masked_full, cropped) using the aggregated rectangle.
    masked_full keeps the original tile shape with everything outside the
    band set to NaN, cropped is just the band rows."""
    masked_full = np.full(img.shape, np.nan, dtype=np.float64)
    masked_full[band['y_min']:band['y_max'], :] = img[band['y_min']:band['y_max'], :]
    cropped = img[band['y_min']:band['y_max'], :]
    return masked_full, cropped


# --------------------------------------------------------------------------
# main loop, mirrors the tiling/plotting pattern already used for sem_summary
# --------------------------------------------------------------------------

def run_track_masking(TRACK_IDS, SEM_DIR, SEM_TILE_WIDTH_MM, FIGURES_DIR, METADATA_DIR,
                       n_show=6):
    import json

    mask_summary = {}
    for track_id in TRACK_IDS:
        tile_paths = get_sem_tile_paths(SEM_DIR, track_id)
        if not tile_paths:
            continue
        n = min(n_show, len(tile_paths))
        fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.2), constrained_layout=True)
        track_records = []
        for col, p in enumerate(tile_paths[:n]):
            img = load_sem_tile(p)
            h, w = img.shape
            sem_height_mm = SEM_TILE_WIDTH_MM * h / w
            band = detect_track_band(img)

            ax_top = axes[0, col]
            ax_bot = axes[1, col]
            extent = [0, SEM_TILE_WIDTH_MM, sem_height_mm, 0]
            ax_top.imshow(img, cmap='gray', extent=extent)
            ax_top.set_title(p.stem, fontsize=8)

            if band is None:
                track_records.append({'file': p.name, 'detected': False})
                ax_bot.imshow(img, cmap='gray', extent=extent)
                ax_bot.set_title('no band found', fontsize=8)
                continue

            px_to_mm = sem_height_mm / h
            y_min_mm = band['y_min'] * px_to_mm
            y_max_mm = band['y_max'] * px_to_mm
            x_mm = np.arange(w) * (SEM_TILE_WIDTH_MM / w)

            # green rectangle over the original tile, outline plus light fill
            ax_top.axhspan(y_min_mm, y_max_mm, facecolor='lime', alpha=0.30,
                            edgecolor='lime', linewidth=1.5)

            valid = band['valid']
            inlier = band['inlier']
            outlier = valid & ~inlier
            # points that got thrown out by the robustness check, shown faint
            ax_top.plot(x_mm[outlier], band['y_top'][outlier] * px_to_mm, '.', ms=1.5, color='0.6')
            ax_top.plot(x_mm[outlier], band['y_bottom'][outlier] * px_to_mm, '.', ms=1.5, color='0.6')
            # points that actually set the rectangle
            ax_top.plot(x_mm[inlier], band['y_top'][inlier] * px_to_mm, '.', ms=1.5, color='cyan')
            ax_top.plot(x_mm[inlier], band['y_bottom'][inlier] * px_to_mm, '.', ms=1.5, color='yellow')

            # masked view, full tile size, everything outside the band shown
            # as solid green so the mask is obvious even if the band is wide
            masked_full, _ = mask_track_tile(img, band)
            rgb = np.stack([img, img, img], axis=-1).astype(np.float64)
            rgb -= rgb.min()
            if rgb.max() > 0:
                rgb /= rgb.max()
            outside = np.isnan(masked_full)
            rgb[outside] = [0.0, 1.0, 0.0]
            ax_bot.imshow(rgb, extent=extent)
            ax_bot.set_title('green = masked out', fontsize=8)

            track_records.append({
                'file': p.name, 'detected': True,
                'y_min_px': band['y_min'], 'y_max_px': band['y_max'],
                'y_min_mm': float(y_min_mm), 'y_max_mm': float(y_max_mm),
                'coverage_frac': band['coverage'], 'inlier_frac': band['inlier_frac'],
            })

        fig.suptitle(f'Track {track_id}: detected and masked track band', fontsize=12)
        fig.savefig(FIGURES_DIR / f'sem_track_mask_{track_id}.png', dpi=400, bbox_inches='tight')
        fig.savefig(FIGURES_DIR / f'sem_track_mask_{track_id}.pdf', bbox_inches='tight')
        plt.show()

        mask_summary[str(track_id)] = track_records

    with open(METADATA_DIR / 'sem_track_mask_summary.json', 'w') as f:
        json.dump(mask_summary, f, indent=2)

    return mask_summary