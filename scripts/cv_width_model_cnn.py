"""
Leave-one-track-out CV for the CNN + heteroscedastic head width model,
same rotation over Track_8, Track_10, Track_14 that scripts/cv_width_model.py
already runs for the GPR, so the two summary tables are directly comparable
fold for fold.

One ThermalStackWidthDataset is built per track up front, exactly the way
cv_width_model.py builds per_track_df once and reuses it across folds,
so each track's thermal stacks and SEM crops only get extracted from disk
once no matter how many folds use that track. For each held-out track,
the other two tracks' datasets are concatenated into a training pool, a
val_frac slice is carved out of THAT pool for early stopping (never from
the held-out track, which stays untouched until final evaluation), and
width_mean/width_std are fit on the training pool only, then applied to
both the training pool and the held-out test set, the same
fit-on-train-apply-to-test discipline multimodal_dataset.py already uses
for its own height normalization.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from nsf_fmrg.cnn_dataset import ThermalStackWidthDataset
from nsf_fmrg.cnn_width_model import train_heteroscedastic_width_model, evaluate_cnn_width_model


def run_leave_one_track_out_cnn_cv(
    track_ids, n_per_track=300, SEM_DIR=None, SEM_TILE_WIDTH_MM=None,
    n_frames=5, frame_spacing=1, thermal_size=(64, 64), sem_size=(64, 64),
    val_frac=0.15, seed=0, **train_kwargs,
):
    """
    Returns (summary_df, fold_results, per_track_ds), same shape as
    cv_width_model.run_leave_one_track_out_cv: summary_df has one row per
    held-out track with mae/nll, fold_results holds each fold's trained
    model/history/predictions, per_track_ds is the raw per-track dataset
    dict in case you want a different split than the standard rotation.
    """
    per_track_ds = {
        tid: ThermalStackWidthDataset(
            track_ids=[tid], n_per_track=n_per_track, n_frames=n_frames,
            frame_spacing=frame_spacing, thermal_size=thermal_size, sem_size=sem_size,
            SEM_DIR=SEM_DIR, SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM, seed=seed,
        )
        for tid in track_ids
    }

    summary_rows = []
    fold_results = {}

    for held_out in track_ids:
        train_tracks = [t for t in track_ids if t != held_out]

        # fit normalization on the training tracks only, then share it
        # with both the training pool and the held-out test set
        train_vals = np.concatenate([
            np.array([s["width_mean_mm"] for s in per_track_ds[t]._samples])
            for t in train_tracks
        ])
        width_mean, width_std = float(train_vals.mean()), float(train_vals.std()) or 1.0
        for t in train_tracks + [held_out]:
            per_track_ds[t].width_mean = width_mean
            per_track_ds[t].width_std = width_std

        full_train = ConcatDataset([per_track_ds[t] for t in train_tracks])
        n_val = max(1, int(len(full_train) * val_frac))
        n_train = len(full_train) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_split, val_split = random_split(full_train, [n_train, n_val], generator=generator)
        # random_split gives back a Subset, which has no n_frames of its
        # own; train_heteroscedastic_width_model only reads that
        # attribute off whatever's passed in as train_dataset, so it's
        # patched on directly rather than needing a whole wrapper class.
        train_split.n_frames = n_frames

        print(f"\n=== held out: {held_out}  (trained on {train_tracks}, "
              f"{n_train} train / {n_val} val samples) ===")

        model, history = train_heteroscedastic_width_model(
            train_split, val_split, seed=seed, **train_kwargs,
        )

        test_ds = per_track_ds[held_out]
        results = evaluate_cnn_width_model(model, test_ds)

        summary_rows.append({
            "held_out_track": held_out,
            "n_train": n_train, "n_val": n_val, "n_test": len(test_ds),
            "mae": results["mae"], "nll": results["nll"],
        })
        fold_results[held_out] = {
            "model": model, "history": history,
            "pred_mean": results["pred_mean"], "pred_std": results["pred_std"],
            "calibration": results["calibration"],
        }

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, fold_results, per_track_ds


if __name__ == "__main__":
    summary_df, fold_results, per_track_ds = run_leave_one_track_out_cnn_cv(
        track_ids=["Track_8", "Track_10", "Track_14"],
        n_per_track=300,
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        n_frames=5,
        max_epochs=300,
        warmup_epochs=15,
        patience=25,
    )
    print("\n=== CNN cross-validation summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nmean MAE across folds: {summary_df['mae'].mean():.4f} mm")
    print(f"mean NLL across folds: {summary_df['nll'].mean():.4f}")
