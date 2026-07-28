"""
Contour descriptor training, validation, and individual publication-quality
figures, mirroring the width/boundary pipelines: one file per descriptor,
timing, and saved models.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import build_feature_dataset
from gpr_width_model import check_feature_coverage, FEATURE_COLUMNS
from gpr_contour_model import fit_contour_gpr, evaluate_contour_model, predict_contours, CONTOUR_TARGETS
from report_style import apply_manuscript_style, save_fig, save_model, Timer

RESULTS_DIR = "ML_results"

apply_manuscript_style()


def save_contour_result_figures(test_df, per_target, results_dir, stem):
    x = test_df["x_mm"].values
    order = np.argsort(x)
    paths = {}
    for target in CONTOUR_TARGETS:
        mean, std = per_target[target]["pred_mean"], per_target[target]["pred_std"]
        y_true = test_df[target].values

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(x[order], (mean - 1.96 * std)[order], (mean + 1.96 * std)[order],
                         color="tab:purple", alpha=0.2, label="predicted 95% interval")
        ax.plot(x[order], mean[order], color="tab:purple", lw=1.8, label="predicted mean")
        ax.scatter(x, y_true, color="black", s=18, zorder=5, label="true")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel(target)
        mae, nll = per_target[target]["mae"], per_target[target]["nll"]
        ax.set_title(f"{target}    MAE = {mae:.4f}    NLL = {nll:.4f}")
        ax.legend()
        paths[target] = save_fig(fig, results_dir, stem, target)
        plt.show()
    return paths


def run_and_save_contour_results(
    train_track_ids=("Track_8", "Track_10", "Track_14"),
    test_track_ids=("Track_21",),
    n_per_track=300,
    SEM_DIR=None,
    SEM_TILE_WIDTH_MM=None,
    ard=False,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"Contour_result_{timestamp}"
    timer = Timer()

    with timer.stage("data extraction"):
        train_df = build_feature_dataset(
            track_ids=list(train_track_ids), n_per_track=n_per_track,
            SEM_DIR=SEM_DIR, SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM, seed=0,
        )
        test_df = build_feature_dataset(
            track_ids=list(test_track_ids), n_per_track=n_per_track,
            SEM_DIR=SEM_DIR, SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM, seed=42,
        )

    with timer.stage("model fitting"):
        models = fit_contour_gpr(train_df, ard=ard)

    with timer.stage("evaluation"):
        per_target = evaluate_contour_model(models, test_df)
        coverage = check_feature_coverage(train_df, test_df, FEATURE_COLUMNS)

    with timer.stage("figure generation"):
        fig_paths = save_contour_result_figures(test_df, per_target, RESULTS_DIR, stem)

    with timer.stage("saving model/predictions"):
        model_path = save_model(models, stem, "model")

        predictions_df = test_df[["track_id", "x_mm"] + list(CONTOUR_TARGETS)].copy()
        for target in CONTOUR_TARGETS:
            mean, std = per_target[target]["pred_mean"], per_target[target]["pred_std"]
            predictions_df[f"predicted_{target}_mean"] = mean
            predictions_df[f"predicted_{target}_std"] = std
        csv_path = os.path.join(RESULTS_DIR, f"{stem}_predictions.csv")
        predictions_df.to_csv(csv_path, index=False)

        summary_path = os.path.join(RESULTS_DIR, f"{stem}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Contour descriptor GPR results, generated {timestamp}\n")
            f.write(f"Trained on: {list(train_track_ids)}  (n={len(train_df)})\n")
            f.write(f"Validated on: {list(test_track_ids)}  (n={len(test_df)})\n\n")
            for target in CONTOUR_TARGETS:
                r = per_target[target]
                f.write(f"{target}: MAE={r['mae']:.4f}  NLL={r['nll']:.4f}\n")
            f.write("\nFeature coverage (Track_21 vs training range):\n")
            f.write(coverage.to_string(index=False) + "\n\n")
            f.write("Timing:\n")
            f.write(timer.summary() + "\n")

    print("\ntiming:\n" + timer.summary())
    print(f"\nsaved model:       {model_path}")
    print(f"saved figures:     {list(fig_paths.values())}")
    print(f"saved predictions: {csv_path}")
    print(f"saved summary:     {summary_path}")

    return models, per_target, coverage, train_df, test_df


if __name__ == "__main__":
    run_and_save_contour_results(
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
    )
