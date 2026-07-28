"""
Trains the boundary GPR (y_left_mm, y_right_mm) on the pooled training
tracks, validates on Track_21, and saves individual publication-quality
figures, predictions, a text summary, timing, and the fitted models into
./ML_results/, mirroring generate_width_results.py.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import build_feature_dataset
from gpr_width_model import check_feature_coverage, FEATURE_COLUMNS
from gpr_boundary_model import (
    fit_boundary_gpr, predict_boundaries, derive_width_from_boundaries,
    evaluate_boundary_model, BOUNDARY_TARGETS,
)
from report_style import apply_manuscript_style, save_fig, save_model, Timer

RESULTS_DIR = "ML_results"

apply_manuscript_style()


def save_boundary_result_figures(test_df, models, eval_results, coverage, results_dir, stem):
    x = test_df["x_mm"].values
    order = np.argsort(x)
    predictions = predict_boundaries(models, test_df)
    left_mean, left_std = predictions["y_left_mm"]
    right_mean, right_std = predictions["y_right_mm"]
    width_mean, width_std = derive_width_from_boundaries(predictions)
    paths = {}

    # --- boundaries vs x ---
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.fill_between(x[order], (left_mean - 1.96 * left_std)[order], (left_mean + 1.96 * left_std)[order],
                     color="tab:blue", alpha=0.2)
    ax.plot(x[order], left_mean[order], color="tab:blue", lw=1.8, label="predicted y_left")
    ax.scatter(x, test_df["y_left_mm"].values, color="navy", s=16, zorder=5, label="true y_left")
    ax.fill_between(x[order], (right_mean - 1.96 * right_std)[order], (right_mean + 1.96 * right_std)[order],
                     color="tab:orange", alpha=0.2)
    ax.plot(x[order], right_mean[order], color="tab:orange", lw=1.8, label="predicted y_right")
    ax.scatter(x, test_df["y_right_mm"].values, color="darkred", s=16, zorder=5, label="true y_right")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("cross-track position (mm)")
    ax.set_title("Predicted vs true track boundaries along Track_21")
    ax.legend(ncol=2, framealpha=0.9)
    left_res, right_res = eval_results["per_target"]["y_left_mm"], eval_results["per_target"]["y_right_mm"]
    width_res = eval_results["derived_width"]
    summary_text = (f"y_left  MAE={left_res['mae']:.4f}mm  NLL={left_res['nll']:.4f}\n"
                     f"y_right MAE={right_res['mae']:.4f}mm  NLL={right_res['nll']:.4f}\n"
                     f"derived width MAE={width_res['mae']:.4f}mm  NLL={width_res['nll']:.4f}")
    ax.text(0.01, 0.02, summary_text, transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
            family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
    paths["boundaries"] = save_fig(fig, results_dir, stem, "boundaries")
    plt.show()

    # --- derived width vs x ---
    fig, ax = plt.subplots(figsize=(12, 6))
    width_true = test_df["width_mean_mm"].values
    ax.fill_between(x[order], (width_mean - 1.96 * width_std)[order], (width_mean + 1.96 * width_std)[order],
                     color="tab:green", alpha=0.2, label="derived width 95% interval")
    ax.plot(x[order], width_mean[order], color="tab:green", lw=1.8, label="derived width mean")
    ax.scatter(x, width_true, color="black", s=16, zorder=5, label="true width")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("width (mm)")
    ax.set_title("Width derived from the joint boundary model (right - left)")
    ax.legend()
    paths["derived_width"] = save_fig(fig, results_dir, stem, "derived_width")
    plt.show()

    # --- boundary calibration ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for target, color in zip(BOUNDARY_TARGETS, ["tab:blue", "tab:orange"]):
        calib = eval_results["per_target"][target]["calibration"]
        ax.plot(calib["nominal_coverage"], calib["empirical_coverage"], "o-", color=color, lw=2, ms=8, label=target)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Boundary calibration")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    paths["boundary_calibration"] = save_fig(fig, results_dir, stem, "boundary_calibration")
    plt.show()

    # --- derived width calibration ---
    fig, ax = plt.subplots(figsize=(6, 6))
    wcal = width_res["calibration"]
    ax.plot(wcal["nominal_coverage"], wcal["empirical_coverage"], "o-", color="tab:green", lw=2, ms=8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Derived width calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    paths["derived_width_calibration"] = save_fig(fig, results_dir, stem, "derived_width_calibration")
    plt.show()

    # --- feature coverage ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cov_sorted = coverage.sort_values("test_frac_outside_train_range", ascending=True).tail(8)
    ax.barh(cov_sorted["feature"], cov_sorted["test_frac_outside_train_range"] * 100, color="tab:orange")
    ax.set_xlabel("% of Track_21 samples outside training feature range")
    ax.set_title("Worst feature coverage")
    ax.set_xlim(0, 100)
    paths["coverage"] = save_fig(fig, results_dir, stem, "coverage")
    plt.show()

    return paths


def run_and_save_boundary_results(
    train_track_ids=("Track_8", "Track_10", "Track_14"),
    test_track_ids=("Track_21",),
    n_per_track=300,
    SEM_DIR=None,
    SEM_TILE_WIDTH_MM=None,
    ard=False,
    feature_columns=None,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"Boundary_result_{timestamp}"
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
        models = fit_boundary_gpr(train_df, feature_columns=feature_columns, ard=ard)
    for target, (gp, _) in models.items():
        print(f"{target} fitted kernel:", gp.kernel_)

    with timer.stage("evaluation"):
        eval_results = evaluate_boundary_model(models, test_df)
        coverage = check_feature_coverage(train_df, test_df, feature_columns or FEATURE_COLUMNS)

    with timer.stage("figure generation"):
        fig_paths = save_boundary_result_figures(test_df, models, eval_results, coverage, RESULTS_DIR, stem)

    with timer.stage("saving model/predictions"):
        model_path = save_model(models, stem, "model")

        predictions = predict_boundaries(models, test_df)
        width_mean, width_std = derive_width_from_boundaries(predictions)
        predictions_df = test_df[["track_id", "x_mm", "y_left_mm", "y_right_mm", "width_mean_mm"]].copy()
        predictions_df["predicted_y_left_mean"] = predictions["y_left_mm"][0]
        predictions_df["predicted_y_left_std"] = predictions["y_left_mm"][1]
        predictions_df["predicted_y_right_mean"] = predictions["y_right_mm"][0]
        predictions_df["predicted_y_right_std"] = predictions["y_right_mm"][1]
        predictions_df["derived_width_mean"] = width_mean
        predictions_df["derived_width_std"] = width_std
        csv_path = os.path.join(RESULTS_DIR, f"{stem}_predictions.csv")
        predictions_df.to_csv(csv_path, index=False)

        summary_path = os.path.join(RESULTS_DIR, f"{stem}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Boundary GPR results, generated {timestamp}\n")
            f.write(f"Trained on: {list(train_track_ids)}  (n={len(train_df)})\n")
            f.write(f"Validated on: {list(test_track_ids)}  (n={len(test_df)})\n\n")
            for target, (gp, _) in models.items():
                f.write(f"{target} kernel: {gp.kernel_}\n")
            f.write("\n")
            for target in BOUNDARY_TARGETS:
                r = eval_results["per_target"][target]
                f.write(f"{target}: MAE={r['mae']:.4f}mm  NLL={r['nll']:.4f}\n")
            w = eval_results["derived_width"]
            f.write(f"derived width: MAE={w['mae']:.4f}mm  NLL={w['nll']:.4f}\n\n")
            f.write("Feature coverage (Track_21 vs training range):\n")
            f.write(coverage.to_string(index=False) + "\n\n")
            f.write("Timing:\n")
            f.write(timer.summary() + "\n")

    print("\ntiming:\n" + timer.summary())
    print(f"\nsaved model:       {model_path}")
    print(f"saved figures:     {list(fig_paths.values())}")
    print(f"saved predictions: {csv_path}")
    print(f"saved summary:     {summary_path}")

    return models, eval_results, coverage, train_df, test_df


if __name__ == "__main__":
    run_and_save_boundary_results(
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        ard=False,
    )
