"""
Trains the final width GPR on all three training tracks pooled together,
validates on Track_21, and saves individual publication-quality figures
(one file per plot, not a combined multi-panel figure) plus predictions,
a text summary, timing, and the fitted model itself into ./ML_results/.

Model choice, locked in after the leave-one-track-out investigation:
  - isotropic Matern kernel (ard=False), not per-feature ARD. ARD showed
    real instability at larger sample sizes (kernel amplitude swinging
    from 1.23**2 to 6.31**2 between n=100 and n=300 on the same data),
    while the isotropic kernel stayed stable across both sample sizes.
  - full FEATURE_COLUMNS (including cooling_rate_proxy and
    linear_energy_density, excluding thermal_gradient_proxy per the
    ablation study).
  - known per-sample noise from width_std_mm via alpha, not a free
    WhiteKernel.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import build_feature_dataset
from gpr_width_model import (
    fit_width_gpr, evaluate_width_model, check_feature_coverage, FEATURE_COLUMNS,
)
from report_style import apply_manuscript_style, save_fig, save_model, Timer

RESULTS_DIR = "ML_results"

apply_manuscript_style()


def save_width_result_figures(train_df, test_df, gp, results, coverage, results_dir, stem):
    """Saves five separate figures instead of one combined panel: the main
    width-vs-x plot, calibration, parity, residuals, and feature coverage.
    Returns a dict of {panel_name: path}."""
    mean, std = results["pred_mean"], results["pred_std"]
    y_true = test_df["width_mean_mm"].values
    x = test_df["x_mm"].values
    order = np.argsort(x)
    paths = {}

    # --- main: width vs x with uncertainty band ---
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x[order], (mean - 1.96 * std)[order], (mean + 1.96 * std)[order],
                     color="tab:blue", alpha=0.2, label="predicted 95% interval")
    ax.plot(x[order], mean[order], color="tab:blue", lw=1.8, label="predicted mean")
    ax.scatter(x, y_true, color="black", s=20, zorder=5, label="true width")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("local width (mm)")
    ax.set_title("Predicted vs true local width along Track_21")
    ax.legend(loc="upper right", framealpha=0.9)
    mae, nll = results["mae"], results["nll"]
    summary_text = (f"MAE = {mae:.4f} mm\nNLL = {nll:.4f}\n"
                     f"n_train = {len(train_df)}   n_test = {len(test_df)}\n"
                     f"kernel: {gp.kernel_}")
    ax.text(0.01, 0.02, summary_text, transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
            family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
    paths["main"] = save_fig(fig, results_dir, stem, "main")
    plt.show()

    # --- calibration ---
    fig, ax = plt.subplots(figsize=(6, 6))
    calib = results["calibration"]
    ax.plot(calib["nominal_coverage"], calib["empirical_coverage"], "o-", color="tab:blue", lw=2, ms=8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    paths["calibration"] = save_fig(fig, results_dir, stem, "calibration")
    plt.show()

    # --- parity plot ---
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.errorbar(y_true, mean, yerr=1.96 * std, fmt="o", ms=4, alpha=0.5,
                ecolor="lightgray", capsize=0, color="tab:blue")
    lims = [min(y_true.min(), mean.min()), max(y_true.max(), mean.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, label="perfect prediction")
    ax.set_xlabel("true width (mm)")
    ax.set_ylabel("predicted width (mm)")
    ax.set_title("Parity plot")
    ax.legend()
    paths["parity"] = save_fig(fig, results_dir, stem, "parity")
    plt.show()

    # --- residuals vs x ---
    fig, ax = plt.subplots(figsize=(11, 5))
    residuals = mean - y_true
    ax.scatter(x, residuals, s=16, color="tab:red", alpha=0.6)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("predicted - true (mm)")
    ax.set_title("Residuals along track")
    paths["residuals"] = save_fig(fig, results_dir, stem, "residuals")
    plt.show()

    # --- feature coverage ---
    fig, ax = plt.subplots(figsize=(11, 6))
    cov_sorted = coverage.sort_values("test_frac_outside_train_range", ascending=True)
    bars = ax.barh(cov_sorted["feature"], cov_sorted["test_frac_outside_train_range"] * 100, color="tab:orange")
    ax.set_xlabel("% of Track_21 samples outside training feature range")
    ax.set_title("Feature coverage: Track_21 vs training tracks")
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, cov_sorted["test_frac_outside_train_range"] * 100):
        if val > 1:
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%", va="center", fontsize=10)
    paths["coverage"] = save_fig(fig, results_dir, stem, "coverage")
    plt.show()

    return paths


def run_and_save_width_results(
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
    stem = f"Width_result_{timestamp}"
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
        gp, scaler = fit_width_gpr(train_df, ard=ard, feature_columns=feature_columns)
    print("fitted kernel:", gp.kernel_)

    with timer.stage("evaluation"):
        results = evaluate_width_model(gp, scaler, test_df)
        coverage = check_feature_coverage(train_df, test_df, feature_columns or FEATURE_COLUMNS)

    with timer.stage("figure generation"):
        fig_paths = save_width_result_figures(train_df, test_df, gp, results, coverage, RESULTS_DIR, stem)

    with timer.stage("saving model/predictions"):
        model_path = save_model((gp, scaler), stem, "model")

        predictions_df = test_df[["track_id", "x_mm", "width_mean_mm", "width_std_mm"]].copy()
        predictions_df["predicted_mean_mm"] = results["pred_mean"]
        predictions_df["predicted_std_mm"] = results["pred_std"]
        csv_path = os.path.join(RESULTS_DIR, f"{stem}_predictions.csv")
        predictions_df.to_csv(csv_path, index=False)

        summary_path = os.path.join(RESULTS_DIR, f"{stem}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Width GPR results, generated {timestamp}\n")
            f.write(f"Trained on: {list(train_track_ids)}  (n={len(train_df)})\n")
            f.write(f"Validated on: {list(test_track_ids)}  (n={len(test_df)})\n")
            f.write(f"Kernel: {gp.kernel_}\n\n")
            f.write(f"MAE: {results['mae']:.4f} mm\n")
            f.write(f"NLL: {results['nll']:.4f}\n\n")
            f.write("Calibration:\n")
            f.write(results["calibration"].to_string(index=False) + "\n\n")
            f.write("Feature coverage (Track_21 vs training range):\n")
            f.write(coverage.to_string(index=False) + "\n\n")
            f.write("Timing:\n")
            f.write(timer.summary() + "\n")

    print("\ntiming:\n" + timer.summary())
    print(f"\nsaved model:       {model_path}")
    print(f"saved figures:     {list(fig_paths.values())}")
    print(f"saved predictions: {csv_path}")
    print(f"saved summary:     {summary_path}")

    return gp, scaler, results, coverage, train_df, test_df


if __name__ == "__main__":
    run_and_save_width_results(
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        ard=False,
    )
