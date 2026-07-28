"""
Leave-one-track-out cross validation across Track_8, Track_10, and
Track_14, rotating which one is held out. This is deliberately separate
from Track_21 evaluation, since the feature coverage check showed Track_21
sits well outside the training distribution on several melt-pool features,
likely a different process condition entirely. That makes Track_21 a
genuine extrapolation test, useful for robustness but not a fair read on
how well the model interpolates within a covered regime. This gives you
that fairer read, training on two known tracks and testing on the third,
three times over.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_extraction import build_feature_dataset
from gpr_width_model import fit_width_gpr, evaluate_width_model, calibration_curve, check_feature_coverage


def run_leave_one_track_out_cv(
    track_ids,
    n_per_track=100,
    SEM_DIR=None,
    SEM_TILE_WIDTH_MM=None,
    ard=True,
    feature_columns=None,
    seed=0,
):
    """
    Builds one feature dataset per track (so each track's data is drawn
    once, not rebuilt per fold), then for each track in turn, trains on
    the other two and evaluates on the held-out one.

    n_per_track: thermal frames are natively spaced 0.2mm apart (get_
    thermal_frame interpolates between them for anything finer), and SEM
    tiles only change identity every ~6.41mm, so sampling much denser than
    0.2mm apart doesn't add much genuinely new information, nearby x
    locations end up with nearly identical features either way. Something
    around 300 per track, given roughly 80mm of common range, is close to
    where most of the real available signal has been captured without
    spending a lot of extra compute on near-duplicate samples. 100 was a
    reasonable starting point but leaves real data on the table.

    Runs check_feature_coverage for every fold automatically, printed
    right before that fold's model results, so a fold with unusually bad
    metrics can be immediately checked against whether the held-out track
    actually sits inside the training feature range, rather than that
    check being a separate manual step you only think to run after the
    fact.

    Returns (summary_df, fold_results, per_track_df). fold_results is a
    dict keyed by the held-out track_id, holding the fitted gp, scaler,
    train_df, test_df, the coverage report, predicted mean/std, and the
    evaluate_width_model metrics. per_track_df is each track's feature
    table on its own, in case you want to build a different train/test
    split than the standard leave-one-out rotation.
    """
    per_track_df = {
        tid: build_feature_dataset(
            track_ids=[tid], n_per_track=n_per_track,
            SEM_DIR=SEM_DIR, SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM, seed=seed,
        )
        for tid in track_ids
    }

    summary_rows = []
    fold_results = {}

    for held_out in track_ids:
        train_tracks = [t for t in track_ids if t != held_out]
        train_df = pd.concat([per_track_df[t] for t in train_tracks], ignore_index=True)
        test_df = per_track_df[held_out]

        print(f"\n=== held out: {held_out}  (trained on {train_tracks}) ===")
        coverage = check_feature_coverage(train_df, test_df)
        worst = coverage.iloc[0]
        print(f"feature coverage, worst offender: {worst['feature']} "
              f"({worst['test_frac_outside_train_range']*100:.0f}% of {held_out} outside "
              f"the {train_tracks} training range)")
        print(coverage.to_string(index=False))

        gp, scaler = fit_width_gpr(train_df, ard=ard, feature_columns=feature_columns)
        print("fitted kernel:", gp.kernel_)
        results = evaluate_width_model(gp, scaler, test_df)

        summary_rows.append({
            "held_out_track": held_out,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mae": results["mae"],
            "nll": results["nll"],
            "worst_shifted_feature": worst["feature"],
            "worst_shifted_feature_frac_outside": worst["test_frac_outside_train_range"],
        })
        fold_results[held_out] = {
            "gp": gp, "scaler": scaler, "train_df": train_df, "test_df": test_df,
            "coverage": coverage,
            "pred_mean": results["pred_mean"], "pred_std": results["pred_std"],
            "calibration": results["calibration"],
        }

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, fold_results, per_track_df


def plot_cv_summary(summary_df, fold_results):
    """Bar chart of MAE and NLL per fold, plus overlaid calibration curves
    so you can see whether calibration is consistent across tracks or
    whether one track behaves very differently from the others."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(summary_df["held_out_track"], summary_df["mae"])
    axes[0].set_ylabel("MAE (mm)")
    axes[0].set_title("width MAE by held-out track")

    axes[1].bar(summary_df["held_out_track"], summary_df["nll"])
    axes[1].set_ylabel("NLL")
    axes[1].set_title("width NLL by held-out track")

    for held_out, res in fold_results.items():
        calib = res["calibration"]
        axes[2].plot(calib["nominal_coverage"], calib["empirical_coverage"], "o-", label=held_out)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    axes[2].set_xlabel("nominal coverage")
    axes[2].set_ylabel("empirical coverage")
    axes[2].set_title("calibration by fold")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    summary_df, fold_results, per_track_df = run_leave_one_track_out_cv(
        track_ids=["Track_8", "Track_10", "Track_14"],
        n_per_track=300,  # bumped up from 100, see run_leave_one_track_out_cv's docstring
        SEM_DIR=SEM_DIR,
        SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM,
        ard=True,
    )
    print("\n=== cross-validation summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nmean MAE across folds: {summary_df['mae'].mean():.4f} mm")
    print(f"mean NLL across folds: {summary_df['nll'].mean():.4f}")

    plot_cv_summary(summary_df, fold_results)