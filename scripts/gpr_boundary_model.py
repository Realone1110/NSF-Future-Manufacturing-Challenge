"""
Boundary model for y_left_mm and y_right_mm, following the same validated
recipe as the width model: isotropic Matern kernel, full feature set,
known per-sample noise.

Honest limitation worth stating up front: this fits y_left and y_right as
two independent GPs, not a true joint/coregionalized model. A proper joint
fit (e.g. an intrinsic coregionalization kernel) would model the
correlation between the two boundaries directly and derive width's
uncertainty from their true joint covariance. sklearn's GaussianProcessRegressor
doesn't support that directly, it would need GPy or GPflow. Two independent
GPs is a reasonable and simple starting point, but it means the derived
width uncertainty below (Var(right) + Var(left), assuming independence)
is an approximation, and given the two boundaries almost certainly move
together to some degree (same melt pool, same substrate), this likely
overestimates the true width uncertainty rather than underestimating it,
the safer direction for this simplification to err in.

There's also no separate per-boundary noise estimate available yet,
width_std_mm is used as a shared proxy noise for both y_left_mm and
y_right_mm. If the bundle exposes something like y_left_std_mm and
y_right_std_mm directly, swap those in via the noise_col argument for a
more accurate fit.
"""

import numpy as np
import pandas as pd

from gpr_width_model import fit_width_gpr, predict_width, gaussian_nll, calibration_curve

BOUNDARY_TARGETS = ("y_left_mm", "y_right_mm")


def fit_boundary_gpr(train_df, targets=BOUNDARY_TARGETS, noise_col="width_std_mm",
                      feature_columns=None, ard=False, random_state=0):
    """
    Fits one GP per boundary target, reusing fit_width_gpr directly since
    it already takes target_col as a parameter, nothing boundary-specific
    needed in the fitting itself.

    Returns a dict: {target_name: (gp, scaler)}.
    """
    models = {}
    for target in targets:
        gp, scaler = fit_width_gpr(
            train_df, target_col=target, noise_col=noise_col,
            feature_columns=feature_columns, ard=ard, random_state=random_state,
        )
        models[target] = (gp, scaler)
    return models


def predict_boundaries(models, df, targets=BOUNDARY_TARGETS):
    """Returns a dict {target_name: (mean, std)} for every row in df."""
    predictions = {}
    for target in targets:
        gp, scaler = models[target]
        mean, std = predict_width(gp, scaler, df)
        predictions[target] = (mean, std)
    return predictions


def derive_width_from_boundaries(predictions, left_key="y_left_mm", right_key="y_right_mm"):
    """
    width = right - left. Uncertainty combined assuming independence
    (see module docstring for why this is an approximation, and which
    direction it's likely biased).
    """
    left_mean, left_std = predictions[left_key]
    right_mean, right_std = predictions[right_key]
    width_mean = right_mean - left_mean
    width_std = np.sqrt(left_std ** 2 + right_std ** 2)
    return width_mean, width_std


def evaluate_boundary_model(models, test_df, targets=BOUNDARY_TARGETS):
    """
    Per-target MAE/NLL/calibration, the same metrics used for the width
    model, plus the derived width comparison against the bundle's own
    width_mean_mm, so you can see directly how the joint boundary model's
    implied width compares to fitting width directly.
    """
    predictions = predict_boundaries(models, test_df, targets)
    per_target = {}
    for target in targets:
        mean, std = predictions[target]
        y_true = test_df[target].values
        mae = float(np.mean(np.abs(y_true - mean)))
        nll = gaussian_nll(y_true, mean, std)
        calib = calibration_curve(y_true, mean, std)
        per_target[target] = {"mae": mae, "nll": nll, "calibration": calib,
                               "pred_mean": mean, "pred_std": std}
        print(f"\n{target}")
        print(f"  MAE: {mae:.4f} mm")
        print(f"  NLL: {nll:.4f}")
        print("  calibration:")
        print("  " + calib.to_string(index=False).replace("\n", "\n  "))

    width_mean, width_std = derive_width_from_boundaries(predictions)
    width_true = test_df["width_mean_mm"].values
    width_mae = float(np.mean(np.abs(width_true - width_mean)))
    width_nll = gaussian_nll(width_true, width_mean, width_std)
    width_calib = calibration_curve(width_true, width_mean, width_std)

    print("\nderived width (right - left)")
    print(f"  MAE: {width_mae:.4f} mm")
    print(f"  NLL: {width_nll:.4f}")
    print("  calibration:")
    print("  " + width_calib.to_string(index=False).replace("\n", "\n  "))

    return {
        "per_target": per_target,
        "derived_width": {
            "mae": width_mae, "nll": width_nll, "calibration": width_calib,
            "pred_mean": width_mean, "pred_std": width_std,
        },
    }
