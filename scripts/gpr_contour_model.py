"""
Contour-shape descriptor models: centerline_y_mm, peak_depth_mm,
profile_skewness, profile_waviness_mm. Same recipe as the width and
boundary models, isotropic kernel, same finalized feature set.

No per-sample noise estimate exists for these four targets the way
width_std_mm does for width, so noise_col=None here, falling back to a
learned WhiteKernel. Worth naming as a limitation in the report's
uncertainty section, calibration for these four is less trustworthy than
for width/boundary until a real per-sample noise estimate is derived.
"""

import numpy as np

from gpr_width_model import fit_width_gpr, predict_width, gaussian_nll, calibration_curve

CONTOUR_TARGETS = ("centerline_y_mm", "peak_depth_mm", "profile_skewness", "profile_waviness_mm")


def fit_contour_gpr(train_df, targets=CONTOUR_TARGETS, feature_columns=None, ard=False, random_state=0):
    """One GP per contour descriptor, reusing fit_width_gpr directly."""
    models = {}
    for target in targets:
        gp, scaler = fit_width_gpr(
            train_df, target_col=target, noise_col=None,
            feature_columns=feature_columns, ard=ard, random_state=random_state,
        )
        models[target] = (gp, scaler)
    return models


def predict_contours(models, df, targets=CONTOUR_TARGETS):
    return {target: predict_width(*models[target], df) for target in targets}


def evaluate_contour_model(models, test_df, targets=CONTOUR_TARGETS):
    predictions = predict_contours(models, test_df, targets)
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
        print(f"  MAE: {mae:.4f}")
        print(f"  NLL: {nll:.4f}")
    return per_target
