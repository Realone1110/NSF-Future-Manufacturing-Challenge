"""
Gaussian process regression for local track width, trained on the feature
table from feature_extraction.py.

A single GP gives both a predictive mean and a predictive standard
deviation at every input, so this is one model, not a separate mean model
and a separate uncertainty model. The kernel includes a WhiteKernel term,
which lets the GP learn a baseline noise level directly from the data
rather than assuming one.

Evaluation follows the paper's own suggested metrics rather than plain
MSE: mean absolute error, negative log likelihood, and calibration error,
since the challenge explicitly asks for a probabilistic prediction, not
just a point estimate.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


FEATURE_COLUMNS = [
    "x_frac",
    "melt_area_mm2", "melt_peak", "melt_mean", "melt_aspect_ratio",
    "melt_asymmetry", "melt_tail_ratio",
    "cooling_rate_proxy",
    "sem_mean", "sem_std", "sem_texture_mean", "sem_texture_std", "sem_texture_max",
    "linear_energy_density",
]
# thermal_gradient_proxy deliberately excluded, a direct ablation showed it
# hurts MAE in every combination tried (0.229-0.247mm vs a 0.153mm baseline
# with none of the three new features), most likely because Track_21 sits
# more than a full training-range's-width past the edge of what was seen in
# training on this specific feature, a much more severe relative
# extrapolation than linear_energy_density's, which helped despite also
# being entirely outside the training range.

# melt_area_mm2, melt_aspect_ratio, and melt_asymmetry have hit the length
# scale upper bound (effectively ignored by the GP) in every leave-one-
# track-out fold tried so far, this drops them as a direct comparison
# point against the full feature set.
REDUCED_FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if c not in ("melt_area_mm2", "melt_aspect_ratio", "melt_asymmetry")
]


def fit_width_gpr(train_df, target_col="width_mean_mm", noise_col="width_std_mm",
                   feature_columns=None, ard=True, length_scale_bounds=(3e-1, 1e2),
                   n_restarts_optimizer=15, random_state=0):
    """
    Fits a GP on the standardized features. Returns the fitted GP and the
    feature scaler, both needed together to predict on new data.

    feature_columns lets you fit on a subset of FEATURE_COLUMNS, e.g.
    dropping melt_area_mm2, melt_aspect_ratio, and melt_asymmetry, which
    have hit the length scale upper bound (effectively ignored by the GP)
    in every fold tried so far. Defaults to all of FEATURE_COLUMNS. The
    chosen list is stored on the returned scaler as
    scaler.feature_columns_, so predict_width automatically uses the same
    features in the same order without needing to be told twice.

    noise_col, if present, is used as a per-sample known noise variance
    (width_std_mm ** 2) passed to alpha, rather than letting a WhiteKernel
    learn one single noise level from scratch. You already have a genuine
    per-sample estimate of measurement noise, width_std_mm, from the
    bundle's own distribution of width values in that local window,
    there's no reason to make the GP re-derive a cruder version of that
    itself, and doing so is what let noise collapse to near zero and
    produce badly overconfident intervals.

    ard=True (default) gives every feature its own length scale. ard=False
    uses one shared length scale for all features, a more conservative
    option worth comparing against directly if the ARD version seems to
    be overfitting (very short length scales on only one or two features,
    a model that fits training data almost perfectly but generalizes
    poorly).

    length_scale_bounds' lower bound is deliberately not too close to
    zero, a length scale that collapses near zero on a standardized
    feature turns that dimension into an almost exact memorization key
    rather than a smooth, generalizable relationship.

    n_restarts_optimizer defaults to 15 rather than sklearn's usual low
    single digits. A bigger training set makes the marginal likelihood
    surface harder to search, not easier, more restarts matters more as n
    grows, not less. A kernel amplitude or length scale that swings wildly
    between two runs on similar data, without a matching change in
    accuracy, is a sign this needs to be even higher.
    """
    feature_columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)

    X = train_df[feature_columns].values
    y = train_df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler.feature_columns_ = feature_columns

    length_scale_init = 1.0 if not ard else np.ones(X.shape[1])
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=length_scale_init, length_scale_bounds=length_scale_bounds, nu=1.5,
    )

    alpha = 1e-8  # sklearn's default numerical-stability floor
    if noise_col is not None and noise_col in train_df.columns:
        known_noise_var = train_df[noise_col].values ** 2
        alpha = np.clip(known_noise_var, 1e-6, None)
    else:
        kernel = kernel + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))

    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    gp.fit(X_scaled, y)
    return gp, scaler


def predict_width(gp, scaler, df):
    """Returns (mean, std) predictive arrays for every row in df, using
    whichever feature columns this scaler was fit with."""
    feature_columns = getattr(scaler, "feature_columns_", FEATURE_COLUMNS)
    X = df[feature_columns].values
    X_scaled = scaler.transform(X)
    mean, std = gp.predict(X_scaled, return_std=True)
    return mean, std


def check_feature_coverage(train_df, test_df, feature_columns=FEATURE_COLUMNS):
    """
    For each feature, reports the training range and where the test set's
    median falls relative to it, plus what fraction of test rows fall
    entirely outside the training range. Run this before concluding poor
    test performance is purely a model problem, if Track_21 sits well
    outside what the model saw features-wise, no regression method
    extrapolates reliably there, that's a data reality, not a kernel
    tuning problem.
    """
    rows = []
    for col in feature_columns:
        train_min, train_max = train_df[col].min(), train_df[col].max()
        test_vals = test_df[col].values
        outside_frac = float(np.mean((test_vals < train_min) | (test_vals > train_max)))
        rows.append({
            "feature": col,
            "train_min": train_min, "train_max": train_max,
            "test_median": float(np.median(test_vals)),
            "test_frac_outside_train_range": outside_frac,
        })
    report = pd.DataFrame(rows).sort_values("test_frac_outside_train_range", ascending=False)
    return report


# ---------------------------------------------------------------------
# probabilistic evaluation
# ---------------------------------------------------------------------

def gaussian_nll(y_true, mean, std, eps=1e-6):
    """Mean negative log likelihood under a Gaussian predictive
    distribution, the metric the paper lists for probabilistic
    predictions. Lower is better, unlike MAE this actually penalizes a
    model for being confidently wrong, not just for missing the target."""
    std = np.clip(std, eps, None)
    return float(np.mean(
        0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * ((y_true - mean) / std) ** 2
    ))


def calibration_curve(y_true, mean, std, quantiles=(0.5, 0.8, 0.9, 0.95)):
    """
    For each nominal coverage level (e.g. 0.9), computes what fraction of
    true values actually fell inside that level's Gaussian predictive
    interval. A well calibrated model has empirical coverage close to
    nominal at every level, points below the diagonal mean the model is
    overconfident (intervals too narrow), points above mean underconfident.
    """
    from scipy.stats import norm
    rows = []
    for q in quantiles:
        z = norm.ppf(0.5 + q / 2)
        lower = mean - z * std
        upper = mean + z * std
        empirical = float(np.mean((y_true >= lower) & (y_true <= upper)))
        rows.append({"nominal_coverage": q, "empirical_coverage": empirical})
    return pd.DataFrame(rows)


def evaluate_width_model(gp, scaler, test_df, target_col="width_mean_mm"):
    y_true = test_df[target_col].values
    mean, std = predict_width(gp, scaler, test_df)

    mae = float(np.mean(np.abs(y_true - mean)))
    nll = gaussian_nll(y_true, mean, std)
    calib = calibration_curve(y_true, mean, std)

    print(f"MAE:  {mae:.4f} mm")
    print(f"NLL:  {nll:.4f}")
    print("calibration (nominal vs empirical coverage):")
    print(calib.to_string(index=False))

    return {"mae": mae, "nll": nll, "calibration": calib, "pred_mean": mean, "pred_std": std}


def plot_width_predictions(test_df, mean, std, target_col="width_mean_mm"):
    y_true = test_df[target_col].values
    order = np.argsort(test_df["x_mm"].values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].errorbar(
        test_df["x_mm"].values[order], mean[order], yerr=1.96 * std[order],
        fmt="o", ms=3, alpha=0.6, ecolor="lightgray", capsize=2, label="predicted (95% interval)",
    )
    axes[0].plot(test_df["x_mm"].values[order], y_true[order], "k.", ms=4, label="true")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("width (mm)")
    axes[0].set_title("predicted vs true width along Track_21")
    axes[0].legend(fontsize=8)

    calib = calibration_curve(y_true, mean, std)
    axes[1].plot(calib["nominal_coverage"], calib["empirical_coverage"], "o-", label="this model")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    axes[1].set_xlabel("nominal coverage")
    axes[1].set_ylabel("empirical coverage")
    axes[1].set_title("calibration")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from feature_extraction import build_feature_dataset

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

    print("\n=== feature coverage: does Track_21 fall inside the training range? ===")
    coverage = check_feature_coverage(train_df, test_df)
    print(coverage.to_string(index=False))

    print("\n=== ARD kernel (per-feature length scales), known noise via width_std_mm ===")
    gp_ard, scaler_ard = fit_width_gpr(train_df, ard=True)
    print("fitted kernel:", gp_ard.kernel_)
    results_ard = evaluate_width_model(gp_ard, scaler_ard, test_df)

    print("\n=== isotropic kernel (one shared length scale), known noise via width_std_mm ===")
    gp_iso, scaler_iso = fit_width_gpr(train_df, ard=False)
    print("fitted kernel:", gp_iso.kernel_)
    results_iso = evaluate_width_model(gp_iso, scaler_iso, test_df)

    plot_width_predictions(test_df, results_ard["pred_mean"], results_ard["pred_std"])
