"""
Full ablation across the three new physics-derived features
(cooling_rate_proxy, thermal_gradient_proxy, linear_energy_density),
added individually and in combination on top of the original 12-feature
set, all evaluated the same way on the same train/test split.

This exists specifically because eyeballing which feature's coverage
violation "looks worse" isn't a reliable way to assign blame, two
features can both contribute, or one can dominate in a way that isn't
obvious from coverage percentages alone. Running all 8 combinations gives
a direct, unambiguous answer instead.
"""

import itertools
import pandas as pd

from gpr_width_model import fit_width_gpr, evaluate_width_model, check_feature_coverage

BASE_FEATURES = [
    "x_frac", "melt_area_mm2", "melt_peak", "melt_mean", "melt_aspect_ratio",
    "melt_asymmetry", "melt_tail_ratio",
    "sem_mean", "sem_std", "sem_texture_mean", "sem_texture_std", "sem_texture_max",
]
CANDIDATE_EXTRAS = ["cooling_rate_proxy", "thermal_gradient_proxy", "linear_energy_density"]


def run_feature_ablation(train_df, test_df, ard=False):
    """
    Fits and evaluates a GP for every subset of CANDIDATE_EXTRAS added to
    BASE_FEATURES (2**3 = 8 combinations, including the empty set, the
    original 12-feature baseline). Returns a DataFrame sorted by MAE, one
    row per combination, with MAE, NLL, and empirical coverage at the 90%
    nominal level so you can see accuracy and calibration together rather
    than picking one and ignoring the other.
    """
    rows = []
    for r in range(len(CANDIDATE_EXTRAS) + 1):
        for combo in itertools.combinations(CANDIDATE_EXTRAS, r):
            feature_columns = BASE_FEATURES + list(combo)
            gp, scaler = fit_width_gpr(train_df, ard=ard, feature_columns=feature_columns)
            results = evaluate_width_model(gp, scaler, test_df)

            calib = results["calibration"]
            cov90_row = calib.loc[calib["nominal_coverage"] == 0.9, "empirical_coverage"]
            cov90 = float(cov90_row.values[0]) if len(cov90_row) else float("nan")

            rows.append({
                "extras_added": ", ".join(combo) if combo else "(none, 12-feature baseline)",
                "n_features": len(feature_columns),
                "mae": results["mae"],
                "nll": results["nll"],
                "empirical_coverage_at_90pct_nominal": cov90,
                "kernel_amplitude": float(gp.kernel_.k1.constant_value) if hasattr(gp.kernel_, "k1") else None,
            })

    df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return df


if __name__ == "__main__":
    from feature_extraction import build_feature_dataset

    train_df = build_feature_dataset(
        track_ids=["Track_8", "Track_10", "Track_14"], n_per_track=300,
        SEM_DIR=SEM_DIR, SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM, seed=0,
    )
    test_df = build_feature_dataset(
        track_ids=["Track_21"], n_per_track=300,
        SEM_DIR=SEM_DIR, SEM_TILE_WIDTH_MM=SEM_TILE_WIDTH_MM, seed=42,
    )

    ablation_df = run_feature_ablation(train_df, test_df, ard=False)
    print(ablation_df.to_string(index=False))
