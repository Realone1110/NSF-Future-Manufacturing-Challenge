"""
Shared utilities for the three generate_*_results.py pipelines: consistent
publication styling, saving each plot as its own file instead of a combined
multi-panel figure, simple wall-clock timing, and model persistence via
joblib so a closed notebook doesn't mean retraining from scratch.
"""

import os
import time
import joblib
import matplotlib.pyplot as plt

MODELS_DIR = os.path.join("ML_results", "models")


def apply_manuscript_style():
    """Times New Roman (falls back to Liberation Serif, metrically
    compatible, if Times isn't installed) at a larger, print-legible size,
    with a high savefig DPI. Call once near the top of a script, before any
    figures are created."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 150,     # on-screen preview
        "savefig.dpi": 400,    # exported file quality
    })


def save_fig(fig, results_dir, stem, panel_name):
    """Saves one figure as its own file: <stem>_<panel_name>.png, e.g.
    Width_result_20260727_120000_main.png. Returns the path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{stem}_{panel_name}.png")
    fig.savefig(path, dpi=400, bbox_inches="tight")
    return path


class Timer:
    """Minimal wall-clock stopwatch for reporting compute time per stage,
    e.g. data extraction vs model fitting vs evaluation, for the report's
    reproducibility/computational-cost reporting.

    Usage:
        timer = Timer()
        with timer.stage("data extraction"):
            ...
        with timer.stage("model fitting"):
            ...
        print(timer.summary())
    """

    def __init__(self):
        self.durations = {}

    class _Stage:
        def __init__(self, timer, name):
            self.timer, self.name = timer, name

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.timer.durations[self.name] = time.perf_counter() - self._start

    def stage(self, name):
        return self._Stage(self, name)

    def summary(self):
        lines = [f"  {name}: {seconds:.2f}s" for name, seconds in self.durations.items()]
        lines.append(f"  total: {sum(self.durations.values()):.2f}s")
        return "\n".join(lines)

    def as_dict(self):
        d = dict(self.durations)
        d["total_seconds"] = sum(self.durations.values())
        return d


def save_model(obj, stem, tag):
    """Saves any picklable model object (a (gp, scaler) tuple, or a dict of
    those for the boundary/contour multi-target models) to
    ML_results/models/<stem>_<tag>.joblib. Returns the path."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{stem}_{tag}.joblib")
    joblib.dump(obj, path)
    return path


def load_model(path):
    """Loads back whatever save_model wrote, a (gp, scaler) tuple or a
    dict of them, exactly as it was before saving."""
    return joblib.load(path)
