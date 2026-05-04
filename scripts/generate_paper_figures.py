#!/usr/bin/env python3
"""Generate paper figures from shared samples and trained competitive weights."""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsm_matplotlib_config"),
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_PATH = Path("data/pool_mcmc_samples.csv.gz")
DEFAULT_WEIGHTS_PATH = Path("trained_models/competitive_150_units_seed19.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_figures")
EXPECTED_PROPS = 44
EXPECTED_UNITS = 150

n_samples_show = 100

selected_props = [
    ("win(charlie)", "C wins"),
    ("lose(charlie)", "C loses"),
    ("play(charlie,chess)", "C plays chess"),
    ("play(charlie,soccer)", "C plays soccer"),
    ("play(charlie,soccer)", "C plays soccer"),
    ("location(charlie,street)", "C @ street"),
    ("play(heidi,soccer)", "H plays soccer"),
    ("play(heidi,ball)", "H plays ball"),
    ("play_manner(charlie,well)", "C plays well"),
    ("win(charlie)", "C wins"),
]

FOCUSED_PAIRS = [
    (1, 0, "HARD INCOMP.", "#d62728", "P(win|lose)"),
    (3, 2, "HARD INCOMP.", "#d62728", "P(chess|soccer)"),
    (4, 5, "HARD IMPLIC.", "#2ca02c", "P(street|soccer)"),
    (6, 7, "HARD IMPLIC.", "#2ca02c", "P(ball|soccer)"),
    (8, 9, "SOFT CORREL.", "#1f77b4", "P(win|well)"),
]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the paper figures from shared samples and trained weights."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES_PATH,
        help=f"Pool MCMC sample CSV path. Default: {DEFAULT_SAMPLES_PATH}",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Competitive weights path. Default: {DEFAULT_WEIGHTS_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_samples(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    if dataframe.shape[1] != EXPECTED_PROPS:
        raise ValueError(
            f"Expected {EXPECTED_PROPS} proposition columns, got {dataframe.shape[1]}."
        )
    if not np.isin(dataframe.to_numpy(), [0, 1]).all():
        raise ValueError("Pool MCMC samples must contain only binary 0/1 values.")
    return dataframe


def load_weights(path: Path, columns: list[str]) -> np.ndarray:
    model_file = np.load(path, allow_pickle=True)
    weights = model_file["weights"]
    feature_names = [str(name) for name in model_file["feature_names"]]
    if weights.shape != (EXPECTED_UNITS, EXPECTED_PROPS):
        raise ValueError(
            f"Competitive weight shape is {weights.shape}; "
            f"expected {(EXPECTED_UNITS, EXPECTED_PROPS)}."
        )
    if feature_names != columns:
        raise ValueError("Competitive weight feature names do not match sample columns.")
    if not np.isfinite(weights).all():
        raise ValueError("Competitive weights contain non-finite values.")
    return weights


def conditional(weights: np.ndarray, outcome_idx: int, condition_idx: int) -> float:
    condition_prior = float(np.mean(weights[:, condition_idx]))
    if condition_prior <= 1e-12:
        return float("nan")
    joint = float(np.mean(weights[:, outcome_idx] * weights[:, condition_idx]))
    return joint / condition_prior


def add_bracket(
    ax: plt.Axes,
    y1: int,
    y2: int,
    probability: float,
    label: str,
    color: str,
) -> None:
    x_right = 152
    ax.plot([x_right, x_right], [y1 + 0.5, y2 + 0.5], color=color, lw=2, clip_on=False)
    ax.plot([x_right - 1.5, x_right], [y1 + 0.5, y1 + 0.5], color=color, lw=2, clip_on=False)
    ax.plot([x_right - 1.5, x_right], [y2 + 0.5, y2 + 0.5], color=color, lw=2, clip_on=False)
    ax.text(
        x_right + 2,
        (y1 + y2) / 2 + 0.5,
        f"{label}\n={probability:.2f}",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


def write_sample_to_vector_figure(
    samples: pd.DataFrame,
    weights: np.ndarray,
    output_path: Path,
) -> None:
    prop_names = list(samples.columns)
    prop_index = {name: idx for idx, name in enumerate(prop_names)}
    indices = [prop_index[name] for name, _ in selected_props]
    labels = [label for _, label in selected_props]
    sample_values = samples.to_numpy()[:, indices]
    selected_weights = weights[:, indices]

    np.random.seed(42)
    sample_indices = np.sort(np.random.choice(len(sample_values), n_samples_show, replace=False))
    sample_subset = sample_values[sample_indices]

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.1, 2]})

    binary_cmap = ListedColormap(["#FFF0F5", "#003366"])
    axes[0].pcolormesh(
        np.arange(n_samples_show + 1),
        np.arange(len(labels) + 1),
        sample_subset.T,
        cmap=binary_cmap,
        shading="flat",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0].set_yticks(np.arange(len(labels)) + 0.5)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"Pool MCMC Samples ({n_samples_show} of 250,000)", fontsize=10)
    axes[0].set_title("Binary Model-Theoretic Training Data: True=1 / False=0", fontsize=11, pad=8)

    continuous_cmap = LinearSegmentedColormap.from_list("custom", ["#FFF0F5", "#003366"])
    image = axes[1].pcolormesh(
        np.arange(EXPECTED_UNITS + 1),
        np.arange(len(labels) + 1),
        selected_weights.T,
        cmap=continuous_cmap,
        shading="flat",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.1,
    )
    axes[1].set_yticks([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("150 Competitive Neural-Network Units", fontsize=10)
    axes[1].set_title("Learned Distributed Situation Models", fontsize=11, pad=8)

    cbar_ax = figure.add_axes([0.905, 0.15, 0.012, 0.70])
    colorbar = plt.colorbar(image, cax=cbar_ax)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_label("Continuous Activation in $[0,1]$", fontsize=10)

    print("Conditional probabilities P(p|q) for sample-to-vector figure")
    for condition_row, outcome_row, constraint_type, color, label in FOCUSED_PAIRS:
        condition_idx = indices[condition_row]
        outcome_idx = indices[outcome_row]
        probability = conditional(weights, outcome_idx, condition_idx)
        print(f"  {label:<20}: P = {probability:.2f}  [{constraint_type}]")
        y1, y2 = min(condition_row, outcome_row), max(condition_row, outcome_row)
        add_bracket(axes[1], y1, y2, probability, label, color)

    plt.suptitle(
        "From Binary Model-Theoretic Data to Distributed Situation Models "
        "(only some of the 44 atomic propositions are shown on $y$ axis)",
        fontsize=13,
        y=0.98,
    )
    plt.subplots_adjust(left=0.11, right=0.84, top=0.88, bottom=0.12, wspace=0.08)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    samples = load_samples(REPO_ROOT / args.samples)
    weights = load_weights(REPO_ROOT / args.weights, list(samples.columns))
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Samples: {samples.shape}")
    print(f"Weights: {weights.shape}")

    focused_output = output_dir / "samples_to_vectors_focused.png"
    write_sample_to_vector_figure(samples, weights, focused_output)
    print(f"Wrote {display_path(focused_output)}")


if __name__ == "__main__":
    main()
