#!/usr/bin/env python3
"""Train the competitive neural model."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from competitive import CompetitiveConfig, train_competitive_model
from probabilities import compute_probabilities, require_binary_values, safe_correlation


EXPECTED_PROPS = 44
DEFAULT_SAMPLE_PATH = Path("data/pool_mcmc_samples.csv.gz")
DEFAULT_OUTPUT_PATH = Path("trained_models/competitive_150_units_seed19.npz")
EXPECTED_ROUNDED_CORRELATIONS = {
    "Prior": "1.000",
    "Conjunction": "1.000",
    "Conditional": "0.997",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the competitive model on shared Pool MCMC "
            "samples and verify the paper's probability correlations."
        )
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help=f"Training sample CSV path. Default: {DEFAULT_SAMPLE_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output .npz path for learned weights. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--n-units", type=int, default=150)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--initial-weight-lr", type=float, default=0.3)
    parser.add_argument("--final-weight-lr", type=float, default=0.05)
    parser.add_argument("--initial-bias-lr", type=float, default=1e-4)
    parser.add_argument("--final-bias-lr", type=float, default=1e-5)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=19,
        help="Random seed for training-order permutations. Default: 19",
    )
    return parser.parse_args()


def load_samples(path: Path) -> tuple[np.ndarray, list[str]]:
    dataframe = pd.read_csv(path)
    if dataframe.shape[1] != EXPECTED_PROPS:
        raise ValueError(
            f"Expected {EXPECTED_PROPS} proposition columns, got {dataframe.shape[1]}."
        )
    observations = dataframe.to_numpy()
    require_binary_values(observations, "Training samples")
    return observations.astype(float), list(dataframe.columns)


def correlation_rows(observations: np.ndarray, weights: np.ndarray) -> list[tuple[str, int, int, float, str]]:
    true_probs = compute_probabilities(observations)
    estimated_probs = compute_probabilities(weights)
    comparisons = [
        ("Prior", true_probs.priors, estimated_probs.priors),
        ("Conjunction", true_probs.conjunctions, estimated_probs.conjunctions),
        ("Conditional", true_probs.conditionals, estimated_probs.conditionals),
    ]

    rows = []
    for label, true_values, estimated_values in comparisons:
        correlation = safe_correlation(true_values, estimated_values)
        rows.append(
            (
                label,
                int(np.isfinite(true_values).sum()),
                int(np.isfinite(estimated_values).sum()),
                correlation,
                EXPECTED_ROUNDED_CORRELATIONS[label],
            )
        )
    return rows


def validate_rows(rows: list[tuple[str, int, int, float, str]]) -> None:
    failures = []
    for label, _, _, correlation, expected_rounded in rows:
        actual_rounded = f"{correlation:.3f}"
        if actual_rounded != expected_rounded:
            failures.append(
                f"{label}: expected r = {expected_rounded}, found r = {actual_rounded}"
            )
    if failures:
        raise ValueError("; ".join(failures))


def save_model(
    output_path: Path,
    weights: np.ndarray,
    biases: np.ndarray,
    config: CompetitiveConfig,
    columns: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        weights=weights,
        biases=biases,
        config=asdict(config),
        feature_names=np.asarray(columns, dtype=object),
    )


def print_rows(rows: list[tuple[str, int, int, float, str]]) -> None:
    print("Trained competitive-model probability correlations")
    print(f"{'Quantity':<14} {'True N':>8} {'Est. N':>8} {'r':>10} {'Paper r':>10}")
    print("-" * 55)
    for label, true_count, estimated_count, correlation, expected_rounded in rows:
        print(
            f"{label:<14} {true_count:>8} {estimated_count:>8} "
            f"{correlation:>10.6f} {expected_rounded:>10}"
        )


def print_progress(epoch: int, total_epochs: int, weight_lr: float, bias_lr: float) -> None:
    print(
        f"Epoch {epoch}/{total_epochs}: "
        f"weight_lr={weight_lr:.6f}, bias_lr={bias_lr:.8f}"
    )


def main() -> None:
    args = parse_args()
    observations, columns = load_samples(args.samples)
    config = CompetitiveConfig(
        n_units=args.n_units,
        n_epochs=args.n_epochs,
        initial_weight_lr=args.initial_weight_lr,
        final_weight_lr=args.final_weight_lr,
        initial_bias_lr=args.initial_bias_lr,
        final_bias_lr=args.final_bias_lr,
        random_seed=args.random_seed,
    )

    print(f"Training samples: {observations.shape[0]} rows x {observations.shape[1]} columns")
    print(f"Competitive units: {config.n_units}")
    print(f"Epochs: {config.n_epochs}")
    print(f"Random seed: {config.random_seed}")

    model = train_competitive_model(observations, config, progress_callback=print_progress)
    save_model(args.output, model.weights, model.biases, config, columns)
    print(f"Wrote {args.output}")
    rows = correlation_rows(observations, model.weights)
    print_rows(rows)
    validate_rows(rows)
    print()
    print("Rounded correlations match the paper values.")


if __name__ == "__main__":
    main()
