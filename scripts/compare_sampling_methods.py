#!/usr/bin/env python3
"""Compare Pool MCMC and sequential sample files used in the paper."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 250_000
EXPECTED_PROPS = 44

DEFAULT_POOL_PATH = Path("data/pool_mcmc_samples.csv.gz")
DEFAULT_SEQUENTIAL_PATH = Path("data/sequential_samples.csv.gz")


@dataclass(frozen=True)
class SampleData:
    name: str
    values: np.ndarray
    columns: list[str]
    total_columns: int


@dataclass(frozen=True)
class ConditionalTest:
    condition: str
    outcome: str
    label: str


PAPER_SOFT_TESTS = [
    ConditionalTest(
        "play_manner(charlie,well)",
        "win(charlie)",
        "P(win|play_well) Charlie",
    ),
    ConditionalTest(
        "play_manner(heidi,well)",
        "win(heidi)",
        "P(win|play_well) Heidi",
    ),
    ConditionalTest(
        "play_manner(sophia,well)",
        "win(sophia)",
        "P(win|play_well) Sophia",
    ),
    ConditionalTest(
        "play_manner(charlie,badly)",
        "lose(charlie)",
        "P(lose|play_badly) Charlie",
    ),
    ConditionalTest(
        "play(charlie,chess)",
        "win(charlie)",
        "P(win|specialty) Charlie",
    ),
    ConditionalTest(
        "play(heidi,hide_n_seek)",
        "win(heidi)",
        "P(win|specialty) Heidi",
    ),
    ConditionalTest(
        "play(sophia,soccer)",
        "win(sophia)",
        "P(win|specialty) Sophia",
    ),
]

PAPER_HARD_TESTS = [
    ConditionalTest("lose(charlie)", "win(charlie)", "P(win|lose)"),
    ConditionalTest(
        "play(charlie,soccer)",
        "location(charlie,street)",
        "P(street|soccer)",
    ),
    ConditionalTest(
        "play(heidi,soccer)",
        "play(heidi,ball)",
        "P(ball|soccer)",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the CogSci paper comparison between Pool MCMC and "
            "sequential sample files."
        )
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=DEFAULT_POOL_PATH,
        help=f"Pool MCMC CSV path. Default: {DEFAULT_POOL_PATH}",
    )
    parser.add_argument(
        "--sequential",
        type=Path,
        default=DEFAULT_SEQUENTIAL_PATH,
        help=f"Sequential samples CSV path. Default: {DEFAULT_SEQUENTIAL_PATH}",
    )
    return parser.parse_args()


def require_binary(values: np.ndarray, name: str) -> None:
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary 0/1 sample values.")


def load_pool_samples(path: Path) -> SampleData:
    dataframe = pd.read_csv(path)
    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Pool MCMC row count is {len(dataframe)}; expected {EXPECTED_ROWS}."
        )
    if len(dataframe.columns) != EXPECTED_PROPS:
        raise ValueError(
            f"Pool MCMC CSV has {len(dataframe.columns)} columns; "
            f"expected {EXPECTED_PROPS}."
        )

    columns = list(dataframe.columns)
    values = dataframe.to_numpy()
    require_binary(values, "Pool MCMC samples")
    return SampleData("Pool MCMC", values.astype(np.uint8), columns, len(dataframe.columns))


def load_sequential_samples(path: Path) -> SampleData:
    dataframe = pd.read_csv(path)
    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Sequential row count is {len(dataframe)}; expected {EXPECTED_ROWS}."
        )
    if len(dataframe.columns) != EXPECTED_PROPS:
        raise ValueError(
            f"Sequential CSV has {len(dataframe.columns)} columns; "
            f"expected {EXPECTED_PROPS}."
        )

    columns = list(dataframe.columns)
    values = dataframe.to_numpy()
    require_binary(values, "Sequential samples")
    return SampleData("Sequential", values.astype(np.uint8), columns, len(dataframe.columns))


def validate_matching_columns(pool: SampleData, sequential: SampleData) -> None:
    if pool.columns != sequential.columns:
        missing_from_sequential = sorted(set(pool.columns) - set(sequential.columns))
        missing_from_pool = sorted(set(sequential.columns) - set(pool.columns))
        raise ValueError(
            "Pool MCMC and sequential proposition columns do not match exactly. "
            f"Missing from sequential: {missing_from_sequential}. "
            f"Missing from Pool MCMC: {missing_from_pool}."
        )


def prop_index(columns: list[str], name: str) -> int:
    return columns.index(name)


def conditional_probability(samples: np.ndarray, outcome_idx: int, condition_idx: int) -> float:
    condition_true = samples[:, condition_idx] == 1
    if condition_true.sum() == 0:
        return float("nan")
    return float(samples[condition_true, outcome_idx].mean())


def compute_test_rows(
    pool: SampleData,
    sequential: SampleData,
    tests: list[ConditionalTest],
) -> list[tuple[str, float, float, float]]:
    rows = []
    for test in tests:
        condition_idx = prop_index(pool.columns, test.condition)
        outcome_idx = prop_index(pool.columns, test.outcome)
        pool_prob = conditional_probability(pool.values, outcome_idx, condition_idx)
        sequential_prob = conditional_probability(
            sequential.values,
            outcome_idx,
            condition_idx,
        )
        rows.append((test.label, pool_prob, sequential_prob, pool_prob - sequential_prob))
    return rows


def compute_all_conditionals(samples: np.ndarray) -> np.ndarray:
    n_props = samples.shape[1]
    conditionals = np.full((n_props, n_props), np.nan)
    for outcome_idx in range(n_props):
        for condition_idx in range(n_props):
            if outcome_idx != condition_idx:
                conditionals[outcome_idx, condition_idx] = conditional_probability(
                    samples,
                    outcome_idx,
                    condition_idx,
                )
    return conditionals


def compute_global_stats(pool: SampleData, sequential: SampleData) -> dict[str, float]:
    pool_conditionals = compute_all_conditionals(pool.values)
    sequential_conditionals = compute_all_conditionals(sequential.values)

    off_diagonal = ~np.eye(pool.values.shape[1], dtype=bool)
    pool_flat = pool_conditionals[off_diagonal]
    sequential_flat = sequential_conditionals[off_diagonal]
    valid = ~(np.isnan(pool_flat) | np.isnan(sequential_flat))

    pool_valid = pool_flat[valid]
    sequential_valid = sequential_flat[valid]
    differences = pool_valid - sequential_valid

    return {
        "pairs": float(len(pool_valid)),
        "correlation": float(np.corrcoef(pool_valid, sequential_valid)[0, 1]),
        "mean_abs_delta": float(np.abs(differences).mean()),
        "max_abs_delta": float(np.abs(differences).max()),
        "mean_delta": float(differences.mean()),
    }


def print_validation(pool: SampleData, sequential: SampleData) -> None:
    print("Input validation")
    print(f"  Pool MCMC rows: {pool.values.shape[0]}")
    print(f"  Pool MCMC proposition columns: {pool.values.shape[1]}")
    print(f"  Sequential rows: {sequential.values.shape[0]}")
    print(f"  Sequential proposition columns: {sequential.values.shape[1]}")
    print("  Proposition columns match exactly")
    print("  Sample values are binary 0/1")


def print_rows(title: str, rows: list[tuple[str, float, float, float]]) -> None:
    print()
    print(title)
    print(f"{'Conditional':<32} {'Pool MCMC':>10} {'Sequential':>12} {'Delta':>9}")
    print("-" * 67)
    for label, pool_prob, sequential_prob, delta in rows:
        print(f"{label:<32} {pool_prob:>10.3f} {sequential_prob:>12.3f} {delta:>+9.3f}")


def print_global_stats(stats: dict[str, float]) -> None:
    print()
    print("All pairwise conditional probabilities")
    print(f"  Pairs compared: {int(stats['pairs'])}")
    print(f"  Correlation: r = {stats['correlation']:.4f}")
    print(f"  Mean |delta|: {stats['mean_abs_delta']:.4f}")
    print(f"  Max |delta|: {stats['max_abs_delta']:.4f}")
    print(f"  Mean delta: {stats['mean_delta']:+.4f}")


def main() -> None:
    args = parse_args()
    pool = load_pool_samples(args.pool)
    sequential = load_sequential_samples(args.sequential)
    validate_matching_columns(pool, sequential)

    print_validation(pool, sequential)
    print_rows("Soft constraints", compute_test_rows(pool, sequential, PAPER_SOFT_TESTS))
    print_rows("Hard constraints", compute_test_rows(pool, sequential, PAPER_HARD_TESTS))
    print_global_stats(compute_global_stats(pool, sequential))


if __name__ == "__main__":
    main()
