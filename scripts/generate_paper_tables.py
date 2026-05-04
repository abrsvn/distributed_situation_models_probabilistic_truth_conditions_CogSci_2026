#!/usr/bin/env python3
"""Generate TeX table snippets for the CogSci paper."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from probabilities import (
    CONDITIONAL_DENOMINATOR_TOLERANCE,
    compute_probabilities,
    require_binary_values,
    safe_correlation,
)

from compare_sampling_methods import (
    PAPER_HARD_TESTS,
    PAPER_SOFT_TESTS,
    compute_test_rows,
    load_pool_samples,
    load_sequential_samples,
    validate_matching_columns,
)


EXPECTED_PROPS = 44
EXPECTED_UNITS = 150
DEFAULT_POOL_PATH = Path("data/pool_mcmc_samples.csv.gz")
DEFAULT_SEQUENTIAL_PATH = Path("data/sequential_samples.csv.gz")
DEFAULT_WEIGHTS_PATH = Path("trained_models/competitive_150_units_seed19.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CogSci paper TeX tables from shared data files."
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
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Competitive proposition weights path. Default: {DEFAULT_WEIGHTS_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for TeX snippets. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_pool_dataframe(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    if dataframe.shape[1] != EXPECTED_PROPS:
        raise ValueError(
            f"Expected {EXPECTED_PROPS} proposition columns, got {dataframe.shape[1]}."
        )
    require_binary_values(dataframe.to_numpy(), "Pool MCMC proposition samples")
    return dataframe


def load_weights(path: Path) -> np.ndarray:
    model_file = np.load(path, allow_pickle=True)
    weights = model_file["weights"]
    if weights.shape != (EXPECTED_UNITS, EXPECTED_PROPS):
        raise ValueError(
            f"Competitive weight shape is {weights.shape}; "
            f"expected {(EXPECTED_UNITS, EXPECTED_PROPS)}."
        )
    if not np.isfinite(weights).all():
        raise ValueError("Competitive weights contain non-finite values.")
    return weights


def prop_index(columns: list[str], proposition: str) -> int:
    return columns.index(proposition)


def prior(weights: np.ndarray, idx: int) -> float:
    return float(np.mean(weights[:, idx]))


def conditional(weights: np.ndarray, outcome_idx: int, condition_idx: int) -> float:
    condition_prior = prior(weights, condition_idx)
    if condition_prior <= CONDITIONAL_DENOMINATOR_TOLERANCE:
        return float("nan")
    joint = float(np.mean(weights[:, outcome_idx] * weights[:, condition_idx]))
    return joint / condition_prior


def conditional_given_conjunction(
    weights: np.ndarray,
    outcome_idx: int,
    first_condition_idx: int,
    second_condition_idx: int,
) -> float:
    condition = weights[:, first_condition_idx] * weights[:, second_condition_idx]
    condition_prior = float(np.mean(condition))
    if condition_prior <= CONDITIONAL_DENOMINATOR_TOLERANCE:
        return float("nan")
    joint = float(np.mean(weights[:, outcome_idx] * condition))
    return joint / condition_prior


def format_prob(value: float) -> str:
    return f"{value:.2f}"


def format_delta(value: float) -> str:
    if abs(value) < 0.005:
        return "0.00"
    return f"{value:+.2f}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {display_path(path)}")


def sampling_table(pool_path: Path, sequential_path: Path) -> str:
    pool = load_pool_samples(pool_path)
    sequential = load_sequential_samples(sequential_path)
    validate_matching_columns(pool, sequential)

    soft_rows = compute_test_rows(pool, sequential, PAPER_SOFT_TESTS)
    hard_rows = compute_test_rows(pool, sequential, PAPER_HARD_TESTS)

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "\\textbf{Conditional} & \\textbf{Pool} & \\textbf{Seq.} & \\textbf{$\\Delta$} \\\\",
        "\\midrule",
    ]
    for label, pool_prob, sequential_prob, delta in soft_rows:
        lines.append(
            f"{sampling_label_tex(label)} & {format_prob(pool_prob)} & "
            f"{format_prob(sequential_prob)} & {format_delta(delta)} \\\\"
        )
    lines.append("\\midrule")
    for label, pool_prob, sequential_prob, delta in hard_rows:
        lines.append(
            f"{sampling_label_tex(label)} & {format_prob(pool_prob)} & "
            f"{format_prob(sequential_prob)} & {format_delta(delta)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{-0.5em}",
            "\\caption{\\small Cond.\\ prob.s estimated from Pool MCMC vs.\\ seq.\\ samples. Soft constraints (top) show large discrepancies; hard constraints (bottom, not dependent on sampling order) agree.}",
            "\\label{tab:sampling-comparison}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def sampling_label_tex(label: str) -> str:
    labels = {
        "P(win|play_well) Charlie": "$P(\\text{win}|\\text{play\\_well})$ Charlie",
        "P(win|play_well) Heidi": "$P(\\text{win}|\\text{play\\_well})$ Heidi",
        "P(win|play_well) Sophia": "$P(\\text{win}|\\text{play\\_well})$ Sophia",
        "P(lose|play_badly) Charlie": "$P(\\text{lose}|\\text{play\\_badly})$ Charlie",
        "P(win|specialty) Charlie": "$P(\\text{win}|\\text{specialty})$ Charlie",
        "P(win|specialty) Heidi": "$P(\\text{win}|\\text{specialty})$ Heidi",
        "P(win|specialty) Sophia": "$P(\\text{win}|\\text{specialty})$ Sophia",
        "P(win|lose)": "$P(\\text{win}|\\text{lose})$",
        "P(street|soccer)": "$P(\\text{street}|\\text{soccer})$",
        "P(ball|soccer)": "$P(\\text{ball}|\\text{soccer})$",
    }
    return labels[label]


def correlation_table(samples: pd.DataFrame, weights: np.ndarray) -> str:
    true_probs = compute_probabilities(samples.to_numpy())
    estimated_probs = compute_probabilities(weights)
    rows = [
        ("Prior $P(p)$", true_probs.priors, estimated_probs.priors),
        ("Conjunction $P(p \\land q)$", true_probs.conjunctions, estimated_probs.conjunctions),
        ("Conditional $P(p \\mid q)$", true_probs.conditionals, estimated_probs.conditionals),
    ]

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lc@{}}",
        "\\toprule",
        "\\textbf{Probability Type} & \\textbf{Correlation ($r$)} \\\\",
        "\\midrule",
    ]
    for label, true_values, estimated_values in rows:
        lines.append(f"{label} & {safe_correlation(true_values, estimated_values):.3f} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{-0.5em}",
            "\\caption{\\small Correlation between probability estimates from learned vectors and frequencies from Pool MCMC samples.}",
            "\\label{tab:correlations}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def hard_inference_table(columns: list[str], weights: np.ndarray) -> str:
    rows = [
        (
            "$P(\\text{win(charlie)} \\mid \\text{lose(charlie)})$",
            conditional(
                weights,
                prop_index(columns, "win(charlie)"),
                prop_index(columns, "lose(charlie)"),
            ),
        ),
        (
            "$P(\\text{play(charlie,chess)} \\mid \\text{play(charlie,soccer)})$",
            conditional(
                weights,
                prop_index(columns, "play(charlie,chess)"),
                prop_index(columns, "play(charlie,soccer)"),
            ),
        ),
        (
            "$P(\\text{location(charlie,street)} \\mid \\text{play(charlie,soccer)})$",
            conditional(
                weights,
                prop_index(columns, "location(charlie,street)"),
                prop_index(columns, "play(charlie,soccer)"),
            ),
        ),
        (
            "$P(\\text{play(heidi,ball)} \\mid \\text{play(heidi,soccer)})$",
            conditional(
                weights,
                prop_index(columns, "play(heidi,ball)"),
                prop_index(columns, "play(heidi,soccer)"),
            ),
        ),
    ]

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lc@{}}",
        "\\toprule",
        "\\textbf{Conditional} & \\textbf{$P$} \\\\",
        "\\midrule",
        "\\multicolumn{2}{@{}l@{}}{\\textit{Incompatibility ($P \\approx 0$)}} \\\\",
    ]
    for label, value in rows[:2]:
        lines.append(f"{label} & {format_prob(value)} \\\\")
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{2}{@{}l@{}}{\\textit{Implication ($P \\approx 1$)}} \\\\",
        ]
    )
    for label, value in rows[2:]:
        lines.append(f"{label} & {format_prob(value)} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{-0.5em}",
            "\\caption{\\small Hard constraints yield extreme conditional probabilities based on the learned situation-model vectors.}",
            "\\label{tab:hard-inference}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def soft_inference_table(columns: list[str], weights: np.ndarray) -> str:
    win_charlie_idx = prop_index(columns, "win(charlie)")
    lose_charlie_idx = prop_index(columns, "lose(charlie)")
    with_ease_idx = prop_index(columns, "win_manner(with_ease)")

    rows = [
        (
            "$P(\\text{win(C)} \\mid \\text{play\\_manner(C, well)})$",
            conditional(
                weights,
                win_charlie_idx,
                prop_index(columns, "play_manner(charlie,well)"),
            ),
            prior(weights, win_charlie_idx),
        ),
        (
            "$P(\\text{lose(C)} \\mid \\text{play\\_manner(C, badly)})$",
            conditional(
                weights,
                lose_charlie_idx,
                prop_index(columns, "play_manner(charlie,badly)"),
            ),
            prior(weights, lose_charlie_idx),
        ),
    ]
    with_ease_conditional = conditional_given_conjunction(
        weights,
        with_ease_idx,
        win_charlie_idx,
        prop_index(columns, "play(charlie,chess)"),
    )
    with_ease_prior = prior(weights, with_ease_idx)

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "\\textbf{Conditional} & \\textbf{$P(p|q)$} & \\textbf{Prior $P(p)$} \\\\",
        "\\midrule",
    ]
    for label, conditional_value, prior_value in rows:
        lines.append(
            f"{label} & {format_prob(conditional_value)} & {format_prob(prior_value)} \\\\"
        )
    lines.extend(
        [
            f"$P(\\text{{win(with\\_ease)}} \\mid \\text{{win(C)}} \\; \\land$ & {format_prob(with_ease_conditional)} & {format_prob(with_ease_prior)} \\\\",
            "\\phantom{$P(\\text{win(with\\_ease)} \\mid$} $\\text{play(C, chess)})$ &  &  \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{-0.5em}",
            "\\caption{\\small Soft constraints yield elevated conditional probabilities $P(p|q)$. Prior probabilities $P(p)$ shown for comparison.}",
            "\\label{tab:soft-inference}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_pool_dataframe(REPO_ROOT / args.pool)
    weights = load_weights(REPO_ROOT / args.weights)
    columns = list(samples.columns)

    write_text(
        output_dir / "sampling_comparison_table.tex",
        sampling_table(REPO_ROOT / args.pool, REPO_ROOT / args.sequential),
    )
    write_text(
        output_dir / "vector_correlations_table.tex",
        correlation_table(samples, weights),
    )
    write_text(
        output_dir / "hard_inference_table.tex",
        hard_inference_table(columns, weights),
    )
    write_text(
        output_dir / "soft_inference_table.tex",
        soft_inference_table(columns, weights),
    )


if __name__ == "__main__":
    main()
