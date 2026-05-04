"""Probability computations for proposition samples and learned weights."""

from dataclasses import dataclass

import numpy as np


CONDITIONAL_DENOMINATOR_TOLERANCE = 1e-12
CORRELATION_STD_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ProbabilitySet:
    priors: np.ndarray
    conjunctions: np.ndarray
    conditionals: np.ndarray


def require_binary_values(values: np.ndarray, label: str) -> None:
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{label} must contain only binary 0/1 values.")


def compute_probabilities(values: np.ndarray) -> ProbabilitySet:
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D value matrix, got {values.ndim}D.")

    n_props = values.shape[1]
    priors = np.mean(values, axis=0)

    conjunctions = []
    for left_idx in range(n_props):
        for right_idx in range(left_idx + 1, n_props):
            conjunctions.append(np.mean(values[:, left_idx] * values[:, right_idx]))

    conditionals = []
    for outcome_idx in range(n_props):
        for condition_idx in range(n_props):
            if outcome_idx != condition_idx:
                joint = np.mean(values[:, outcome_idx] * values[:, condition_idx])
                if priors[condition_idx] > CONDITIONAL_DENOMINATOR_TOLERANCE:
                    conditionals.append(joint / priors[condition_idx])
                else:
                    conditionals.append(float("nan"))

    return ProbabilitySet(
        priors=priors,
        conjunctions=np.asarray(conjunctions),
        conditionals=np.asarray(conditionals),
    )


def safe_correlation(true_values: np.ndarray, estimated_values: np.ndarray) -> float:
    if true_values.shape != estimated_values.shape:
        raise ValueError(
            "Correlation inputs must have matching shapes, "
            f"got {true_values.shape} and {estimated_values.shape}."
        )
    if true_values.size == 0:
        return float("nan")

    valid = np.isfinite(true_values) & np.isfinite(estimated_values)
    if not np.any(valid):
        return float("nan")

    true_valid = true_values[valid]
    estimated_valid = estimated_values[valid]
    if (
        np.std(true_valid) < CORRELATION_STD_TOLERANCE
        or np.std(estimated_valid) < CORRELATION_STD_TOLERANCE
    ):
        return float("nan")

    correlation = float(np.corrcoef(true_valid, estimated_valid)[0, 1])
    return correlation if np.isfinite(correlation) else float("nan")
