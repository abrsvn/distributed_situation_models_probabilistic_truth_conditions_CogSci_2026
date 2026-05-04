"""Proposition-only competitive neural model."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompetitiveConfig:
    n_units: int = 150
    n_epochs: int = 30
    initial_weight_lr: float = 0.3
    final_weight_lr: float = 0.05
    initial_bias_lr: float = 1e-4
    final_bias_lr: float = 1e-5
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_units <= 0:
            raise ValueError(f"n_units must be positive, got {self.n_units}.")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}.")
        if not (0 < self.initial_weight_lr <= 1):
            raise ValueError(
                f"initial_weight_lr must be in (0, 1], got {self.initial_weight_lr}."
            )
        if not (0 < self.final_weight_lr <= self.initial_weight_lr):
            raise ValueError(
                "final_weight_lr must be in (0, initial_weight_lr], "
                f"got {self.final_weight_lr}."
            )
        if not (0 < self.initial_bias_lr <= 1):
            raise ValueError(
                f"initial_bias_lr must be in (0, 1], got {self.initial_bias_lr}."
            )
        if not (0 < self.final_bias_lr <= self.initial_bias_lr):
            raise ValueError(
                "final_bias_lr must be in (0, initial_bias_lr], "
                f"got {self.final_bias_lr}."
            )


@dataclass(frozen=True)
class TrainingHistory:
    weight_lrs: list[float]
    bias_lrs: list[float]


@dataclass(frozen=True)
class CompetitiveModel:
    weights: np.ndarray
    biases: np.ndarray
    history: TrainingHistory


def learning_rates(config: CompetitiveConfig, epoch: int) -> tuple[float, float]:
    if epoch < config.n_epochs // 3:
        progress = epoch / (config.n_epochs // 3)
        weight_lr = config.initial_weight_lr + progress * (
            config.final_weight_lr - config.initial_weight_lr
        )
        bias_lr = config.initial_bias_lr + progress * (
            config.final_bias_lr - config.initial_bias_lr
        )
    else:
        weight_lr = config.final_weight_lr
        bias_lr = config.final_bias_lr

    return weight_lr, bias_lr


def quantization_error(weights: np.ndarray, biases: np.ndarray, observations: np.ndarray) -> float:
    total_error = 0.0
    for observation in observations:
        distances = np.sum(np.abs(weights - observation), axis=1)
        winner_idx = int(np.argmin(distances - biases))
        total_error += float(distances[winner_idx])
    return total_error / len(observations)


def train_competitive_model(
    observations: np.ndarray,
    config: CompetitiveConfig,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
) -> CompetitiveModel:
    if observations.ndim != 2:
        raise ValueError(f"Expected a 2D observation matrix, got {observations.ndim}D.")
    if not np.isfinite(observations).all():
        raise ValueError("Observations contain non-finite values.")

    rng = np.random.default_rng(config.random_seed)
    weights = np.full((config.n_units, observations.shape[1]), 0.5, dtype=float)
    biases = np.ones(config.n_units, dtype=float)
    weight_lrs = []
    bias_lrs = []

    for epoch in range(config.n_epochs):
        weight_lr, bias_lr = learning_rates(config, epoch)
        for idx in rng.permutation(len(observations)):
            observation = observations[idx]
            distances = np.sum(np.abs(weights - observation), axis=1)
            winner_idx = int(np.argmin(distances - biases))

            weights[winner_idx] += weight_lr * (observation - weights[winner_idx])
            bias_delta = biases.copy()
            bias_delta[winner_idx] = biases[winner_idx] * (1 - biases[winner_idx])
            biases += bias_lr * bias_delta

        weight_lrs.append(weight_lr)
        bias_lrs.append(bias_lr)
        if progress_callback is not None:
            progress_callback(epoch + 1, config.n_epochs, weight_lr, bias_lr)

    return CompetitiveModel(
        weights=weights,
        biases=biases,
        history=TrainingHistory(
            weight_lrs=weight_lrs,
            bias_lrs=bias_lrs,
        ),
    )
