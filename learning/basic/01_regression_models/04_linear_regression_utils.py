"""Utility functions for a simple linear regression example.

This module defines a linear model f(x) = w * x + b, a mean squared error
cost function, and helper routines to evaluate and visualise how the cost
changes with different slope (w) and intercept (b) values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class LinearRegressionConfig:
    """Configuration object holding training data and intercept."""

    x_values: np.ndarray
    y_values: np.ndarray
    bias: float = 0.0

    @classmethod
    def default(cls) -> "LinearRegressionConfig":
        x_vals = np.arange(-1, stop=6, step=1, dtype=float)
        true_w, true_b = 0.75, 1.5
        y_vals = true_w * x_vals + true_b
        return cls(x_values=x_vals, y_values=y_vals, bias=true_b)


def linear_model(x: np.ndarray | float, weight: float, bias: float) -> np.ndarray:
    """Compute the linear model f(x) = w * x + b."""

    return weight * np.asarray(x, dtype=float) + bias


def mean_squared_cost(
    weight: float, config: LinearRegressionConfig, bias: float | None = None
) -> float:
    """Return 1/(2m) * sum((f(x) - y)^2) for the provided parameters."""

    bias_to_use = config.bias if bias is None else float(bias)
    predictions = linear_model(config.x_values, weight, bias_to_use)
    errors = predictions - config.y_values
    m = config.x_values.size
    return float((errors @ errors) / (2 * m))


def evaluate_weight(
    weight: float, bias: float | None = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate predictions and cost for x in [-1, 5] given a weight.

    Parameters
    ----------
    weight: float
        The slope (w) to use in the linear model.
    bias: Optional[float]
        If provided, overrides the default bias from the configuration.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        A tuple with the x values, predicted y values, and the cost.
    """

    config = LinearRegressionConfig.default()
    bias_to_use = config.bias if bias is None else float(bias)

    predictions = linear_model(config.x_values, weight, bias_to_use)
    cost_value = mean_squared_cost(weight, config, bias=bias_to_use)
    return config.x_values, predictions, cost_value


def plot_cost_curve(weights: Iterable[float], bias: float | None = None) -> None:
    """Plot the cost function over a collection of weight values."""

    config = LinearRegressionConfig.default()
    weights = np.asarray(list(weights), dtype=float)
    bias_to_use = config.bias if bias is None else float(bias)
    costs = np.array([mean_squared_cost(w, config, bias=bias_to_use) for w in weights])

    plt.figure(figsize=(8, 4))
    plt.plot(weights, costs, marker="o")
    plt.title("Cost as a Function of Weight")
    plt.xlabel("Weight (w)")
    plt.ylabel("Cost J(w)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def _cost_grid(
    weights: Iterable[float], biases: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrids of weights, biases, and corresponding cost values."""

    config = LinearRegressionConfig.default()
    weight_values = np.asarray(list(weights), dtype=float)
    bias_values = np.asarray(list(biases), dtype=float)
    weight_grid, bias_grid = np.meshgrid(weight_values, bias_values, indexing="xy")

    cost_grid = np.empty_like(weight_grid)
    for i in range(bias_grid.shape[0]):
        for j in range(weight_grid.shape[1]):
            cost_grid[i, j] = mean_squared_cost(
                weight_grid[i, j], config, bias=bias_grid[i, j]
            )

    return weight_grid, bias_grid, cost_grid


def plot_cost_contour(
    weights: Iterable[float], biases: Iterable[float]
) -> None:
    """Plot a filled contour of the cost surface over weight and bias values."""

    weight_grid, bias_grid, cost_grid = _cost_grid(weights, biases)

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(weight_grid, bias_grid, cost_grid, levels=30, cmap="viridis")
    plt.colorbar(contour, label="Cost J(w, b)")
    plt.title("Cost Surface over Weight and Bias")
    plt.xlabel("Weight (w)")
    plt.ylabel("Bias (b)")
    plt.tight_layout()
    plt.show()


def plot_cost_surface(
    weights: Iterable[float], biases: Iterable[float]
) -> None:
    """Render a 3D surface of the cost across weight and bias values."""

    weight_grid, bias_grid, cost_grid = _cost_grid(weights, biases)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        weight_grid,
        bias_grid,
        cost_grid,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )
    fig.colorbar(surface, shrink=0.6, aspect=12, label="Cost J(w, b)")
    ax.set_title("3D Cost Surface")
    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Bias (b)")
    ax.set_zlabel("Cost")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    w_candidates = np.linspace(-4.0, 5.0, 25)
    b_candidates = np.linspace(-1.0, 3.0, 25)
    xs, preds, cost_val = evaluate_weight(weight=0.75, bias=0.2)

    print("x values:", xs)
    print("predicted y values:", preds)
    print("cost for w=0.75:", cost_val)

    plot_cost_curve(w_candidates, bias=0.2)
    plot_cost_contour(w_candidates, b_candidates)
