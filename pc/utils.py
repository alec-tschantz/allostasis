from typing import List
import matplotlib.pyplot as plt

from pc.types import Variable


def plot_values(
    values: List[Variable],
    names: List[str],
    lims,
    lines: List[float] = None,
    fig_path: str = None,
    shape=None,
    figsize=(12, 8),
):
    num_plots = len(values)

    if shape is None:
        _, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3))
    else:
        _, axes = plt.subplots(*shape, figsize=figsize)

    axes = axes.flatten()
    for i in range(num_plots):
        axes[i].plot(values[i].history, color="red")
        axes[i].set_title(names[i], fontsize=14)
        axes[i].set_ylim(lims[i])
        if lines is not None:
            [axes[i].axvline(line, linestyle="--", color="gray", zorder=0) for line in lines]
        [spine.set_linewidth(1.3) for spine in axes[i].spines.values()]

    plt.tight_layout()
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(fig_path, dpi=300)


def plot_values_b(
    values: List[Variable],
    values_b: List[Variable],
    names: List[str],
    lims,
    lines: List[float] = None,
    fig_path: str = None,
    shape=None,
    figsize=(12, 8),
):
    num_plots = len(values)
    if shape is None:
        _, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3))
    else:
        _, axes = plt.subplots(*shape, figsize=figsize)
    
    axes = axes.flatten()
    for i in range(num_plots):
        if i == 0:
            axes[i].plot(values[i].history, color="red", label="Normal precision")
            axes[i].plot(values_b[i].history, color="green", label="Low precision")
            axes[i].legend()
        else:
            axes[i].plot(values[i].history, color="red")
            axes[i].plot(values_b[i].history, color="green")
        axes[i].set_title(names[i], fontsize=14)
        axes[i].set_ylim(lims[i])
        if lines is not None:
            if i == 0:
                lines = lines[1:]
            [axes[i].axvline(line, linestyle="--", color="gray", zorder=0) for line in lines]
        [spine.set_linewidth(1.3) for spine in axes[i].spines.values()]

    plt.tight_layout()
    
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(fig_path, dpi=300)
