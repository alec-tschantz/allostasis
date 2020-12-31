from typing import List
import matplotlib.pyplot as plt

from pc.types import Variable


def plot_values(values: List[Variable], names: List[str], lines: List[float] = None, fig_path: str = None):
    num_plots = len(values)
    _, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3))
    for i in range(num_plots):
        axes[i].plot(values[i].history, color="red")
        axes[i].set_title(names[i], fontsize=14)
        if lines is not None:
            [axes[i].axvline(line, linestyle="--", color="gray", zorder=100) for line in lines]
        [spine.set_linewidth(1.3) for spine in axes[i].spines.values()]
        
    plt.tight_layout()
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(fig_path)

def plot_values_b(values: List[Variable], values_b: List[Variable], names: List[str], lines: List[float] = None, fig_path: str = None):
    num_plots = len(values)
    _, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3))
    for i in range(num_plots):
        axes[i].plot(values[i].history, color="red")
        axes[i].plot(values_b[i].history, color="blue")
        axes[i].set_title(names[i], fontsize=14)
        if lines is not None:
            [axes[i].axvline(line, linestyle="--", color="gray", zorder=100) for line in lines]
        [spine.set_linewidth(1.3) for spine in axes[i].spines.values()]
        
    plt.tight_layout()
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(fig_path)
