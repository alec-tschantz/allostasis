from typing import List
import matplotlib.pyplot as plt

from pc.types import Variable


def plot_values(values: List[Variable], names: List[str], lines: List[float] = None, fig_path: str = None):
    num_plots = len(values)
    fig_width = min(12, num_plots * 3)
    _, axes = plt.subplots(1, num_plots, figsize=(fig_width, 3))
    for i in range(num_plots):
        axes[i].plot(values[i].history, color="red")
        axes[i].set_title(names[i])
        if lines is not None:
            [axes[i].axvline(line, linestyle="--", color="gray", zorder=100) for line in lines]
        [spine.set_linewidth(1.2) for spine in axes[i].spines.values()]
        
    plt.tight_layout()
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(fig_path)
