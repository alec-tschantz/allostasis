from typing import List
import matplotlib.pyplot as plt

from allostatis.types import Variable


def plot_values(values: List[Variable], names: List[str]):
    num_plots = len(values)
    fig_width = min(12, num_plots * 3)
    _, axes = plt.subplots(1, num_plots, figsize=(fig_width, 3))
    for i in range(num_plots):
        axes[i].plot(values[i].history, color="red")
        axes[i].set_title(names[i])
        [spine.set_linewidth(1.2) for spine in axes[i].spines.values()]
    plt.tight_layout()
    plt.show()
