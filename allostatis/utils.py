from typing import List
import matplotlib.pyplot as plt 

from allostatis.types import Value

def plot_values(values: List[Value], names: List[str]):
    num_plots = len(values)
    fig_width = max(12, num_plots * 3)
    _, axes = plt.subplots(1, num_plots, figsize=(fig_width, 3))
    for i in range(num_plots):
        axes[i].plot(values[i].history)
        axes[i].set_title(names[i])
    plt.show()