from typing import List
import matplotlib.pyplot as plt 

from allostatis.types import Value

def plot_values(values: List[Value], names: List[str]):
    num_plots = len(values)
    _, axes = plt.subplots(1, num_plots)
    for i in range(num_plots):
        axes[i].plot(values[i].history)
        axes[i].set_title(names[i])
    plt.show()