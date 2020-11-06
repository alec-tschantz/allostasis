"""
@author: Alexander Tschantz
"""

from typing import Optional
import numpy as np

from allostatis.types import Variable, Node, Error, Data, LinearFunction
from allostatis.functions import update_node, update_error, update_param, calc_free_energy
from allostatis.utils import plot_values


def update_intero_data(
    data: Data, stimulus: Optional[float] = None, noise_std: float = 0.1
) -> Data:
    data.update(stimulus + np.random.normal(0.0, noise_std))
    return data


if __name__ == "__main__":
    num_iterations = 1000
    stim_value = 10.0
    prior_value = 10.0
    param_value = 0.1
    delta_time = 0.1
    learning_rate = 0.01
    noise_std = 0.01

    prior = Node(init_value=prior_value)
    mu = Node()
    function = LinearFunction(param=param_value)
    data = Data()
    prior_err = Error()
    data_err = Error()
    free_energy = Variable()

    for itr in range(num_iterations):
        data = update_intero_data(data, stimulus=stim_value, noise_std=noise_std)
        data_err = update_error(data_err, mu, data, function=function)
        prior_err = update_error(prior_err, prior, mu)

        mu = update_node(
            mu, data_errs=[data_err], functions=[function], prior_errs=[prior_err], dt=delta_time
        )

        function = update_param(function, mu, data_err, lr=learning_rate)
        if itr % 100 == 0:
            print(function.param)

        free_energy.update(calc_free_energy([data_err, prior_err]))

    plot_values(
        [mu, data_err, prior_err, free_energy], ["Mu", "Data Err", "Prior Err", "Free energy"]
    )
