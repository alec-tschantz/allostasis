"""
@author: Alexander Tschantz
"""

from typing import Optional
import numpy as np

from allostatis.types import (
    IdentityFunction,
    Variable,
    Node,
    Error,
    Data,
    Action,
    InverseFunction,
    IdentityFunction,
)
from allostatis.functions import update_node, update_error, update_action, calc_free_energy
from allostatis.utils import plot_values


def update_intero_data(
    data: Data,
    time: int,
    action: Optional[Action] = None,
    pulse_time: int = 300,
    noise_std: float = 0.1,
    pulse_value: float = 10.0,
    dt: float = 0.01,
) -> Data:
    """ Update interoceptive data

    Args:
        data (Data): [current data]
        time (int): [global time]
        action (Action, optional): [action]. Defaults to None.
        pulse_time (int, optional): [time at which to apply pulse]. Defaults to 3.
        noise_std (float, optional): [standard deviation of noise]. Defaults to 0.1.
        pulse_value (float, optional): [magnitude of pulse]. Defaults to 10.0.
        dt (float, optional): [delta time]. Defaults to 0.01.

    Returns:
        Data: [updated data]
    """
    delta = np.random.normal(0.0, noise_std)

    if time == pulse_time:
        delta = delta + pulse_value

    if action is not None:
        delta = delta + action.value

    data.update(data.value + dt * (delta))
    return data


def update_proprio_data(data: Data, action: Action, noise_std: float = 0.1) -> Data:
    """ Update proprioceptive data

    Args:
        data (Data): [current data]
        action (Action): [proprioceptive action]
        noise_std (float, optional): [standard deviation of noise]. Defaults to 0.1.

    Returns:
        Data: [updated data]
    """
    value = action.value + np.random.normal(0.0, noise_std)
    data.update(value)
    return data


if __name__ == "__main__":
    num_iterations = 1000
    reflex_param = 1.0
    delta_time = 0.1
    noise_std = 0.01

    prior = Node()
    intero_mu = Node()
    proprio_mu = Node()

    reflex_function = InverseFunction(param=reflex_param)
    proprio_function = IdentityFunction()

    intero_data = Data()
    proprio_data = Data()

    prior_err = Error()
    intero_err = Error()
    proprio_err = Error()
    reflex_err = Error()

    action = Action()

    free_energy = Variable()

    for itr in range(num_iterations):
        intero_data = update_intero_data(
            intero_data, itr, action=action, noise_std=noise_std, dt=delta_time
        )
        proprio_data = update_proprio_data(proprio_data, action, noise_std=noise_std)

        intero_err = update_error(intero_err, intero_mu, intero_data)
        proprio_err = update_error(proprio_err, proprio_mu, proprio_data)
        prior_err = update_error(prior_err, prior, intero_mu)
        reflex_err = update_error(reflex_err, proprio_mu, intero_mu, function=reflex_function)

        intero_mu = update_node(
            intero_mu, data_errs=[intero_err], prior_errs=[prior_err, reflex_err], dt=delta_time
        )
        proprio_mu = update_node(
            proprio_mu,
            data_errs=[reflex_err, proprio_err],
            functions=[reflex_function, proprio_function],
            dt=delta_time,
        )

        action = update_action(action, proprio_err, dt=delta_time)

        free_energy.update(calc_free_energy([intero_err, prior_err, reflex_err]))

    plot_values([intero_mu, proprio_mu, free_energy], ["Intero Mu", "Proprio Mu", "Free energy"])
    plot_values([intero_data, proprio_data, action], ["Intero Data", "Proprio Data", "Action"])
    plot_values(
        [intero_err, proprio_err, reflex_err, prior_err],
        ["Intero Error", "Proprio Error", "Reflex Error", "Prior Error"],
    )
