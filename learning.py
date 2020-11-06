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
from allostatis.functions import (
    update_node,
    update_error,
    update_action,
    update_param,
    calc_free_energy,
)
from allostatis.utils import plot_values


def update_intero_data(
    data: Data,
    stimulus: Optional[float] = None,
    action: Optional[Action] = None,
    noise_std: float = 0.1,
    dt: float = 0.1,
) -> Data:
    delta = np.random.normal(0.0, noise_std)
    if stimulus is not None:
        delta = delta + stimulus
    if action is not None:
        delta = delta + action.value
    data.update(data.value + dt * (delta))
    return data


def update_extero_data(
    data: Data, stimulus: Optional[float] = None, noise_std: float = 0.1, dt: float = 0.1
) -> Data:
    delta = np.random.normal(0.0, noise_std)
    if stimulus is not None:
        delta = delta + stimulus
    data.update(data.value + dt * (delta))
    return data


def update_proprio_data(data: Data, action: Action, noise_std: float = 0.1) -> Data:
    data.update(action.value + np.random.normal(0.0, noise_std))
    return data


if __name__ == "__main__":
    num_iterations = 1000
    reflex_param = 1.0
    reflex_extero_param = 0.1
    delta_time = 0.1
    learning_rate = 0.1
    noise_std = 0.01
    intero_stim_mag = 10
    intero_stim_time = 0
    extero_stim_mag = 0
    extero_stim_time = 0

    prior = Node()
    intero_mu = Node()
    proprio_mu = Node()
    extero_mu = Node()

    intero_function = IdentityFunction()
    proprio_function = IdentityFunction()
    extero_function = IdentityFunction()
    reflex_function = InverseFunction(param=reflex_param)
    reflex_extero_function = InverseFunction(param=reflex_extero_param)

    intero_data = Data()
    proprio_data = Data()
    extero_data = Data()

    prior_err = Error()
    intero_err = Error()
    proprio_err = Error()
    extero_err = Error()
    reflex_err = Error()
    reflex_extero_err = Error()

    action = Action()

    free_energy = Variable()

    for itr in range(num_iterations):
        intero_stim = intero_stim_mag if itr == intero_stim_time else None
        intero_data = update_intero_data(
            intero_data, stimulus=intero_stim, action=action, noise_std=noise_std, dt=delta_time
        )
        proprio_data = update_proprio_data(proprio_data, action, noise_std=noise_std)
        extero_stim = extero_stim_mag if itr == extero_stim_time else None
        extero_data = update_intero_data(
            extero_data, stimulus=extero_stim, noise_std=noise_std, dt=delta_time
        )

        intero_err = update_error(intero_err, intero_mu, intero_data)
        proprio_err = update_error(proprio_err, proprio_mu, proprio_data)
        prior_err = update_error(prior_err, prior, intero_mu)
        extero_err = update_error(extero_err, extero_mu, extero_data)
        reflex_err = update_error(reflex_err, proprio_mu, intero_mu, function=reflex_function)
        reflex_extero_err = update_error(
            reflex_extero_err, extero_mu, proprio_mu, function=reflex_function
        )

        intero_mu = update_node(
            intero_mu,
            data_errs=[intero_err],
            functions=[intero_function],
            prior_errs=[prior_err, reflex_err],
            dt=delta_time,
        )
        proprio_mu = update_node(
            proprio_mu,
            data_errs=[reflex_err, proprio_err],
            functions=[reflex_function, proprio_function],
            prior_errs=[reflex_extero_err],
            dt=delta_time,
        )
        extero_mu = update_node(
            extero_mu,
            data_errs=[reflex_extero_err, extero_err],
            functions=[reflex_function, extero_function],
            dt=delta_time,
        )

        reflex_extero_function = update_param(
            reflex_extero_function, extero_mu, reflex_extero_err, lr=learning_rate
        )
        print(reflex_extero_function.param)

        action = update_action(action, proprio_err, dt=delta_time)
        free_energy.update(calc_free_energy([intero_err, prior_err, reflex_err]))

    plot_values(
        [intero_mu, proprio_mu, extero_mu, free_energy],
        ["Intero Mu", "Proprio Mu", "Extero Mu", "Free Energy"],
    )
    plot_values(
        [intero_data, proprio_data, extero_data, action],
        ["Intero Data", "Proprio Data", "Extero Data", "Action"],
    )
    plot_values(
        [intero_err, proprio_err, reflex_err, reflex_extero_err],
        ["Intero Error", "Proprio Error", "Reflex Error", "Reflex Extero Error"],
    )
