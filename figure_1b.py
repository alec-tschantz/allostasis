import numpy as np 

from pc.types import Variable, Node, Data, Action
from pc.model import Model
from pc.utils import plot_values

def can_perform_action(err, err_stats, perform_action):
    if perform_action is False:
        if err > err_stats["err_thresh"]:
            err_stats["num_action_steps"] = 0
            perform_action = True

    if perform_action is True:
        if err_stats["num_action_steps"] > err_stats["action_len"]:
            perform_action = False
        err_stats["num_action_steps"] += 1

    return perform_action, err_stats


if __name__ == "__main__":
    fig_path = "figures"
    num_iterations = 10000
    prior_value = 0.0
    delta_time = 0.01
    noise = 0.03
    stim_delta_fast = 5.0
    stim_delta_slow = 0.5
    stim_range_fast = [1000.0]
    stim_range_slow = range(5000, 7000)
    lines = [1000.0, 5000.0]

    err_stats = {"err_thresh": 3.0, "action_len": 1000, "num_action_steps": 0}

    prior = Node(is_fixed=True, init_value=prior_value)
    intero_mu = Node(dt=delta_time, init_value=prior_value)
    intero_data = Data(noise=noise, init_value=prior_value)

    action = Action()
    free_energy = Variable()
    valence = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    uuid = model.add_connection(intero_mu, intero_data)
    err = model.get_error(uuid)

    perform_action = False
    prev_fe = 0
    valence_val = 0
    for itr in range(num_iterations):
        if itr in stim_range_fast:
            intero_data.update(stim_delta_fast, skip_history=True)
        if itr in stim_range_slow:
            intero_data.update(intero_data.value + delta_time * stim_delta_slow, skip_history=True)

        perform_action, err_stats = can_perform_action(err.value, err_stats, perform_action)
        if perform_action:
            action.update(-err.value)
            intero_data.update(intero_data.value + delta_time * action.value, skip_history=True)
        else:
            action.update(np.random.normal(0, noise))

        intero_data.update(intero_data.value)

        model.update()
        fe = model.get_free_energy()
        free_energy.update(prev_fe)
        if itr % 10 == 0:
            valence_val = max(-1, min(1, prev_fe - fe))
            prev_fe = fe

        valence.update(valence_val)

    # VALENCE

    fig_path = "figures"
    num_iterations = 10000
    prior_value = 0.0
    delta_time = 0.01
    noise = 0.00
    stim_delta_fast = 5.0
    stim_delta_slow = 0.5
    stim_range_fast = [1000.0]
    stim_range_slow = range(5000, 7000)
    lines = [1000.0, 5000.0]

    err_stats = {"err_thresh": 3.0, "action_len": 1000, "num_action_steps": 0}

    _prior = Node(is_fixed=True, init_value=prior_value)
    _intero_mu = Node(dt=delta_time, init_value=prior_value)
    _intero_data = Data(noise=noise, init_value=prior_value)

    _action = Action()
    valence = Variable()

    model = Model()
    model.add_connection(_prior, _intero_mu)
    uuid = model.add_connection(_intero_mu, _intero_data)
    _err = model.get_error(uuid)

    perform_action = False
    prev_fe = 0
    valence_val = 0
    for itr in range(num_iterations):
        if itr in stim_range_fast:
            _intero_data.update(stim_delta_fast, skip_history=True)
        if itr in stim_range_slow:
            _intero_data.update(_intero_data.value + delta_time * stim_delta_slow, skip_history=True)

        perform_action, err_stats = can_perform_action(_err.value, err_stats, perform_action)
        if perform_action:
            _action.update(-_err.value)
            _intero_data.update(intero_data.value + delta_time * _action.value, skip_history=True)
        else:
            _action.update(np.random.normal(0, noise))

        _intero_data.update(_intero_data.value)

        model.update()
        fe = model.get_free_energy()
        if itr % 20 == 0:
            valence_val = max(-1, min(1, prev_fe - fe))
            prev_fe = fe

        valence.update(valence_val)

    
    lims = [(-6, 7), (-6, 7), (-6, 7), (-6, 1), (-1, 15), (-1, 1)]
    plot_values(
        [intero_data, prior, intero_mu, action, free_energy, valence],
        ["Intero Data", "Mu Prior", "Mu Intero", "Action", "Free Energy", "Valence"],
        lims,
        lines=lines,
        fig_path=f"{fig_path}/figure_1b.png",
        shape=(3,2)
    )
