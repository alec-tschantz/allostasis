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

    err_stats = {"err_thresh": 3.0, "action_len": 4000, "num_action_steps": 0}

    prior = Node(is_fixed=True, init_value=prior_value)
    intero_mu = Node(dt=delta_time, init_value=prior_value)
    intero_data = Data(noise=noise, init_value=prior_value)

    action = Action()
    free_energy = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    uuid = model.add_connection(intero_mu, intero_data)
    err = model.get_error(uuid)

    perform_action = False
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
        free_energy.update(model.get_free_energy())

    plot_values(
        [intero_data, intero_mu, prior, action, free_energy],
        ["Intero Data", "Intero Mu", "Prior", "Action", "Free Energy"],
        lines=lines,
        fig_path=f"{fig_path}/figure_1b.png",
    )
