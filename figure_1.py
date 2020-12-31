from pc.types import Variable, Node, Data, Action
from pc.model import Model
from pc.utils import plot_values


if __name__ == "__main__":
    fig_path = "figures"
    num_iterations = 10000
    prior_value = 0.0
    delta_time = 0.01
    noise = 0.03
    stim_delta_fast = 5.0
    stim_delta_slow = 1.0
    stim_range_fast = [1000.0]
    stim_range_slow = range(5000, 6000)
    lines = [1000.0, 5000.0]

    prior = Node(is_fixed=True, init_value=prior_value)
    intero_mu = Node(dt=delta_time, init_value=prior_value)
    intero_data = Data(noise=noise, init_value=prior_value)

    action = Action()
    free_energy = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    model.add_connection(intero_mu, intero_data, action=action)

    for itr in range(num_iterations):
        if itr in stim_range_fast:
            intero_data.update(stim_delta_fast, skip_history=True)
        if itr in stim_range_slow:
            intero_data.update(intero_data.value + delta_time * stim_delta_slow, skip_history=True)
        intero_data.update(intero_data.value + delta_time * action.value)

        model.update()
        free_energy.update(model.get_free_energy())

    plot_values(
        [intero_mu, intero_data, prior, action, free_energy],
        ["Intero Mu", "Intero Data", "Prior", "Action", "Free Energy"],
        lines=lines,
        fig_path=f"{fig_path}/figure_1.png",
    )
