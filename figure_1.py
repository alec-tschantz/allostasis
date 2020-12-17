from pc.types import Variable, Node, Data, Action
from pc.model import Model
from pc.utils import plot_values


if __name__ == "__main__":
    fig_path = "figures"
    num_iterations = 100
    prior_value = 4.0
    delta_time = 0.1
    stim_delta = 0.0
    noise = 0.0
    stim_range = [20.0]

    prior = Node(is_fixed=True, init_value=prior_value)
    intero_mu = Node(dt=delta_time, init_value=prior_value)
    intero_data = Data(noise=noise, init_value=prior_value)

    action = Action(dt=0.0001)
    free_energy = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    uuid = model.add_connection(intero_mu, intero_data, action=action)
    err = model.get_error(uuid)

    for itr in range(num_iterations):
        if itr in stim_range:
            intero_data.update(stim_delta)

        intero_data.update(intero_data.value + delta_time * action.value)
        action.update(-err.value)

        model.update(skip_action=True)
        free_energy.update(model.get_free_energy())

    plot_values(
        [intero_data, intero_mu, action, free_energy],
        ["Intero Data", "Intero Mu", "Action", "Free Energy"],
        lines=stim_range,
        fig_path=f"{fig_path}/figure_1.png",
    )
