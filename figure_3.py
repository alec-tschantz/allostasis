from pc.types import Variable, Node, Data, Action, Param, InverseFunction
from pc.model import Model
from pc.utils import plot_values


if __name__ == "__main__":

    fig_path = "figures"
    num_iterations = 10000
    prior_value = 0.0
    delta_time = 0.01
    intero_stim_value = 5.0
    extero_stim_value = 5.0
    noise = 0.03
    extero_stim_range = range(1000, 9000)
    intero_stim_range = [5000]
    lines = [1000.0, 5000.0, 9000.0]

    prior = Node(init_value=prior_value)
    intero_mu = Node(dt=delta_time, init_value=prior_value)
    intero_data = Data(init_value=prior_value)
    extero_mu = Node(dt=delta_time)
    action = Action()

    intero_data = Data(noise=noise)
    extero_data = Data(noise=noise)

    extero_param = Param(is_fixed=True, init_value=1.0)
    extero_func = InverseFunction(param=extero_param)
    free_energy = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    model.add_connection(intero_mu, intero_data, action=action, variance=100)
    model.add_connection(extero_mu, extero_data)
    model.add_connection(extero_mu, prior, func=extero_func)

    for itr in range(num_iterations):
        if itr in intero_stim_range:
            intero_data.update(intero_data.value + intero_stim_value, skip_history=True)

        if itr in extero_stim_range:
            extero_data.update(extero_stim_value)
        else:
            extero_data.update(0.0)

        intero_data.update(intero_data.value + delta_time * action.value)

        model.update()
        free_energy.update(model.get_free_energy())

    plot_values(
        [intero_mu, intero_data, prior, extero_mu, extero_data, action, free_energy],
        ["Intero Mu", "Intero Data", "Prior", "Extero Mu", "Extero Data", "Action", "Free Energy"],
        lines=lines,
        fig_path=f"{fig_path}/figure_3.png",
    )

