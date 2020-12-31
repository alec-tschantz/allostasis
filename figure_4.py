from pc.types import Variable, Node, Data, Action, Param, InverseFunction
from pc.model import Model
from pc.utils import plot_values_b


def run_a():
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
    uuid = model.add_connection(intero_mu, intero_data, action=action)
    err = model.get_error(uuid)
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
        fe = min(5, model.get_free_energy())
        free_energy.update(fe)

    return intero_mu, intero_data, prior, extero_mu, extero_data, action, free_energy


def run_b():
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
        fe = min(5, model.get_free_energy())
        free_energy.update(fe)

    return intero_mu, intero_data, prior, extero_mu, extero_data, action, free_energy


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

    intero_mu, intero_data, prior, extero_mu, extero_data, action, free_energy = run_a()
    intero_mu_b, intero_data_b, prior_b, extero_mu_b, extero_data_b, action_b, free_energy_b = (
        run_b()
    )

    plot_values_b(
        [intero_mu, intero_data, prior, extero_mu, extero_data, action, free_energy],
        [intero_mu_b, intero_data_b, prior_b, extero_mu_b, extero_data_b, action_b, free_energy_b],
        ["Intero Mu", "Intero Data", "Prior", "Extero Mu", "Extero Data", "Action", "Free Energy"],
        lines=lines,
        fig_path=f"{fig_path}/figure_4.png",
    )

