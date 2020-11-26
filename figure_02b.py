from pc.types import Variable, Node, Data, Action, Param, InverseFunction
from pc.model import Model
from pc.utils import plot_values

prior_value = 0.0
num_iterations = 4000
delta_time = 0.01
intero_stim_value = 1.0
extero_stim_value = 1.0
extero_stim_range = range(500, 3000)
intero_stim_range = [1700]
lines = [500.0, 1700.0, 3000.0]


def run_1():
    prior = Node(init_value=prior_value)
    intero_mu = Node(dt=delta_time, init_value=prior_value)
    intero_data = Data(init_value=prior_value)
    extero_mu = Node(dt=delta_time)
    action = Action(dt=0.0001)
    
    intero_data = Data()
    extero_data = Data()

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
            intero_data.update(intero_data.value + intero_stim_value)
        else:
            intero_data.update(intero_data.value)

        if itr in extero_stim_range:
            extero_data.update(extero_stim_value)
        else:
            extero_data.update(0.0)

        intero_data.update(intero_data.value + delta_time * action.value, stop_history=True)
        action.update(-err.value, stop_history=True)

        model.update(skip_action=True)
        free_energy.update(model.get_free_energy())

    plot_values(
        [intero_data, intero_mu, action, free_energy],
        ["Intero Data", "Intero Mu", "Action", "Free Energy"],
        lines=lines,
        fig_path="figures/figure_02_ab.png"
    )


def run_2():
    prior = Node(is_fixed=True, init_value=prior_value)
    intero_mu = Node(dt=delta_time, init_value=prior_value)
    intero_data = Data(init_value=prior_value)
    extero_mu = Node(dt=delta_time)
    action = Action(dt=0.0001)
    
    intero_data = Data()
    extero_data = Data()

    extero_param = Param(is_fixed=True, init_value=1.0)
    extero_func = InverseFunction(param=extero_param)

    free_energy = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    uuid = model.add_connection(intero_mu, intero_data, action=action)
    err = model.get_error(uuid)
    # model.add_connection(extero_mu, extero_data)
    # model.add_connection(extero_mu, prior, func=extero_func)

    for itr in range(num_iterations):
        if itr in intero_stim_range:
            intero_data.update(intero_data.value + intero_stim_value, stop_history=True)
        else:
            intero_data.update(intero_data.value, stop_history=True)

        if itr in extero_stim_range:
            extero_data.update(extero_stim_value)
        else:
            extero_data.update(0.0)

        intero_data.update(intero_data.value + delta_time * action.value)
        action.update(-err.value, stop_history=True)

        model.update(skip_action=True)
        free_energy.update(model.get_free_energy())

    plot_values(
        [intero_data, intero_mu, action, free_energy],
        ["Intero Data", "Intero Mu", "Action", "Free Energy"],
        lines=lines,
        fig_path="figures/figure_02_abb.png"
    )


if __name__ == "__main__":
    run_1()
    run_2()
