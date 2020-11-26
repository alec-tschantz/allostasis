from pc.types import Variable, Node, Data, Action, Param, InverseFunction
from pc.model import Model
from pc.utils import plot_values

num_iterations = 4000
delta_time = 0.01
intero_stim_impulse = 1.0
extero_stim_value = 1.0
extero_stim_range = range(1000, 3000)
intero_stim_range = [1700]
lines = [1000.0, 1700.0, 3000.0]


def run_1():
    intero_mu = Node(dt=delta_time)
    proprio_mu = Node(dt=delta_time)
    extero_mu = Node(dt=delta_time)

    intero_data = Data()
    proprio_data = Data()
    extero_data = Data()

    proprio_func = InverseFunction()
    extero_param = Param(is_fixed=True, init_value=1.0)
    extero_func = InverseFunction(param=extero_param)

    action = Action(dt=delta_time)
    free_energy = Variable()
    total_action = Variable()

    model = Model()
    model.add_connection(intero_mu, intero_data)
    model.add_connection(proprio_mu, proprio_data, action=action)
    model.add_connection(extero_mu, extero_data)
    model.add_connection(extero_mu, proprio_mu, func=extero_func)
    model.add_connection(proprio_mu, intero_mu, func=proprio_func)

    for itr in range(num_iterations):
        if itr in intero_stim_range:
            intero_data.update(intero_data.value + intero_stim_impulse, store_history=False)
        else:
            intero_data.update(intero_data.value, store_history=False)

        if itr in extero_stim_range:
            extero_data.update(extero_stim_value)
        else:
            extero_data.update(0.0)

        intero_data.update(intero_data.value + delta_time * action.value)
        proprio_data.update(action.value)

        model.update()
        free_energy.update(model.get_free_energy())
        total_action.update(total_action.value + abs(action.value))

    plot_values(
        [intero_data, intero_mu, action, free_energy],
        ["Intero Data", "Intero Mu", "Action", "Free Energy"],
        lines=lines,
        fig_path="figures/figure_02_a.png"
    )


def run_2():
    intero_mu = Node(dt=delta_time)
    proprio_mu = Node(dt=delta_time)

    intero_data = Data()
    proprio_data = Data()

    proprio_func = InverseFunction()

    action = Action(dt=delta_time)
    free_energy = Variable()
    total_action = Variable()

    model = Model()
    model.add_connection(intero_mu, intero_data)
    model.add_connection(proprio_mu, proprio_data, action=action)
    model.add_connection(proprio_mu, intero_mu, func=proprio_func)

    for itr in range(num_iterations):
        if itr in intero_stim_range:
            intero_data.update(intero_data.value + intero_stim_impulse, store_history=False)
        else:
            intero_data.update(intero_data.value, store_history=False)

        intero_data.update(intero_data.value + delta_time * action.value)
        proprio_data.update(action.value)

        model.update()
        free_energy.update(model.get_free_energy())
        total_action.update(total_action.value + abs(action.value))

    plot_values(
        [intero_data, intero_mu, action, free_energy],
        ["Intero Data", "Intero Mu", "Action", "Free Energy"],
        lines=lines,
        fig_path="figures/figure_02_b.png"
    )


if __name__ == "__main__":
    run_1()
    run_2()
