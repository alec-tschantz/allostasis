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
    model.add_connection(intero_mu, intero_data, action=action)
    model.add_connection(extero_mu, extero_data)
    model.add_connection(extero_mu, prior, func=extero_func)

    for itr in range(num_iterations):
        if itr in intero_stim_range:
            intero_data.update(intero_data.value + intero_stim_value, skip_history=True)
        intero_data.update(intero_data.value + delta_time * action.value)

        if itr in extero_stim_range:
            extero_data.update(extero_stim_value)
        else:
            extero_data.update(0.0)

        model.update()
        free_energy.update(model.get_free_energy())


    # Valence

    noise = 0.0

    _prior = Node(init_value=prior_value)
    _intero_mu = Node(dt=delta_time, init_value=prior_value)
    _intero_data = Data(init_value=prior_value)
    _extero_mu = Node(dt=delta_time)
    _action = Action()

    _intero_data = Data(noise=noise)
    _extero_data = Data(noise=noise)

    extero_param = Param(is_fixed=True, init_value=1.0)
    extero_func = InverseFunction(param=extero_param)

    # free_energy = Variable()

    model = Model()
    model.add_connection(_prior, _intero_mu)
    model.add_connection(_intero_mu, _intero_data, action=_action)
    model.add_connection(_extero_mu, _extero_data)
    model.add_connection(_extero_mu, _prior, func=extero_func)

    prev_fe = 0
    valence_val = 0
    valence = Variable()
    for itr in range(num_iterations):
        if itr in intero_stim_range:
            _intero_data.update(_intero_data.value + intero_stim_value, skip_history=True)
        _intero_data.update(_intero_data.value + delta_time * _action.value)

        if itr in extero_stim_range:
            _extero_data.update(extero_stim_value)
        else:
            _extero_data.update(0.0)

        model.update()
        fe = model.get_free_energy()
        if itr % 20 == 0:
            valence_val = max(-1, min(1, prev_fe - fe))
            prev_fe = fe

        valence.update(valence_val)

    lims = [(-6, 7), (-6, 7), (-6, 7), (-6, 7), (-6, 1), (-6, 7), (-1, 15), (-1, 1)]
    plot_values(
        [intero_data, extero_data, intero_mu, extero_mu, action, prior, free_energy, valence],
        ["Intero Data", "Extero Data", "Mu Intero", "Mu Extero", "Action", "Mu Prior", "Free Energy", "Valence"],
        lims,
        lines=lines,
        fig_path=f"{fig_path}/figure_2.png",
        shape=(4, 2),
        figsize=(12, 10),
    )

