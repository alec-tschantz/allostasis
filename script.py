from allostatis.types import Variable, Node, Data, Action, InverseFunction
from allostatis.model import Model
from allostatis.utils import plot_values


if __name__ == "__main__":
    num_iterations = 500
    prior_value = 0.0
    delta_time = 0.1
    intero_stim_delta = 0.1
    extero_stim_delta = 1.0
    extero_stim_range = range(100, 300)
    intero_stim_range = range(200, 300)

    prior = Node(is_fixed=True, init_value=prior_value)
    intero_mu = Node()
    proprio_mu = Node()
    extero_mu = Node()

    intero_data = Data()
    proprio_data = Data()
    extero_data = Data()

    proprio_func = InverseFunction()
    extero_func = InverseFunction()

    action = Action()
    free_energy = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    model.add_connection(intero_mu, intero_data)
    model.add_connection(proprio_mu, proprio_data, action=action)
    model.add_connection(extero_mu, extero_data)
    model.add_connection(extero_mu, proprio_mu, func=extero_func)
    model.add_connection(proprio_mu, intero_mu, func=proprio_func)

    for itr in range(num_iterations):
        intero_delta = intero_stim_delta if itr in intero_stim_range else 0.0
        extero_stim = extero_stim_delta if itr in extero_stim_range else 0.0
        intero_data.update(intero_data.value + delta_time * (intero_delta))
        extero_data.update(extero_stim)

        intero_data.update(intero_data.value + delta_time * action.value)
        proprio_data.update(action.value)

        model.update()
        free_energy.update(model.get_free_energy())

    plot_values(
        [intero_data, extero_data, proprio_data, intero_mu, extero_mu, proprio_mu, action],
        [
            "Intero Data",
            "Extero Data",
            "Proprio Data",
            "Intero Mu",
            "Extero Mu",
            "Proprio Mu",
            "Action",
        ],
    )
