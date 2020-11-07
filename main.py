from allostatis.types import Variable, Node, Data, Action, InverseFunction
from allostatis.model import Model
from allostatis.utils import plot_values


if __name__ == "__main__":
    num_iterations = 100
    prior_value = 0.0
    delta_time = 0.1
    stim_delta = 1.0
    stim_range = range(20, 30)

    prior = Node(is_fixed=True, init_value=prior_value)
    intero_mu = Node()
    proprio_mu = Node()

    intero_data = Data()
    proprio_data = Data()

    reflex_func = InverseFunction()
    action = Action()
    free_energy = Variable()

    model = Model()
    model.add_connection(prior, intero_mu)
    model.add_connection(intero_mu, intero_data)
    model.add_connection(proprio_mu, proprio_data, action=action)
    model.add_connection(proprio_mu, intero_mu, func=reflex_func)

    for itr in range(num_iterations):
        if itr in stim_range:
            intero_data.update(intero_data.value + delta_time * (stim_delta))
        intero_data.update(intero_data.value + delta_time * action.value)
        proprio_data.update(action.value)

        model.update()
        free_energy.update(model.get_free_energy())
    
    plot_values(
        [intero_data, proprio_data, intero_mu, proprio_mu, action, free_energy],
        ["Intero Data", "Proprio Data", "Intero Mu", "Proprio Mu", "Action", "Free Energy"],
    )
