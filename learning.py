from allostatis.types import LinearFunction, Variable, Node, Data, Param, InverseFunction
from allostatis.model import Model
from allostatis.utils import plot_values


if __name__ == "__main__":
    num_iterations = 100
    prior_value = 10.0
    data_value = 10.0
    param_value = 0.1
    l_rate = 0.01
    delta_time = 0.1
    stim_delta = 1.0
    stim_range = range(20, 30)

    prior = Node(is_fixed=True, init_value=prior_value)
    mu = Node()
    data = Data()
    param = Param(init_value=param_value, l_rate=l_rate)
    func = LinearFunction(param)
    free_energy = Variable()

    model = Model()
    model.add_connection(prior, mu)
    model.add_connection(mu, data, func=func)

    for itr in range(num_iterations):
        data.update(data_value)
        model.update()
        free_energy.update(model.get_free_energy())
    
    plot_values(
        [mu, data, param, free_energy],
        ["Mu", "Data", "Param", "Free Energy"],
    )
