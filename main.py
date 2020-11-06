from allostatis.types import Node, Error, Data, LinearFunction
from allostatis.functions import update_node, update_error
from allostatis.utils import plot_values

if __name__ == "__main__":
    num_iterations = 1000
    data_value = 4.0
    prior_value = 20.0

    mu = Node()
    prior = Node(init_value=prior_value)
    data = Data(init_value=data_value)

    function = LinearFunction()

    data_err = Error()
    prior_err = Error()

    for itr in range(num_iterations):
        data_err = update_error(data_err, mu, data, function=function)
        prior_err = update_error(prior_err, prior, mu)
        mu = update_node(mu, data_errs=[data_err], functions=[function], prior_errs=[prior_err])

    plot_values([mu, data_err, prior_err], ["Mu", "Data Error", "Prior Error"])
