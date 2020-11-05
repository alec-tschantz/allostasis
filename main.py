from allostatis.types import Node, Error, Data, LinearFunction
from allostatis.functions import update_node, update_error
from allostatis.utils import plot_values

if __name__ == "__main__":
    num_iterations = 1000
    data_value = 4.0
    prior_value = 10.0

    mu = Node()
    prior = Node(init_value=prior_value)
    data = Data(init_value=data_value)

    function = LinearFunction()

    likelihood_error = Error()
    prior_error = Error()

    for itr in range(num_iterations):
        likelihood_error = update_error(likelihood_error, mu, data, function=function)
        prior_error = update_error(prior_error, prior, mu)
        mu = update_node(
            mu, likelihood_errors=[likelihood_error], prior_errors=[prior_error], functions=[function]
        )

    plot_values([mu, likelihood_error, prior_error], ["Mu", "Likelihood Error", "Prior Error"])
