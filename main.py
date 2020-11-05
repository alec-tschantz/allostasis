from allostatis.types import Node, Error, Data
from allostatis.functions import update_node, update_error
from allostatis.utils import plot_values

if __name__ == "__main__":
    dt = 0.01
    num_iterations = 1000
    data_value = 4.0

    mu = Node()
    error = Error()
    data = Data()

    data.set_value(data_value)

    for itr in range(num_iterations):
        data.set_value(data_value)
        error = update_error(error, mu, data)
        node = update_node(mu, likelihood_errors=[error], dt=dt)

    plot_values([mu, error], ["Mu", "Error"])