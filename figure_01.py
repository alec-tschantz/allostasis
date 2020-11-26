from pc.types import Variable, Node, Data, Action, InverseFunction
from pc.model import Model
from pc.utils import plot_values


if __name__ == "__main__":
    num_iterations = 100
    delta_time = 0.1
    stim_delta = 1.0
    noise = 0.0
    stim_range = [20.0]

    intero_mu = Node(dt=delta_time)
    proprio_mu = Node(dt=delta_time)

    intero_data = Data(noise=noise)
    proprio_data = Data(noise=noise)
 
    reflex_func = InverseFunction()
    action = Action(dt=delta_time)
    free_energy = Variable()

    model = Model()
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
        [intero_data, intero_mu, action, free_energy],
        ["Intero Data", "Intero Mu", "Action", "Free Energy"],
        lines=stim_range,
        fig_path="figures/figure_01.png"
    )