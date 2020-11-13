import numpy as np

from mdp.mdp import MDP


if __name__ == "__main__":
    prior = 3
    init_state = 0
    num_control = 5
    num_states = 5
    num_intero_obs = 5
    num_extero_obs = 5

    A_extero = np.eye(num_extero_obs, num_states)
    A_intero = np.eye(num_intero_obs, num_states)
    B = np.zeros((num_control, num_states, num_states))
    for i in range(num_control):
        B[i, i, :] = 1.0
    C = np.zeros((num_intero_obs, 1))
    C[prior] = 1.0
    policies = [[init_state, 1, 2, 3], [init_state, 1, 3, 3], [init_state, 1, 1, 1]]

    mdp = MDP(A_extero, A_intero, B, C, policies)
    mdp.reset(init_state, init_state, init_state)
    action = mdp.step(init_state, init_state)
    action = mdp.step(action, action)   
    action = mdp.step(action, action)   
