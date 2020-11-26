from enum import IntEnum

import numpy as np

from pc.types import Variable, Node, Data, Action, InverseFunction
from pc.model import Model
from mdp.mdp import MDP


class Control(IntEnum):
    EAT = 0
    NOT_EATING = 1


class States(IntEnum):
    ZERO_FULL = 0
    ONE_FULL = 1
    TWO_FULL = 2
    THREE_FULL = 3
    FOUR_FULL = 4


class ExteroceptiveObsveraions(IntEnum):
    EAT = 0
    NOT_EATING = 1


class InteroceptiveObservations(IntEnum):
    ZERO_FULL = 0
    ONE_FULL = 1
    TWO_FULL = 2
    THREE_FULL = 3
    FOUR_FULL = 4


def get_observation(state):
    if state == States.ZERO_FULL:
        return ExteroceptiveObsveraions.EAT, InteroceptiveObservations.ZERO_FULL
    elif state is States.ONE_FULL:
        return ExteroceptiveObsveraions.EAT, InteroceptiveObservations.ONE_FULL
    elif state is States.TWO_FULL:
        # TODO: 50/50 chance of eat / not eat
        return ExteroceptiveObsveraions.NOT_EATING, InteroceptiveObservations.TWO_FULL
    elif state is States.THREE_FULL:
        return ExteroceptiveObsveraions.NOT_EATING, InteroceptiveObservations.THREE_FULL
    elif state is States.FOUR_FULL:
        return ExteroceptiveObsveraions.NOT_EATING, InteroceptiveObservations.FOUR_FULL


def get_state(action, state):
    if state is States.ZERO_FULL:
        if action == Control.EAT:
            return States.ONE_FULL
        elif action == Control.NOT_EATING:
            return States.ZERO_FULL
    elif state is States.ONE_FULL:
        if action == Control.EAT:
            return States.TWO_FULL
        elif action == Control.NOT_EATING:
            return States.ZERO_FULL
    elif state is States.TWO_FULL:
        if action == Control.EAT:
            return States.THREE_FULL
        elif action == Control.NOT_EATING:
            return States.ONE_FULL
    elif state is States.THREE_FULL:
        if action == Control.EAT:
            return States.FOUR_FULL
        elif action == Control.NOT_EATING:
            return States.TWO_FULL
    elif state is States.FOUR_FULL:
        if action == Control.EAT:
            return States.FOUR_FULL
        elif action == Control.NOT_EATING:
            return States.THREE_FULL

def get_cont_observation():
    pass

def predict_obs(arr):
    _map = {0: 0, 1:0.2, 2: 0.4, 3:0.6, 4:0.8}
    new_arr = []
    for a in arr:
        n_new_arr = []
        for b in a:
            val = _map[np.argmax(b)]
            n_new_arr.append(val)
        new_arr.append(n_new_arr)
    return new_arr

if __name__ == "__main__":
    delta_time = 0.1
    num_iterations = 100
    intero_range = np.arange(0, 1, 5)

    num_control = len(Control)
    num_states = len(States)
    num_intero_obs = len(InteroceptiveObservations)
    num_extero_obs = len(ExteroceptiveObsveraions)

    A_extero = np.zeros((num_extero_obs, num_states))
    A_extero[ExteroceptiveObsveraions.EAT, States.ZERO_FULL] = 1.0
    A_extero[ExteroceptiveObsveraions.EAT, States.ONE_FULL] = 1.0
    A_extero[ExteroceptiveObsveraions.NOT_EATING, States.THREE_FULL] = 1.0
    A_extero[ExteroceptiveObsveraions.NOT_EATING, States.FOUR_FULL] = 1.0
    A_intero = np.eye(num_intero_obs, num_states)

    B = np.zeros((num_control, num_states, num_states))
    B[Control.EAT, States.ONE_FULL, States.ZERO_FULL] = 1.0
    B[Control.EAT, States.TWO_FULL, States.ONE_FULL] = 1.0 
    B[Control.EAT, States.THREE_FULL, States.TWO_FULL] = 1.0
    B[Control.EAT, States.FOUR_FULL, States.THREE_FULL] = 1.0
    B[Control.EAT, States.FOUR_FULL, States.FOUR_FULL] = 1.0

    B[Control.NOT_EATING, States.ZERO_FULL, States.ZERO_FULL] = 1.0
    B[Control.NOT_EATING, States.ZERO_FULL, States.ONE_FULL] = 1.0
    B[Control.NOT_EATING, States.ONE_FULL, States.TWO_FULL] = 1.0
    B[Control.NOT_EATING, States.TWO_FULL, States.THREE_FULL] = 1.0
    B[Control.NOT_EATING, States.THREE_FULL, States.FOUR_FULL] = 1.0

    C = np.zeros((num_intero_obs, 1))
    C[InteroceptiveObservations.ZERO_FULL] = 0.1
    C[InteroceptiveObservations.ONE_FULL] = 0.6
    C[InteroceptiveObservations.TWO_FULL] = 1.0
    C[InteroceptiveObservations.THREE_FULL] = 0.3
    C[InteroceptiveObservations.FOUR_FULL] = 0.1

    policies = [
        [Control.EAT, Control.EAT, Control.EAT],
        [Control.NOT_EATING, Control.NOT_EATING, Control.NOT_EATING],
        [Control.EAT, Control.EAT, Control.NOT_EATING],
    ]

    mdp = MDP(A_extero, A_intero, B, C, policies, restrictive_policies=False)

    state = States.ZERO_FULL
    extero_obs, intero_obs = get_observation(state)
    mdp.reset(extero_obs, intero_obs)

    # STEP 1
    action, obs, kl = mdp.step(extero_obs, intero_obs, ret_exp_obs=True)
    obs_plot = predict_obs(obs)
    for ob in obs:
        print("/")
        for o in ob:
            print(o.T.round(3))

    for k in kl:
        print("/")
        print(np.array(k).round(3))
    state = get_state(action, state)
    extero_obs, intero_obs = get_observation(state)
    print(f"step 1 action {action} state {state} ext {extero_obs} int {intero_obs}")



    # STEP 2
    action = mdp.step(extero_obs, intero_obs)
    state = get_state(action, state)
    extero_obs, intero_obs = get_observation(state)
    print(f"step 2 action {action} state {state} ext {extero_obs} int {intero_obs}")


    # STEP 3
    action = mdp.step(extero_obs, intero_obs)
    state = get_state(action, state)
    extero_obs, intero_obs = get_observation(state)
    print(f"step 3 action {action} state {state} ext {extero_obs} int {intero_obs}")


    import matplotlib.pyplot as plt 
    fig_width = 6
    _, axes = plt.subplots(1, 1, figsize=(fig_width, 3))
    obs_plot = np.array(obs_plot)
    axes.plot(obs_plot[0], color="red", label="E, E, E")
    axes.plot(obs_plot[1], color="green", label="F, F, F")
    axes.plot(obs_plot[2], color="blue", label="E, E, F")
    plt.ylabel("Hunger signal")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig("figures/4a.png")

    fig_width = 6
    _, axes = plt.subplots(1, 1, figsize=(fig_width, 3))
    obs_plot = np.array(obs_plot)
    axes.plot(kl[0], color="red", label="E, E, E")
    axes.plot(kl[1], color="green", label="F, F, F")
    axes.plot(kl[2], color="blue", label="E, E, F")
    plt.ylabel("Prediction Error (KL)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig("figures/4b.png")