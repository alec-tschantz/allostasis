import matplotlib.pyplot as plt
import numpy as np

from mdp.mdp import MDP

# Need range of continuous intero observations
# Color line based on valence

RUNNING_LEN = 10

TAKE_WATER_RUNNING = 0
NOT_WATER_RUNNING = 1
NULL_WATER_RUNNING = 2

TAKE_WATER_NOT_RUNNING = 3
NOT_WATER_NOT_RUNNING = 4
NULL_WATER_NOT_RUNNING = 5

RUNNING_WATER_STATE = range(0, RUNNING_LEN)
RUNNING_NO_WATER_STATE = range(RUNNING_LEN, RUNNING_LEN * 2)

THIRST_OBS = range(0, RUNNING_LEN)

RUNNING_WATER_OBS = range(0, RUNNING_LEN)
RUNNING_NO_WATER_OBS = range(RUNNING_LEN, RUNNING_LEN * 2)

NUM_CONTROL = 6
NUM_STATES = len(RUNNING_WATER_STATE) + len(RUNNING_NO_WATER_STATE)
NUM_INTERO_OBS = len(THIRST_OBS)
NUM_EXTERO_OBS = len(RUNNING_WATER_OBS) + len(RUNNING_NO_WATER_OBS)


def get_obs(state):
    extero_obs = state
    if state in RUNNING_WATER_STATE:
        intero_obs = 0
    elif state in RUNNING_NO_WATER_STATE:
        intero_obs = RUNNING_NO_WATER_STATE.index(state)
    else:
        raise ValueError(f"{state} is incorrect")
    return extero_obs, intero_obs


def update_state(state, action):
    if state in RUNNING_WATER_STATE:
        has_water = True
        running_state = RUNNING_WATER_STATE.index(state)
    elif state in RUNNING_NO_WATER_STATE:
        has_water = False
        running_state = RUNNING_NO_WATER_STATE.index(state)
    else:
        raise ValueError(f"{state} is incorrect")

    if action is TAKE_WATER_RUNNING:
        if running_state < RUNNING_LEN - 1:
            state = RUNNING_WATER_STATE[running_state + 1]
        else:
            state = state
    elif action is NOT_WATER_RUNNING:
        if running_state < RUNNING_LEN - 1:
            state = RUNNING_NO_WATER_STATE[running_state + 1]
        else:
            state = state
    elif action is NULL_WATER_RUNNING:
        if has_water:
            if running_state < RUNNING_LEN - 1:
                state = RUNNING_WATER_STATE[running_state + 1]
            else:
                state = state
        else:
            if running_state < RUNNING_LEN - 1:
                state = RUNNING_NO_WATER_STATE[running_state + 1]
            else:
                state = state
    elif action is TAKE_WATER_NOT_RUNNING:
        state = RUNNING_WATER_STATE[0]
    elif action is NOT_WATER_NOT_RUNNING:
        state = RUNNING_NO_WATER_STATE[0]
    elif action is NULL_WATER_NOT_RUNNING:
        if has_water:
            state = RUNNING_WATER_STATE[0]
        else:
            state = RUNNING_NO_WATER_STATE[0]
    else:
        raise ValueError(f"{action} is incorrect")

    return state


def predict_obs(arr):
    _map = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0, 6: 1.2, 7: 1.4, 8: 1.6, 9: 1.8, 10: 2.0}
    new_arr = []
    for a in arr:
        n_new_arr = []
        for b in a:
            val = _map[np.argmax(b)]
            n_new_arr.append(val)
        new_arr.append(n_new_arr)
    return new_arr


if __name__ == "__main__":

    # exteroceptive observations
    A_extero = np.eye(NUM_STATES, NUM_EXTERO_OBS)

    # interoceptive observations
    A_intero = np.zeros((NUM_INTERO_OBS, NUM_STATES))
    A_intero[:, RUNNING_NO_WATER_STATE] = np.eye(len(THIRST_OBS), len(RUNNING_NO_WATER_STATE))
    A_intero[0, RUNNING_WATER_STATE] = 1.0

    # transition matrix
    B = np.zeros((NUM_CONTROL, NUM_STATES, NUM_STATES))

    # --------------- Running ------------------
    # running with water
    for i in range(RUNNING_LEN - 1):
        B[TAKE_WATER_RUNNING, RUNNING_WATER_STATE[i + 1], RUNNING_WATER_STATE[i]] = 1.0
        B[TAKE_WATER_RUNNING, RUNNING_WATER_STATE[i + 1], RUNNING_NO_WATER_STATE[i]] = 1.0
    B[TAKE_WATER_RUNNING, RUNNING_WATER_STATE[-1], RUNNING_WATER_STATE[-1]] = 1.0
    B[TAKE_WATER_RUNNING, RUNNING_WATER_STATE[-1], RUNNING_NO_WATER_STATE[-1]] = 1.0
    # running without water
    for i in range(RUNNING_LEN - 1):
        B[NOT_WATER_RUNNING, RUNNING_NO_WATER_STATE[i + 1], RUNNING_NO_WATER_STATE[i]] = 1.0
        B[NOT_WATER_RUNNING, RUNNING_NO_WATER_STATE[i + 1], RUNNING_WATER_STATE[i]] = 1.0
    B[NOT_WATER_RUNNING, RUNNING_NO_WATER_STATE[-1], RUNNING_NO_WATER_STATE[-1]] = 1.0
    B[NOT_WATER_RUNNING, RUNNING_NO_WATER_STATE[-1], RUNNING_WATER_STATE[-1]] = 1.0

    # running null water
    for i in range(RUNNING_LEN - 1):
        B[NULL_WATER_RUNNING, RUNNING_NO_WATER_STATE[i + 1], RUNNING_NO_WATER_STATE[i]] = 1.0
        B[NULL_WATER_RUNNING, RUNNING_WATER_STATE[i + 1], RUNNING_WATER_STATE[i]] = 1.0
    B[NULL_WATER_RUNNING, RUNNING_WATER_STATE[-1], RUNNING_WATER_STATE[-1]] = 1.0
    B[NULL_WATER_RUNNING, RUNNING_NO_WATER_STATE[-1], RUNNING_NO_WATER_STATE[-1]] = 1.0

    # --------------- Not running ------------------
    # not running with water
    for i in range(RUNNING_LEN - 1):
        B[TAKE_WATER_NOT_RUNNING, RUNNING_WATER_STATE[0], RUNNING_WATER_STATE[i]] = 1.0
        B[TAKE_WATER_NOT_RUNNING, RUNNING_WATER_STATE[0], RUNNING_NO_WATER_STATE[i]] = 1.0
    B[TAKE_WATER_NOT_RUNNING, RUNNING_WATER_STATE[0], RUNNING_WATER_STATE[-1]] = 1.0
    B[TAKE_WATER_NOT_RUNNING, RUNNING_WATER_STATE[0], RUNNING_NO_WATER_STATE[-1]] = 1.0

    # not running without water
    B[NOT_WATER_RUNNING, RUNNING_WATER_STATE, :] = 0.0
    for i in range(RUNNING_LEN - 1):
        B[NOT_WATER_NOT_RUNNING, RUNNING_NO_WATER_STATE[0], RUNNING_NO_WATER_STATE[i]] = 1.0
        B[NOT_WATER_NOT_RUNNING, RUNNING_NO_WATER_STATE[0], RUNNING_WATER_STATE[i]] = 1.0
    B[NOT_WATER_NOT_RUNNING, RUNNING_NO_WATER_STATE[0], RUNNING_WATER_STATE[-1]] = 1.0
    B[NOT_WATER_NOT_RUNNING, RUNNING_NO_WATER_STATE[0], RUNNING_NO_WATER_STATE[-1]] = 1.0

    # NOT running null water
    B[NULL_WATER_NOT_RUNNING, RUNNING_WATER_STATE, :] = 0.0
    for i in range(RUNNING_LEN - 1):
        B[NULL_WATER_NOT_RUNNING, RUNNING_NO_WATER_STATE[0], RUNNING_NO_WATER_STATE[i]] = 1.0
        B[NULL_WATER_NOT_RUNNING, RUNNING_WATER_STATE[0], RUNNING_WATER_STATE[i]] = 1.0
    B[NULL_WATER_NOT_RUNNING, RUNNING_WATER_STATE[0], RUNNING_WATER_STATE[-1]] = 1.0
    B[NULL_WATER_NOT_RUNNING, RUNNING_NO_WATER_STATE[0], RUNNING_NO_WATER_STATE[-1]] = 1.0

    # priors
    C = np.zeros((NUM_INTERO_OBS, 1))
    prior = RUNNING_LEN
    for i in range(RUNNING_LEN):
        C[i] = prior
        prior -= 1.0

    # policies
    policy_1 = [
        TAKE_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
    ]

    policy_2 = [
        NOT_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
        NULL_WATER_RUNNING,
    ]

    policies = [policy_1, policy_2]

    # init simulation
    mdp = MDP(A_extero, A_intero, B, C, policies)
    state = RUNNING_NO_WATER_STATE[0]
    extero_obs, intero_obs = get_obs(state)

    action, obs, kl = mdp.step(extero_obs, intero_obs)
    state = update_state(state, action)
    extero_obs, intero_obs = get_obs(state)
    print(f"{0} extero obs {extero_obs} intero obs {intero_obs} action {action}")

    # simulation
    for t in range(1, RUNNING_LEN - 1):
        action, _, _ = mdp.step(extero_obs, intero_obs)
        state = update_state(state, action)
        extero_obs, intero_obs = get_obs(state)
        print(f"{t} extero obs {extero_obs} intero obs {intero_obs} action {action}")


    obs = predict_obs(obs)
    # plot beliefs
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(kl[0], marker='x', color='r')
    ax[0, 0].set_title("KL divergence (run with water)")
    ax[0, 0].set_ylabel("KL divergence")
    ax[0, 0].set_xlabel("Future time")
    
    ax[0, 1].plot(obs[0],marker='x', color='r')
    ax[0, 1].set_title("Predicted thirst (run with water)")
    ax[0, 1].set_ylabel("Predicted thirst")
    ax[0, 1].set_xlabel("Future time")
    ax[0, 1].set_ylim(0, 2)
    ax[0, 0].set_ylim(1, 5)
    
    ax[1, 0].plot(kl[1], marker='x')
    ax[1, 0].set_title("KL divergence (run without water)")
    ax[1, 0].set_xlabel("Future time")
    ax[1, 0].set_ylabel("KL divergence")

    ax[1, 1].plot(obs[1], marker='x')
    ax[1, 1].set_title("Predicted thirst (run without water)")
    ax[1, 1].set_ylabel("Predicted thirst")
    ax[1, 1].set_xlabel("Future time")

    ax[1, 0].set_ylim(1, 5)
    ax[1, 1].set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig("figures/figure_5.png")

