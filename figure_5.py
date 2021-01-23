import matplotlib.pyplot as plt
import numpy as np

from mdp.mdp import MDP

RUNNING_LEN = 20
MAX_THIRST = 15

TAKE_WATER_RUNNING = 0
NOT_WATER_RUNNING = 1
NULL_WATER_RUNNING = 2

TAKE_WATER_NOT_RUNNING = 3
NOT_WATER_NOT_RUNNING = 4
NULL_WATER_NOT_RUNNING = 5

RUNNING_WATER_STATE = range(0, RUNNING_LEN)
RUNNING_NO_WATER_STATE = range(RUNNING_LEN, RUNNING_LEN * 2)

THIRST_OBS = range(0, MAX_THIRST)

RUNNING_WATER_OBS = range(0, RUNNING_LEN)
RUNNING_NO_WATER_OBS = range(RUNNING_LEN, RUNNING_LEN * 2)

NUM_CONTROL = 6
NUM_STATES = len(RUNNING_WATER_STATE) + len(RUNNING_NO_WATER_STATE)
NUM_INTERO_OBS = len(THIRST_OBS)
NUM_EXTERO_OBS = len(RUNNING_WATER_OBS) + len(RUNNING_NO_WATER_OBS)


def get_water_running(states):
    running_water_states = states[RUNNING_WATER_STATE]
    water = np.sum(states[RUNNING_NO_WATER_STATE])
    running_no_water_states = states[RUNNING_NO_WATER_STATE]
    no_water = np.sum(running_no_water_states)
    water_state = np.array([water, no_water])
    running_state = np.zeros(RUNNING_LEN)
    for i in range(RUNNING_LEN):
        prob_water = states[i]
        prob_no_water = states[RUNNING_LEN + i]
        prob = prob_water + prob_no_water
        running_state[i] = prob
    running_state = running_state / np.sum(running_state)
    return running_state, water_state


def get_obs(state, t, n):
    extero_obs = state

    if state in RUNNING_WATER_STATE:
        intero_obs = n
        if t % 3 == 0:
            n = n + 1
        intero_obs = 0

    elif state in RUNNING_NO_WATER_STATE:
        intero_obs = min(NUM_INTERO_OBS - 1, RUNNING_NO_WATER_STATE.index(state))
    else:
        raise ValueError(f"{state} is incorrect")
    return extero_obs, intero_obs, n


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
    _map = np.linspace(0, 2.0, RUNNING_LEN + 1)
    _map[NUM_INTERO_OBS:] = _map[NUM_INTERO_OBS - 1] * NUM_INTERO_OBS
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
    running_no_water = np.eye(NUM_INTERO_OBS, len(RUNNING_NO_WATER_STATE))
    running_no_water[-1, NUM_INTERO_OBS:] = 1.0
    A_intero[:, RUNNING_NO_WATER_STATE] = running_no_water
    A_intero[0, RUNNING_WATER_STATE] = 1.0

    A_intero = A_intero + np.random.rand(*A_intero.shape) / 10
    A_extero = A_extero + np.random.rand(*A_extero.shape) / 10

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
    prior = NUM_INTERO_OBS
    for i in range(NUM_INTERO_OBS):
        C[i] = prior
        prior -= 1.0

    # policies
    policy_1 = [TAKE_WATER_RUNNING] + [NULL_WATER_RUNNING] * (RUNNING_LEN - 2)
    policy_1[RUNNING_LEN // 2] = TAKE_WATER_RUNNING
    policy_2 = [NOT_WATER_RUNNING] + [NULL_WATER_RUNNING] * (RUNNING_LEN - 2)
    policy_2[RUNNING_LEN // 2] = NOT_WATER_RUNNING
    policies = [policy_1, policy_2]

    # init simulation
    n = 0
    mdp = MDP(A_extero, A_intero, B, C, policies)
    state = RUNNING_NO_WATER_STATE[0]
    extero_obs, intero_obs, n = get_obs(state, 0, n)

    action, obs, extero_obs_model, kl = mdp.step(extero_obs, intero_obs)
    state = update_state(state, action)
    extero_obs, intero_obs, n = get_obs(state, 0, n)
    print(f"{0} extero obs {extero_obs} intero obs {intero_obs} action {action}")
    obs_2, extero_obs_model_2, kl_2, = None, None, None
    # simulation
    for t in range(1, RUNNING_LEN - 1):
        if t == RUNNING_LEN // 2:
            action, obs_2, extero_obs_model_2, kl_2 = mdp.step(extero_obs, intero_obs)
        else:
            action, _, _, _ = mdp.step(extero_obs, intero_obs)
        state = update_state(state, action)
        extero_obs, intero_obs, n = get_obs(state, t, n)
        running_state, water_state = get_water_running(mdp.sQ)
        print(f"{t} extero obs {extero_obs} intero obs {intero_obs} action {action}")
        print(f"{t} running_state {running_state.round()} water_state {water_state.round()} ")

    running_obs = [[], []]
    water_obs = [[], []]
    for i in range(2):
        for extero_obs in extero_obs_model[i]:
            running_state, water_state = get_water_running(extero_obs)
            running_obs[i].append(np.argmax(running_state))
            water_obs[i].append(np.argmax(water_state))

    obs = predict_obs(obs)
    # plot beliefs
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(kl[0], marker="x", color="r")
    ax[0, 0].set_title("KL divergence (run with water)")
    ax[0, 0].set_ylabel("KL divergence")
    ax[0, 0].set_xlabel("Future time")

    ax[0, 1].plot(obs[0], marker="x", color="r")
    # ax[0, 1].plot(running_obs[0], marker="x", color="g")
    #   ax[0, 1].plot(water_obs[0], marker="x", color="b")
    ax[0, 1].set_title("Predicted thirst (run with water)")
    ax[0, 1].set_ylabel("Predicted thirst")
    ax[0, 1].set_xlabel("Future time")
    ax[0, 1].set_ylim(0, 2)
    ax[0, 0].set_ylim(0, 3)

    ax[1, 0].plot(kl[1], marker="x")
    ax[1, 0].set_title("KL divergence (run without water)")
    ax[1, 0].set_xlabel("Future time")
    ax[1, 0].set_ylabel("KL divergence")

    ax[1, 1].plot(obs[1], marker="x", color="r")
    #  ax[1, 1].plot(running_obs[1], marker="x", color="g")
    #   ax[1, 1].plot(water_obs[1], marker="x", color="b")
    ax[1, 1].set_title("Predicted thirst (run without water)")
    ax[1, 1].set_ylabel("Predicted thirst")
    ax[1, 1].set_xlabel("Future time")

    ax[1, 0].set_ylim(0, 3)
    ax[1, 1].set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig("figures/figure_5.png")

    obs_2 = predict_obs(obs_2)
    # plot beliefs
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(kl_2[0], marker="x", color="r")
    ax[0, 0].set_title("KL divergence (run with water)")
    ax[0, 0].set_ylabel("KL divergence")
    ax[0, 0].set_xlabel("Future time")

    ax[0, 1].plot(obs_2[0], marker="x", color="r")
    # ax[0, 1].plot(running_obs[0], marker="x", color="g")
    #   ax[0, 1].plot(water_obs[0], marker="x", color="b")
    ax[0, 1].set_title("Predicted thirst (run with water)")
    ax[0, 1].set_ylabel("Predicted thirst")
    ax[0, 1].set_xlabel("Future time")
    ax[0, 1].set_ylim(0, 2)
    ax[0, 0].set_ylim(0, 3)

    ax[1, 0].plot(kl_2[1], marker="x")
    ax[1, 0].set_title("KL divergence (run without water)")
    ax[1, 0].set_xlabel("Future time")
    ax[1, 0].set_ylabel("KL divergence")

    ax[1, 1].plot(obs_2[1], marker="x", color="r")
    #  ax[1, 1].plot(running_obs[1], marker="x", color="g")
    #   ax[1, 1].plot(water_obs[1], marker="x", color="b")
    ax[1, 1].set_title("Predicted thirst (run without water)")
    ax[1, 1].set_ylabel("Predicted thirst")
    ax[1, 1].set_xlabel("Future time")

    ax[1, 0].set_ylim(0, 3)
    ax[1, 1].set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig("figures/figure_6.png")

