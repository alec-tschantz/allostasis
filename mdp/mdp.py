import numpy as np


class MDP(object):
    def __init__(self, A_extero: np.ndarray, A_intero: np.ndarray, B: np.ndarray, C: np.ndarray):
        self.A_extero = A_extero
        self.A_intero = A_intero
        self.B = B
        self.C = C
        self.p0 = np.exp(-16)

        self.num_extero_obs = self.A_extero.shape[0]
        self.num_intero_obs = self.A_intero.shape[0]
        self.num_states = self.A_extero.shape[1]
        self.num_control = self.B.shape[0]

        self.A_extero = self.A_extero + self.p0
        self.A_extero = self.normdist(self.A_extero)
        self.log_A_extero = np.log(self.A_extero)

        self.A_intero = self.A_intero + self.p0
        self.A_intero = self.normdist(self.A_intero)
        self.log_A_intero = np.log(self.A_intero)

        self.B = self.B + self.p0
        for u in range(self.num_control):
            self.B[u] = self.normdist(self.B[u])

        if np.size(self.C, 1) > np.size(self.C, 0):
            self.C = self.C.T
        self.C = self.C + self.p0
        self.C = self.normdist(self.C)

        self.sQ = np.zeros([self.num_states, 1])
        self.uQ = np.zeros([self.num_control, 1])
        self.G = np.zeros([self.num_control, 1])

        self.action_range = np.arange(0, self.num_control)
        self.extero_obs = 0.0
        self.intero_obs = 0
        self.action = 0

    def reset(self, extero_obs: int, intero_obs: int):
        self.extero_obs = extero_obs
        self.intero_obs = intero_obs
        ll_extero = self.log_A_extero[self.extero_obs, :]
        ll_extero = ll_extero[:, np.newaxis]
        ll_intero = self.log_A_intero[self.intero_obs, :]
        ll_intero = ll_intero[:, np.newaxis]
        self.sQ = self.softmax(ll_extero + ll_intero)
        self.action = int(np.random.choice(self.action_range))

    def step(self, extero_obs: int, intero_obs: int) -> int:
        self.extero_obs = extero_obs
        self.intero_obs = intero_obs
        self.infer_sQ()
        self.evaluate_G()
        self.infer_uQ()
        return self.act()

    def infer_sQ(self):
        ll_extero = self.log_A_extero[self.extero_obs, :]
        ll_extero = ll_extero[:, np.newaxis]
        ll_intero = self.log_A_intero[self.intero_obs, :]
        ll_intero = ll_intero[:, np.newaxis]
        prior = np.dot(self.B[self.action], self.sQ)
        prior = np.log(prior)
        self.sQ = self.softmax(ll_extero + ll_intero + prior)

    def evaluate_G(self):
        self.G = np.zeros([self.num_control, 1])

        for u in range(self.num_control):
            fs = np.dot(self.B[u], self.sQ)
            fo_extero = np.dot(self.log_A_extero, fs)
            fo_extero = self.normdist(fo_extero + self.p0)

            fo_intero = np.dot(self.log_A_intero, fs)
            fo_intero = self.normdist(fo_intero + self.p0)

            utility = np.sum(fo_intero * np.log(fo_intero / self.C), axis=0)
            utility = utility[0]
            # TODO: add surprise

            self.G[u] -= utility

    def infer_uQ(self):
        self.uQ = self.softmax(self.G)

    def act(self) -> int:
        hu = max(self.uQ)
        options = np.where(self.uQ == hu)[0]
        self.action = int(np.random.choice(options))
        return self.action

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x: np.ndarray) -> np.ndarray:
        return np.dot(x, np.diag(1 / np.sum(x, 0)))

