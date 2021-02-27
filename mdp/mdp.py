import numpy as np
from pc.types import Node, Data
from pc.model import Model


class MDP(object):
    def __init__(self, A_extero, A_intero, B, C, policies):
        self.A_extero = A_extero
        self.A_intero = A_intero
        self.B = B
        self.C = C
        self.policies = policies
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

        self.extero_obs = 0
        self.intero_obs = 0
        self.action = None
        self.global_time = 0
        self.noise = 0
        self.make_pc_model()

    def make_pc_model(self):
        self.prior = Node()
        self.intero_mu = Node()
        self.intero_data = Data(noise=self.noise)
        self.model = Model()
        self.prior_err = self.model.add_connection(self.prior, self.intero_mu)
        self.ll_err = self.model.add_connection(self.intero_mu, self.intero_data)

    def predict_intero_obs(self, intero_obs):
        # make range a param
        _map = np.linspace(0, 2.0, self.num_intero_obs)
        value = _map[intero_obs]
        # perhaps some transform w/ weights
        value = value + np.random.normal(0, self.noise)
        return value

    def infer_intero_obs(self, data):
        bins = np.linspace(0, 2.0, self.num_intero_obs)
        # add noise
        data = bins[data]
        self.model.reset()
        for _ in range(100):
            self.intero_data.update(data)
            self.model.update()

        bins = np.linspace(0.1, 2.0, self.num_intero_obs)
        value = self.prior.value
        mapped_value = np.digitize([value], bins)
        return mapped_value

    def reset(self):
        self.extero_obs = 0
        self.intero_obs = 0
        self.action = None
        self.global_time = 0

    def step(self, extero_obs, intero_obs):
        self.extero_obs = extero_obs
        #self.intero_obs = intero_obs
        self.intero_obs = self.infer_intero_obs(intero_obs)[0]
        self.infer_sQ()
        obs, extero_obs, mapped_obs, kl = self.evaluate_G()
        self.infer_uQ()
        self.action = self.get_action()
        self.global_time = self.global_time + 1
        self.update_policies()
        return self.action, obs, extero_obs, mapped_obs, kl

    def infer_sQ(self):
        ll_extero = self.log_A_extero[self.extero_obs, :]
        ll_extero = ll_extero[:, np.newaxis]
        ll_intero = self.log_A_intero[self.intero_obs, :]
        ll_intero = ll_intero[:, np.newaxis]
        if self.action is not None:
            prior = np.dot(self.B[self.action], self.sQ)
            prior = np.log(prior)
            self.sQ = self.softmax(ll_extero + ll_intero + prior)
        else:
            self.sQ = self.softmax(ll_extero + ll_intero)

    def evaluate_G(self):
        num_policies = len(self.policies)
        self.G = np.zeros([num_policies, 1])
        obs = []
        mapped_obs = []
        extero_obs = []
        kls = []

        for i, policy in enumerate(self.policies):
            fos = []
            kl = []
            extero_ob = []
            mapped_ob = []

            fs, fo, extero_fo, max_fo_intero_mapped, utility = self.counterfactual(policy[0], self.sQ)
            self.G[i] -= utility
            kl.append(utility)
            fos.append(fo)
            extero_ob.append(extero_fo)
            mapped_ob.append(max_fo_intero_mapped)
            for t in range(1, len(policy)):
                fs, fo, fo_extero, max_fo_intero_mapped, utility = self.counterfactual(policy[t], fs)
                self.G[i] -= utility
                kl.append(utility)
                fos.append(fo)
                extero_ob.append(fo_extero)
                mapped_ob.append(max_fo_intero_mapped)
            kls.append(kl)
            obs.append(fos)
            extero_obs.append(extero_ob)
            mapped_obs.append(mapped_ob)

        return obs, extero_obs, mapped_obs, kls

    def counterfactual(self, action, state):
        fs = np.dot(self.B[action], state)
        fo_extero = np.dot(self.A_extero, fs)
        fo_extero = self.normdist(fo_extero + self.p0)

        fo_intero = np.dot(self.A_intero, fs)
        fo_intero = self.normdist(fo_intero + self.p0)
        # Do BMA here
        max_fo_intero = np.argmax(fo_intero)
        max_fo_intero_mapped = self.predict_intero_obs(max_fo_intero)
        
        utility = np.sum(fo_intero * np.log(fo_intero / self.C), axis=0)
        utility = utility[0]
        # No epistemic as we have no uncertainty

        return fs, fo_intero, fo_extero, max_fo_intero_mapped, utility

    def infer_uQ(self):
        self.uQ = self.softmax(self.G)

    def get_action(self) -> int:
        hu = max(self.uQ)
        options = np.where(self.uQ == hu)[0]
        policy = int(np.random.choice(options))
        action = self.policies[policy][0]
        return action

    def update_policies(self):
        policies = []
        for policy in self.policies:
            if policy[0] == self.action or True:
                policies.append(policy[1:])
        self.policies = policies

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x: np.ndarray) -> np.ndarray:
        return np.dot(x, np.diag(1 / np.sum(x, 0)))
