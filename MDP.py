import numpy as np
import time

import copy

class mdp(object):
    def __init__(self, kernel = np.ones((1,1,1,1)), R = [1], action_mask = np.ones((1, 1))):
        """
        :param kernel(s_,r,s,a): joint prob. of transition to state s_ and R[r] when choosing action a at state s
        :param R: R[r] is reward value at index r
        :param action_mask: action_mask[s][a]==1 only if action a is allowed at state s
        """

        assert np.shape(kernel)[0] == np.shape(kernel)[2]
        assert np.shape(kernel)[1] == len(R)
        assert np.shape(action_mask)[1] == np.shape(kernel)[3]
        self.kernel = kernel


        # State space
        self.lenS = np.shape(kernel)[0]
        self.S = np.arange(0, self.lenS)

        # Reward space
        self.lenR = len(R)
        self.R = np.asarray(R)

        # Action space
        self.lenA = np.shape(kernel)[3]
        self.A = np.zeros(self.lenS, dtype=object)
        for s in self.S:
            self.A[s] = (np.argwhere(action_mask[s] == 1)).reshape(self.lenA)
    def get_transition_matrix(self, f = None):
        """
        :param f: a fixed policy f:S -> A
        :return Pf: transition matrix s.t. Pf(s, s_) is the probability of transition from s to s_ under policy f
        """
        if f is None:
            f = np.zeros(self.lenS)
        Pf = np.zeros((self.lenS, self.lenS))
        """
        IMPLEMENT HERE
        """

        for s in range(self.lenS):
            next_state = np.array(list(set([(s + 1) % self.lenS, 0])))

            Pf[s, next_state] = np.sum(self.kernel[next_state, :, s, int(f[s])], axis = 1)

        return Pf
    def get_reward_vector(self, f = None):
        """
        :param f: a fixed policy f:S -> A
        :return Rf: reward vector s.t. Rf(s) is the expected reward at s under policy f
        """
        if f is None:
            f = np.zeros(self.lenS)
        Rf = np.zeros(self.lenS)
        """
        IMPLEMENT HERE
        """
        for s in range(self.lenS):
            next_state = np.array(list(set([(s + 1) % self.lenS, 0])))

            Rf[s] = np.sum(np.dot(self.kernel[next_state, :, s, int(f[s])], self.R))

        return Rf

def cliffwalk(action_str = [1, 1, 1, 1], reward_str = [0, 0, 0, 1], noise = 0.1):
    assert len(action_str) == len(reward_str)
    # State space
    lenS = len(action_str)
    S = np.arange(0,lenS)

    # Action space
    lenA = 2
    A = np.arange(0,lenA)

    # Reward space
    R = np.asarray(list(set(np.concatenate((reward_str, [0])))))
    lenR = len(R)
    kernel = np.zeros((lenS, lenR, lenS, lenA))
    for s in S:
        # action_str[s]: action intended at state 2
        # (action_str[s]+1)%2: action not intended at state 2
        kernel[(s + 1) % lenS][np.argwhere(R==reward_str[s])[0][0]][s][action_str[s]] += 1 - noise
        kernel[0][np.argwhere(R == 0)[0][0]][s][action_str[s]] += noise
        kernel[0][np.argwhere(R == 0)[0][0]][s][(action_str[s]+1)%2] = 1
    env = mdp(kernel, R, np.ones((lenS, lenA)))
    return env

class policy_iteration(object):
    def __init__(self, env = cliffwalk(), discount = 0.99, init_v = None, init_policy = None):
        self.env = env
        self.discount = discount
        self.elapsed_time = 0.0
        self.elapsed_iter = 0

        if init_v is None:
            self.Vt = np.zeros(env.lenS)
        else:
            self.Vt = init_v
        if init_policy is None:
            self.ft = np.ones(env.lenS)
        else:
            self.ft = init_policy
    def iteration(self):
        start_time = time.time()
        # Perform an iteration
        self.policy_evaluate()
        self.policy_improve()
        # update statistics
        self.elapsed_time += time.time() - start_time
        self.elapsed_iter += 1

    def policy_evaluate(self):
        """
        IMPLEMENT HERE
        self.Vt = ...
        Use Vt = (I-discount*Pf)^(-1)*Rf
        """
        Pf = self.env.get_transition_matrix(f = self.ft)
        Rf = self.env.get_reward_vector(f = self.ft)

        self.Vt = np.matmul(np.linalg.inv((np.identity(self.env.lenS) - self.discount * Pf)), Rf).reshape([-1])

    def policy_improve(self):
        """
        IMPLEMENT HERE
        self.ft = ...
        """
        for s in range(self.env.lenS):
            per_state_action_val = np.zeros(self.env.lenA)

            next_state = np.array(list(set([(s + 1) % self.env.lenS, 0])))

            for a in range(self.env.lenA):
                per_state_action_val[a] = np.sum(np.dot(self.env.kernel[next_state, :, s, a], self.env.R)) + \
                                          np.sum(self.discount * np.dot(self.Vt[next_state], self.env.kernel[next_state, :, s, a]))

            self.ft[s] = np.argmax(per_state_action_val)

class value_iteration(object):
    def __init__(self, env = cliffwalk(), discount = 0.99, init_v = None):
        self.env = env
        self.discount = discount
        self.elapsed_time = 0.0
        self.elapsed_iter = 0

        if init_v is None:
            self.Vt = np.zeros(env.lenS)
        else:
            self.Vt = init_v

        self.ft = np.zeros(env.lenS)

    def iteration(self):
        start_time = time.time()
        # Perform an iteration
        """
        IMPLEMENT HERE
        self.Vt = ...
        self.ft = ...
        """
        # update statistics

        for s in range(self.env.lenS):
            per_state_action_val = np.zeros(self.env.lenA)

            next_state = next_state = np.array(list(set([(s + 1) % self.env.lenS, 0])))

            for a in range(self.env.lenA):
                per_state_action_val[a] = np.sum(np.dot(self.env.kernel[next_state, :, s, a], self.env.R)) + \
                                          np.sum(self.discount * np.dot(self.Vt[next_state], self.env.kernel[next_state, :, s, a]))

            self.Vt[s] = np.max(per_state_action_val)
            self.ft[s] = np.argmax(per_state_action_val)

        self.elapsed_time += time.time() - start_time
        self.elapsed_iter += 1

if __name__ == '__main__':
    # Example main
    action_str = [1, 1, 1, 1]
    reward_str = [0, 0, 0, 1]
    noise = 0
    env = cliffwalk(action_str, reward_str, noise)
    pol_iter = policy_iteration(env, 0.99)
    val_iter = value_iteration(env, 0.99)

    for t in range(0, 1000):
        pol_iter.iteration()
        val_iter.iteration()

    print(pol_iter.Vt)
    print(val_iter.Vt)
    print(pol_iter.ft)
    print(pol_iter.ft)

    print('Policy iteration took %fs'%pol_iter.elapsed_time)
    print('Value iteration took %fs' %val_iter.elapsed_time)
