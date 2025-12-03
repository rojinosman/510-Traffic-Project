import random
import numpy as np
from collections import defaultdict


class QAgent:
    def __init__(
        self,
        actions=2,
        alpha=0.1,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(actions, dtype=float))

    def act(self, state):
        # epsilon-greedy
        if random.random() < self.eps:
            return random.randrange(self.actions)
        return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s_next, done):
        q_sa = self.Q[s][a]
        target = r if done else (r + self.gamma * np.max(self.Q[s_next]))
        self.Q[s][a] = q_sa + self.alpha * (target - q_sa)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
