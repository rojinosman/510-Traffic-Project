# q_agent.py
import numpy as np

class Discretizer:
    """
    Uniform binning for each feature to convert continuous state -> discrete index tuple.
    """
    def __init__(self, ranges, bins_per_feature):
        self.features = ["phase", "queue", "wait", "thru"]
        self.edges = {}
        for f in self.features:
            lo, hi = ranges[f]
            n_bins = bins_per_feature.get(f, 10)
            self.edges[f] = np.linspace(lo, hi, num=n_bins + 1)

    def encode(self, s):
        # s = [phase, queue, wait, throughput]
        vals = [s[0], s[1], s[2], s[3]]
        idxs = []
        for val, f in zip(vals, self.features):
            edges = self.edges[f]
            b = np.digitize(val, edges[1:-1], right=False)  # 0..bins-1
            idxs.append(int(b))
        return tuple(idxs)

class QAgent:
    """Tabular Q-learning (off-policy) with ε-greedy exploration."""
    def __init__(
        self,
        action_space_n,
        discretizer: Discretizer,
        alpha=0.15,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=0,
    ):
        self.nA = action_space_n
        self.disc = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        shape = tuple(len(self.disc.edges[f]) - 1 for f in self.disc.features) + (self.nA,)
        self.Q = np.zeros(shape, dtype=np.float32)

    def act(self, s):
        ds = self.disc.encode(s)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.nA))
        return int(np.argmax(self.Q[ds]))

    def observe(self, s, a, r, s_next, done):
        ds = self.disc.encode(s)
        ds_next = self.disc.encode(s_next)
        qsa = self.Q[ds + (a,)]
        target = r if done else r + self.gamma * np.max(self.Q[ds_next])
        self.Q[ds + (a,)] = (1 - self.alpha) * qsa + self.alpha * target

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
