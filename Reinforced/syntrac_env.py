# syntrac_env.py
import numpy as np
import pandas as pd

class SynTraCEnv:
    """
    Simple traffic signal environment backed by a CSV time-series.

    State (float32 vector, length 4): [signal_phase, queue_length, waiting_time, throughput]
    Action (int): 0 = hold phase, 1 = toggle phase (simulated effect on reward only)
    Episode ends when we reach last row of CSV.
    """
    def __init__(self, csv_path="syntrac_merged.csv", reward_cfg=None, seed=0):
        self.rng = np.random.default_rng(seed)
        df = pd.read_csv(csv_path)
        if not set(["signal_phase", "queue_length", "waiting_time", "throughput"]).issubset(df.columns):
            raise ValueError("CSV missing required columns.")
        self.data = df[["signal_phase", "queue_length", "waiting_time", "throughput"]].to_numpy(dtype=np.float32)
        self.n = len(self.data)
        self.idx = 0
        self.action_space_n = 2

        # Reward config
        self.cfg = dict(
            w_throughput=1.0,
            w_queue=0.1,
            w_wait=0.1,
            action_toggle_bonus=0.2  # small bonus for toggling when queue is high
        )
        if reward_cfg:
            self.cfg.update(reward_cfg)

    def reset(self):
        self.idx = 0
        return self._get_state()

    def _get_state(self):
        # clamp for safety so we can return a valid obs even at terminal
        i = min(self.idx, self.n - 1)
        return self.data[i]

    def step(self, action: int):
        # Current observation
        s = self._get_state()
        phase, queue, wait, throughput = map(float, s)

        # Reward: encourage throughput, penalize queue and waiting.
        reward = (
            self.cfg["w_throughput"] * throughput
            - self.cfg["w_queue"] * queue
            - self.cfg["w_wait"] * wait
        )

        # Small shaping: toggling might help if queue is high
        if action == 1 and queue > 25:
            reward += self.cfg["action_toggle_bonus"]

        # Advance time
        self.idx += 1
        done = self.idx >= self.n
        next_obs = self._get_state()
        info = {}
        return next_obs, float(reward), done, info

    # Useful normalizer for agents (optional)
    @staticmethod
    def feature_ranges():
        # approximate ranges used to build bins for discretization
        return {
            "signal_phase": (0.0, 1.0),
            "queue_length": (0.0, 80.0),
            "waiting_time": (0.0, 15.0),
            "throughput": (0.0, 10.0),
        }
