import csv
import numpy as np


class SynTraCEnv:
    """
    Simple RL environment built from SynTraC lane-level data.

    - State: (queue_bin, current_phase, time_in_phase_bin)
      * queue_bin: discretized vehicle queue length
      * current_phase: 0 or 1 (the phase the controller is currently serving)
      * time_in_phase_bin: discretized time (seconds) since last phase change

    - Action:
      0 = keep current phase
      1 = request phase switch (only happens if min_green is satisfied)

    - Reward:
      reward = -(alpha_queue * queue_length + beta_wait * waiting_time)
               - switch_cost (if a switch actually occurs)

      So the agent is encouraged to keep queues/waiting small and avoid
      unnecessary switching.
    """

    def __init__(
        self,
        data_csv="syntrac_merged.csv",
        min_green=5.0,
        switch_cost=2.0,
        alpha_queue=1.0,
        beta_wait=0.5,
    ):
        # Load the dataset
        self.data = []
        with open(data_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "phase": int(row["signal_phase"]),
                        "queue_length": float(row["queue_length"]),
                        "waiting_time": float(row["waiting_time"]),
                        "throughput": float(row["throughput"]),
                    }
                )

        # Sort time series just in case
        self.data.sort(key=lambda x: x["timestamp"])

        # Environment parameters
        self.min_green = float(min_green)
        self.switch_cost = float(switch_cost)
        self.alpha_queue = float(alpha_queue)
        self.beta_wait = float(beta_wait)

        # Discretization bins
        self.queue_bins = [0, 1, 2, 4, 6, 10, 20]  # slightly tighter bins for lane counts
        self.time_bins = [0, 5, 10, 20, 40, 80, 160]   # time-in-phase bins

        # Number of actions available to the agent
        self.n_actions = 2

        self.reset()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _bin(self, val, bins):
        for i, b in enumerate(bins):
            if val <= b:
                return i
        return len(bins) - 1

    def _get_state(self):
        rec = self.data[self.idx]
        q_bin = self._bin(rec["queue_length"], self.queue_bins)
        t_bin = self._bin(self.time_in_phase, self.time_bins)
        # IMPORTANT: use *current* phase we control, not the logged phase
        return (q_bin, self.phase, t_bin)

    # ------------------------------------------------------------------ #
    # Gym-like API
    # ------------------------------------------------------------------ #
    def reset(self):
        """Start a new episode from the beginning of the dataset."""
        self.idx = 0
        self.time_in_phase = 0.0
        # Initialize controller phase to whatever the dataset starts with
        self.phase = self.data[0]["phase"]
        return self._get_state()

    def step(self, action):
        """
        Take an action.

        Returns:
            next_state, reward, done, info
        """
        assert action in (0, 1), "Invalid action (must be 0 or 1)"
        done = False

        # Current record
        rec = self.data[self.idx]
        queue = rec["queue_length"]
        wait = rec["waiting_time"]

        # Base reward: penalize queue and waiting time (from lane counts)
        reward = -(self.alpha_queue * queue + self.beta_wait * wait)

        # Decide whether to switch
        switched = False
        if action == 1 and self.time_in_phase >= self.min_green:
            self.phase = 1 - self.phase
            self.time_in_phase = 0.0
            switched = True

        # Extra penalty if a switch actually happened
        if switched:
            reward -= self.switch_cost

        # Advance time
        self.idx += 1
        if self.idx >= len(self.data):
            # End of dataset / episode
            self.idx = len(self.data) - 1  # clamp for safe indexing
            done = True
        else:
            delta_t = self.data[self.idx]["timestamp"] - self.data[self.idx - 1]["timestamp"]
            # Just in case there is weirdness in timestamps
            self.time_in_phase += max(delta_t, 0.0)

        return self._get_state(), reward, done, {}
