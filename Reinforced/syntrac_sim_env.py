# syntrac_sim_env.py
import numpy as np

class SynTraCSimEnv:
    """
    Minimal interactive traffic-signal simulator.

    State = [phase, queue_len, avg_wait, throughput]
      - phase: 0 = red (for the target approach), 1 = green
      - queue_len: number of cars waiting
      - avg_wait: a smoothed proxy for average waiting time (seconds)
      - throughput: cars served this step

    Action:
      0 = hold current phase
      1 = toggle phase (if dwell timer allows)

    Episode ends at a fixed horizon.

    Goal: minimize waiting time (and queue). Reward defaults to:
        r = - (w_wait * avg_wait + w_queue * queue_len)
            + w_thru * throughput
            - w_switch * 1{toggled}
    """

    def __init__(
        self,
        horizon=5000,
        seed=0,
        lam_arrival=3.0,     # mean arrivals per step (Poisson)
        svc_green=6.0,       # cars served when green
        svc_red=1.0,         # leakage when red (turning, trickle, etc.)
        dwell_min=5,         # min dwell (steps) before a legal toggle to reduce flicker
        w_wait=1.0,
        w_queue=0.08,
        w_throughput=0.2,
        w_switch=0.3,
        wait_smooth=0.9      # smoothing for avg_wait (0.9 => heavy smoothing)
    ):
        self.rng = np.random.default_rng(seed)
        self.horizon = int(horizon)
        self.lam = float(lam_arrival)
        self.svc_g = float(svc_green)
        self.svc_r = float(svc_red)

        # reward config
        self.w_wait = float(w_wait)
        self.w_queue = float(w_queue)
        self.w_thru = float(w_throughput)
        self.w_switch = float(w_switch)
        self.wait_smooth = float(wait_smooth)

        # phase flicker control
        self.dwell_min = int(dwell_min)

        # internals
        self.t = 0
        self.phase = 0
        self.queue = 0.0
        self.avg_wait = 0.0
        self.throughput = 0.0
        self._dwell = 0      # time spent in current phase

        self.action_space_n = 2

    def reset(self):
        self.t = 0
        self.phase = 0    # start red
        self.queue = float(self.rng.integers(10, 40))
        self.avg_wait = float(self.rng.uniform(2.0, 6.0))
        self.throughput = 0.0
        self._dwell = 0
        return self._obs()

    def _obs(self):
        return np.array(
            [self.phase, self.queue, self.avg_wait, self.throughput],
            dtype=np.float32
        )

    def step(self, action: int):
        toggled = False
        # Enforce dwell time (min-green/red); only allow toggle if _dwell >= dwell_min
        if action == 1 and self._dwell >= self.dwell_min:
            self.phase = 1 - self.phase
            self._dwell = 0
            toggled = True

        # arrivals this step
        arrivals = self.rng.poisson(self.lam)

        # service capacity depends on phase
        svc = self.svc_g if self.phase == 1 else self.svc_r
        served = float(min(self.queue, svc))
        self.queue = max(0.0, self.queue + float(arrivals) - served)

        # update avg wait (smoothed). More arrivals & red increase avg waiting.
        wait_increment = 0.15 * arrivals + (0.25 if self.phase == 0 else -0.25)
        wait_increment = np.clip(wait_increment, -1.0, 1.0)
        self.avg_wait = max(
            0.0,
            self.wait_smooth * self.avg_wait + (1 - self.wait_smooth) * (self.avg_wait + wait_increment)
        )
        self.avg_wait = min(30.0, self.avg_wait)

        # throughput for this step
        self.throughput = served

        # reward focuses on minimizing avg_wait (primary), and queue (secondary)
        reward = (
            - self.w_wait * self.avg_wait
            - self.w_queue * self.queue
            + self.w_thru * self.throughput
            - (self.w_switch if toggled else 0.0)
        )

        self.t += 1
        self._dwell += 1
        done = self.t >= self.horizon
        return self._obs(), float(reward), done, {}
