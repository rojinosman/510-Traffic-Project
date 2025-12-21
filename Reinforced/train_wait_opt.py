# train_wait_opt.py
import numpy as np
from syntrac_sim_env import SynTraCSimEnv
from q_agent import QAgent, Discretizer

def feature_ranges():
    # Reasonable fixed ranges to build bins
    return {
        "phase": (0.0, 1.0),
        "queue": (0.0, 80.0),
        "wait": (0.0, 30.0),
        "thru": (0.0, 10.0),
    }

def train(env, agent, episodes=100, max_steps=None, log_every=10):
    hist = []
    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_ret, steps = 0.0, 0
        while True:
            a = agent.act(s)
            s_next, r, done, _ = env.step(a)
            agent.observe(s, a, r, s_next, done)
            s = s_next
            ep_ret += r
            steps += 1
            if done or (max_steps is not None and steps >= max_steps):
                break
        agent.decay()
        hist.append(ep_ret)
        if (ep % log_every) == 0 or ep == 1:
            print(f"Episode {ep}/{episodes} | return={ep_ret:8.3f} | epsilon={agent.epsilon:.3f}")
    return hist

def eval_policy(env, policy_fn, episodes=5, max_steps=None, label=""):
    avg_waits, avg_queues, avg_thrus = [], [], []
    for _ in range(episodes):
        s = env.reset()
        waits, queues, thrus = [], [], []
        steps = 0
        while True:
            waits.append(float(s[2]))
            queues.append(float(s[1]))
            thrus.append(float(s[3]))
            a = policy_fn(s)
            s, _, done, _ = env.step(a)
            steps += 1
            if done or (max_steps is not None and steps >= max_steps):
                break
        avg_waits.append(np.mean(waits))
        avg_queues.append(np.mean(queues))
        avg_thrus.append(np.mean(thrus))
    print(f"[Eval {label}] mean wait={np.mean(avg_waits):.3f} | mean queue={np.mean(avg_queues):.3f} | mean throughput={np.mean(avg_thrus):.3f}")

if __name__ == "__main__":
    # Build env focused on minimizing waiting time
    env = SynTraCSimEnv(
        horizon=5000, seed=0,
        lam_arrival=3.0,
        svc_green=6.0, svc_red=1.0,
        dwell_min=6,
        w_wait=1.0,          # primary: minimize wait
        w_queue=0.08,        # secondary: discourage long queues
        w_throughput=0.2,    # small positive incentive for flow
        w_switch=0.3,        # discourage excessive switching
        wait_smooth=0.92
    )

    # Discretization tuned to wait optimization
    ranges = feature_ranges()
    bins_per_feature = {
        "phase": 2,
        "queue": 16,
        "wait": 16,
        "thru": 10,
    }
    disc = Discretizer(ranges, bins_per_feature)

    agent = QAgent(
        action_space_n=env.action_space_n,
        discretizer=disc,
        alpha=0.15,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.992,
        seed=0,
    )

    # Baseline eval: "hold" policy (never toggle)
    eval_policy(env, policy_fn=lambda s: 0, label="baseline (hold)")

    print("\n--- Training (minimize waiting time) ---")
    train(env, agent, episodes=120, max_steps=None, log_every=10)

    print("\n--- Evaluation (greedy policy) ---")
    eval_policy(env, policy_fn=lambda s: agent.act(s), label="learned (greedy)")
