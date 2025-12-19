# train_syntrac.py
import os
import numpy as np
from syntrac_env import SynTraCEnv
from q_agent import QAgent, Discretizer

def train(env, agent, episodes=50, max_steps=None, save_every=10, save_path="q_table.npz"):
    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_return = 0.0
        steps = 0
        while True:
            a = agent.act(s)
            s_next, r, done, _ = env.step(a)
            agent.observe(s, a, r, s_next, done)

            s = s_next
            ep_return += r
            steps += 1

            if done or (max_steps is not None and steps >= max_steps):
                break

        agent.decay()
        print(f"Episode {ep}/{episodes}  |  return={ep_return:8.3f}  steps={steps:5d}  epsilon={agent.epsilon:.3f}")

        if save_every and (ep % save_every == 0):
            agent.save(save_path)

def evaluate(env, agent, episodes=3, max_steps=None):
    # Greedy during evaluation
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    returns = []
    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_return = 0.0
        steps = 0
        while True:
            a = agent.act(s)
            s, r, done, _ = env.step(a)
            ep_return += r
            steps += 1
            if done or (max_steps is not None and steps >= max_steps):
                break
        returns.append(ep_return)
        print(f"[Eval] Episode {ep}: return={ep_return:.3f} steps={steps}")
    agent.epsilon = old_eps
    print(f"[Eval] mean return={np.mean(returns):.3f}")

if __name__ == "__main__":
    # Build env
    env = SynTraCEnv("syntrac_merged.csv")

    # Discretizer: choose bins per feature (tune if you want)
    ranges = env.feature_ranges()
    bins_per_feature = {
        "signal_phase": 2,   # 0/1
        "queue_length": 10,  # 0..80
        "waiting_time": 8,   # 0..15
        "throughput": 8,     # 0..10
    }
    disc = Discretizer(ranges, bins_per_feature)

    # Agent
    agent = QAgent(
        action_space_n=env.action_space_n,
        discretizer=disc,
        alpha=0.15,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.98,
        seed=0,
    )

    # Optional: load existing table
    if os.path.exists("q_table.npz"):
        try:
            agent.load("q_table.npz")
        except Exception as e:
            print("Load failed, starting fresh:", e)

    print("\n--- Training RL Agent ---")
    train(env, agent, episodes=50, max_steps=None, save_every=10, save_path="q_table.npz")

    print("\n--- Evaluation (greedy) ---")
    evaluate(env, agent, episodes=3, max_steps=None)
