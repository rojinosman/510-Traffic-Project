from syntrac_env import SynTraCEnv
from q_agent import QAgent
import numpy as np
import random


# ----------------------------------------------------------------------
# Utility: run a single episode with a given policy
# ----------------------------------------------------------------------
def run_episode(env, policy_fn):
    """
    policy_fn: function(state) -> action
    Returns a dict with metrics.
    """
    state = env.reset()
    total_reward = 0.0
    total_wait = 0.0
    total_queue = 0.0
    steps = 0
    done = False

    while not done:
        action = policy_fn(state)
        state, reward, done, _ = env.step(action)

        rec = env.data[env.idx]  # current traffic record
        total_wait += rec["waiting_time"]
        total_queue += rec["queue_length"]
        total_reward += reward
        steps += 1

    return {
        "total_reward": total_reward,
        "avg_reward": total_reward / steps,
        "avg_wait": total_wait / steps,
        "avg_queue": total_queue / steps,
        "steps": steps,
    }


# ----------------------------------------------------------------------
# Training loop for RL agent
# ----------------------------------------------------------------------
def train(env, agent, episodes=100):
    episode_rewards = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        print(
            f"Episode {ep}/{episodes} "
            f"- Total reward: {total_reward:.2f} "
            f"- Epsilon: {agent.eps:.3f}"
        )

    return episode_rewards


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def evaluate_agent(env, agent, episodes=5):
    """Evaluate the trained RL agent using a greedy policy."""
    original_eps = agent.eps
    agent.eps = 0.0  # force greedy during evaluation

    def rl_policy(state):
        # Greedy over learned Q-values; fall back to action 0 if unseen
        if state in agent.Q:
            return int(np.argmax(agent.Q[state]))
        return 0

    metrics = [run_episode(env, rl_policy) for _ in range(episodes)]
    agent.eps = original_eps

    print("\n=== RL Agent Evaluation (greedy policy) ===")
    _print_metrics(metrics)


def evaluate_baselines(env, episodes=5):
    """Evaluate a couple of simple non-RL baselines."""

    def never_switch_policy(state):
        # Always keep the current phase
        return 0

    def random_policy(state):
        # Randomly choose keep/switch
        return np.random.randint(0, 2)

    baselines = {
        "Never-switch baseline": never_switch_policy,
        "Random baseline": random_policy,
    }

    for name, policy in baselines.items():
        metrics = [run_episode(env, policy) for _ in range(episodes)]
        print(f"\n=== {name} ===")
        _print_metrics(metrics)


def _print_metrics(metrics_list):
    avg_reward = np.mean([m["avg_reward"] for m in metrics_list])
    avg_wait = np.mean([m["avg_wait"] for m in metrics_list])
    avg_queue = np.mean([m["avg_queue"] for m in metrics_list])
    steps = np.mean([m["steps"] for m in metrics_list])

    print(f"Episodes:           {len(metrics_list)}")
    print(f"Avg steps/episode:  {steps:.1f}")
    print(f"Avg reward/step:    {avg_reward:.3f}")
    print(f"Avg waiting time:   {avg_wait:.3f} seconds")
    print(f"Avg queue length:   {avg_queue:.3f} vehicles")


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # For reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create environment & agent
    env = SynTraCEnv("syntrac_merged.csv")
    agent = QAgent(actions=env.n_actions)

    print("\n--- Training RL Agent ---")
    train(env, agent, episodes=200)

    print("\n--- Evaluating Trained RL Agent ---")
    evaluate_agent(env, agent, episodes=5)

    print("\n--- Evaluating Baseline Controllers ---")
    evaluate_baselines(env, episodes=5)
