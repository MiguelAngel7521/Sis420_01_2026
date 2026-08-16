import numpy as np
import matplotlib.pyplot as plt
from veiumtum_env import VeiumtumEnv


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _get_key(self, obs):
        return tuple(obs)

    def _get_q(self, obs):
        key = self._get_key(obs)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_space.n)
        return self.q_table[key]

    def act(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        q = self._get_q(obs)
        return int(np.argmax(q))

    def learn(self, obs, action, reward, next_obs, done):
        q = self._get_q(obs)
        target = reward
        if not done:
            next_q = self._get_q(next_obs)
            target += self.gamma * np.max(next_q)
        q[action] += self.lr * (target - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes=50000):
    env = VeiumtumEnv(natural=False, sab=False)
    agent = QLearningAgent(env)

    rewards = []
    wins = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

        if (ep + 1) % 5000 == 0:
            avg_reward = np.mean(rewards[-1000:])
            win_rate = wins / (ep + 1)
            print(f"Episodio {ep+1}: avg_reward={avg_reward:.3f}, "
                  f"win_rate={win_rate:.3f}, epsilon={agent.epsilon:.4f}, "
                  f"q_table_size={len(agent.q_table)}")

    env.close()
    return agent, rewards


def evaluate(agent, episodes=10000):
    env = VeiumtumEnv(natural=False, sab=False)
    wins = 0
    draws = 0
    losses = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            q = agent._get_q(obs)
            action = int(np.argmax(q))
            obs, reward, done, _, _ = env.step(action)
        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1

    print(f"\nEvaluacion ({episodes} episodios):")
    print(f"  Victorias: {wins} ({100*wins/episodes:.1f}%)")
    print(f"  Empates:   {draws} ({100*draws/episodes:.1f}%)")
    print(f"  Derrotas:  {losses} ({100*losses/episodes:.1f}%)")
    env.close()


def plot_rewards(rewards):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label="Recompensa por episodio")
    N = 1000
    cumsum = np.cumsum(rewards)
    avg = (cumsum[N-1:] - cumsum[:-N+1]) / N if len(rewards) >= N else rewards
    plt.plot(avg, label=f"Media movil ({N} ep.)", color="red")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.title("Evolucion del aprendizaje")

    plt.subplot(1, 2, 2)
    wins = np.cumsum(np.array(rewards) > 0)
    rates = wins / np.arange(1, len(rewards) + 1)
    plt.plot(rates, label="Tasa de victorias")
    plt.xlabel("Episodio")
    plt.ylabel("Win rate")
    plt.legend()
    plt.title("Tasa de victorias acumulada")

    plt.tight_layout()
    plt.savefig("veiumtum_learning.png")
    plt.show()


if __name__ == "__main__":
    print("=== ENTRENANDO AGENTE Q-LEARNING PARA VEiumTUM ===")
    agent, rewards = train(episodes=50000)
    evaluate(agent, episodes=10000)
    plot_rewards(rewards)
