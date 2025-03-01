import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import csv

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Tanh()  # Ensure action bounds [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

class PPO:
    def __init__(self, num_inputs, num_actions, hidden_size, lr_actor, lr_critic):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(num_inputs, num_actions, hidden_size).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=lr_critic)
        self.num_actions = num_actions
        self.noise_scale = 1.0  # Initial scale of the noise
        self.noise_decay = 0.995  # Decay rate of noise per episode

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_mean, _ = self.actor_critic(state)
        cov_matrix = torch.diag(torch.full((self.num_actions,), 0.1 * self.noise_scale)).to(self.device)
        dist = MultivariateNormal(action_mean, cov_matrix)
        action = dist.sample()
        return action.cpu().detach().numpy()

    def update(self, states, actions, rewards, advantages):
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_means, values = self.actor_critic(states)
        cov_matrix = torch.diag(torch.full((self.num_actions,), 0.1)).to(self.device)
        dist = MultivariateNormal(action_means, cov_matrix)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().mean()

        ratios = torch.exp(log_probs - log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
        critic_loss = F.mse_loss(values.squeeze(), rewards)

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), 1.0)
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), 1.0)
        self.optimizer_critic.step()

env = gym.make("Ant-v4")
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
hidden_size = 64
lr_actor = 0.0003
lr_critic = 0.001
ppo = PPO(num_inputs, num_actions, hidden_size, lr_actor, lr_critic)

max_episodes = 5000
max_steps = 1000

all_episode_rewards = []
average_rewards = []

def save_model(model, episode):
    if episode % 500 == 0:
        torch.save(model.actor_critic.state_dict(), f"model_episode_{episode}.pt")

for episode in range(max_episodes):
    observation, info = env.reset(seed=42)
    states, actions, rewards, dones, next_states = [], [], [], [], []
    episode_reward = 0

    for step in range(max_steps):
        action = ppo.select_action(observation)
        next_state, reward, terminated, truncated, info = env.step(action)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        next_states.append(next_state)
        observation = next_state
        episode_reward += reward

        if terminated or truncated:
            break

    ppo.noise_scale *= ppo.noise_decay  # Reduce noise scale after each episode
    all_episode_rewards.append(episode_reward)
    average_rewards.append(np.mean(all_episode_rewards[-100:]))  # Average of the last 100 episodes

    ppo.update(states, actions, rewards, advantages)
    save_model(ppo, episode)
    print(f"Episode {episode}, Reward: {episode_reward}, Average Reward: {average_rewards[-1]}")

env.close()

# Save episode rewards and average rewards into CSV
with open('episode_rewards.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Episode', 'Reward', 'Average Reward'])
    for i, (reward, avg_reward) in enumerate(zip(all_episode_rewards, average_rewards)):
        writer.writerow([i, reward, avg_reward])

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(all_episode_rewards)
plt.title('Episode Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()
