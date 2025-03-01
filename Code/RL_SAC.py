import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Mixed Precision Scaler
scaler = GradScaler()

# Actor definition 
class SACActor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(SACActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

# Critic definition 
class SACCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(SACCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net(state_action)

# Normalized Actions Wrapper
class NormalizedActions:
    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return self.env.step(action)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

# Replay Buffer 
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.buffer = {
            'states': np.zeros((capacity, state_dim)),
            'actions': np.zeros((capacity, action_dim)),
            'rewards': np.zeros(capacity),
            'next_states': np.zeros((capacity, state_dim)),
            'dones': np.zeros(capacity)
        }
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        index = self.position % self.capacity
        self.buffer['states'][index] = state
        self.buffer['actions'][index] = action
        self.buffer['rewards'][index] = reward
        self.buffer['next_states'][index] = next_state
        self.buffer['dones'][index] = done
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: torch.FloatTensor(v[idx]).to(self.device) for k, v in self.buffer.items()}

    def __len__(self):
        return self.size    

# SAC algorithm with mixed precision training 
class SAC:
    def __init__(self, num_inputs, num_actions, hidden_size, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = SACActor(num_inputs, num_actions, hidden_size).to(self.device)
        self.critic_1 = SACCritic(num_inputs, num_actions, hidden_size).to(self.device)
        self.critic_2 = SACCritic(num_inputs, num_actions, hidden_size).to(self.device)
        self.target_critic_1 = SACCritic(num_inputs, num_actions, hidden_size).to(self.device)
        self.target_critic_2 = SACCritic(num_inputs, num_actions, hidden_size).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-2)
        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.num_actions = num_actions
        self.tau = 0.005
        self.replay_buffer = ReplayBuffer(capacity=10000, state_dim=num_inputs, action_dim=num_actions, device=self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = samples['states'], samples['actions'], samples['rewards'], samples['next_states'], samples['dones']

        # Target Q-values computation
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q1 = self.target_critic_1(next_states, next_actions).view(-1)
            next_q2 = self.target_critic_2(next_states, next_actions).view(-1)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + 0.99 * (1 - dones) * next_q

        # Critic 1 update
        current_q1 = self.critic_1(states, actions).view(-1)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.optimizer_critic_1.zero_grad()
        scaler.scale(critic_1_loss).backward(retain_graph=True)  # Use retain_graph if needed
        scaler.step(self.optimizer_critic_1)

        # Critic 2 update
        current_q2 = self.critic_2(states, actions).view(-1)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.optimizer_critic_2.zero_grad()
        scaler.scale(critic_2_loss).backward()
        scaler.step(self.optimizer_critic_2)

        # Ensuring no gradient computations for critic updates affect actor updates
        for param in self.critic_1.parameters():
            param.requires_grad = False
        for param in self.critic_2.parameters():
            param.requires_grad = False

        # Actor update
        actor_loss = -self.critic_1(states, self.actor(states)).mean()
        self.optimizer_actor.zero_grad()
        scaler.scale(actor_loss).backward()
        scaler.step(self.optimizer_actor)

        # Re-enable gradients for critic parameters
        for param in self.critic_1.parameters():
            param.requires_grad = True
        for param in self.critic_2.parameters():
            param.requires_grad = True

        # Update the scale for mixed precision
        scaler.update()

        # Soft update the target networks
        with torch.no_grad():
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Initialize environment, SAC, and training parameters
env = NormalizedActions(gym.make("Ant-v4"))
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
hidden_size = 256
lr = 0.0003
sac = SAC(num_inputs, num_actions, hidden_size, lr)

# Initialize data tracking
episode_rewards = []
average_episode_rewards = []

# Training loop
max_episodes = 2000
max_steps = 600
batch_size = 64
writer = SummaryWriter()

for episode in range(max_episodes):
    observation = env.reset(seed=42)
    total_episode_reward = 0
    for step in range(max_steps):
        action = sac.select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        sac.replay_buffer.push(observation, action, reward, next_observation, terminated or truncated)
        sac.update(batch_size)
        
        writer.add_scalar('Reward/step', reward, episode * max_steps + step)
        
        observation = next_observation
        total_episode_reward += reward
        if terminated or truncated:
            break

    episode_rewards.append(total_episode_reward)
    if len(episode_rewards) >= 40:
        new_average = np.mean(episode_rewards[-40:])
        average_episode_rewards.append(new_average)
    else:
        average_episode_rewards.append(np.mean(episode_rewards))

    writer.add_scalar('Reward/episode', total_episode_reward, episode)
    print(f"Episode {episode}, Reward: {total_episode_reward}, Average Reward: {average_episode_rewards[-1]}")

    # Save model every 500 episodes
    if episode % 500 == 0:
        torch.save(sac.actor.state_dict(), f'sac_actor_{episode}.pth')
        torch.save(sac.critic_1.state_dict(), f'sac_critic1_{episode}.pth')
        torch.save(sac.critic_2.state_dict(), f'sac_critic2_{episode}.pth')

# Save rewards to CSV
df_rewards = pd.DataFrame({
    'Episode': range(max_episodes),
    'Reward': episode_rewards,
    'Average Reward': average_episode_rewards
})
df_rewards.to_csv('episode_rewards.csv', index=False)

# Plotting the graph
plt.figure(figsize=(10, 5))
plt.plot(df_rewards['Episode'], df_rewards['Reward'], label='Reward per Episode')
plt.plot(df_rewards['Episode'], df_rewards['Average Reward'], label='Average Reward (last 40)', color='red')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards and Average Rewards Over Episodes')
plt.legend()
plt.savefig('rewards_plot.png')
plt.show()

env.env.close()
writer.close()