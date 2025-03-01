from __future__ import annotations

import sys
import random
import time
from random import sample
from IPython import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras

import gymnasium as gym
import mujoco
import glfw
import json

from collections import deque

from keras.optimizers import Adam
from keras.losses import MSE
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam



class Replay_Memory(object):
    def __init__(self, batch_size=128, maxlen=10000):
        self.batch_size = batch_size
        self.memory = deque([], maxlen=maxlen)
        # self.count = 0

    def __len__(self):
        return len(self.memory)

    def update(self, s, a, r, s_, done):
        e = (s, a, r, s_, done)
        self.memory.append(e)
        # self.count = self.count+1
        
    def sample(self):
        samples = random.sample(self.memory, self.batch_size)

        s, a, r, s_, done = zip(*samples)
        s = np.array(s)
        a = np.array(a)
        r = np.array(r)
        s_ = np.array(s_)
        done = np.array(done)

        r = r.reshape(-1,1)
        done = done.reshape(-1,1)

        s = tf.convert_to_tensor(s, dtype = tf.float32)
        a = tf.convert_to_tensor(a, dtype = tf.float32)
        r = tf.convert_to_tensor(r, dtype = tf.float32)
        s_ = tf.convert_to_tensor(s_, dtype = tf.float32)
        done = tf.convert_to_tensor(done, dtype = tf.float32)

        return s, a, r, s_, done
    


class TD3_Agent(object):
    def __init__(self, env,
                 tau=0.005,
                 gamma=0.99,
                 maxlen=100000,
                 batch_size=128,
                 noise_std=0.1,
                 noise_clip=0.5,
                 policy_delay=2):
        
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.memory = Replay_Memory(batch_size, maxlen)

        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_upper_bound = env.action_space.high[0]
        self.action_lower_bound = env.action_space.low[0]
        
        self.actor_eval = self.build_actor_net()
        self.actor_target = self.build_actor_net()
        self.critic_eval_1 = self.build_critic_net()
        self.critic_target_1 = self.build_critic_net()
        self.critic_eval_2 = self.build_critic_net()
        self.critic_target_2 = self.build_critic_net()

        self.actor_target.set_weights(self.actor_eval.get_weights())
        self.critic_target_1.set_weights(self.critic_eval_1.get_weights())
        self.critic_target_2.set_weights(self.critic_eval_2.get_weights())

        self.a_opt = Adam(1e-4)
        self.c_opt_1 = Adam(1e-4)
        self.c_opt_2 = Adam(1e-4)


    def build_actor_net(self):
        last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        inputs = Input(shape=(self.state_dim,))
        x = Dense(256, activation="relu")(inputs)
        x = Dense(256, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        outputs = Dense(self.action_dim, activation="tanh", kernel_initializer=last_init)(x)

        outputs = outputs * self.action_upper_bound
        model = keras.Model(inputs, outputs)
        return model


    def build_critic_net(self):
        state = Input(shape=(self.state_dim,))
        action = Input(shape=(self.action_dim,))
        inputs = Concatenate()([state, action])

        x = Dense(256, activation="relu")(inputs)
        x = Dense(256, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        outputs = Dense(1)(x)

        model = keras.Model([state, action], outputs)
        return model


    def update_target(self):

        def compute_new_target_w(target_w, origin_w):
            for i in range(len(target_w)):
                target_w[i] = origin_w[i] * self.tau + target_w[i] * (1 - self.tau)
            return target_w
            
        target_w = self.actor_target.get_weights()
        origin_w = self.actor_eval.get_weights()
        self.actor_target.set_weights(compute_new_target_w(target_w, origin_w))

        target_w = self.critic_target_1.get_weights()
        origin_w = self.critic_eval_1.get_weights()
        self.critic_target_1.set_weights(compute_new_target_w(target_w, origin_w))

        target_w = self.critic_target_2.get_weights()
        origin_w = self.critic_eval_2.get_weights()
        self.critic_target_2.set_weights(compute_new_target_w(target_w, origin_w))


    def policy(self, state):
        state = state.reshape(1,-1)
        state = tf.convert_to_tensor(state, dtype = tf.float32)
        action = self.actor_eval(state)
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=self.action_dim)
        action = action.numpy() + noise
        action_bounded = np.clip(action, self.action_lower_bound, self.action_upper_bound)

        action_shape = self.env.action_space.shape
        action_bounded = action_bounded.reshape(action_shape)
        return action_bounded


    @tf.function
    def train_critic(self, s, a, r, s_n, done_):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=self.action_dim)
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            noise = tf.cast(noise, dtype=tf.float32)

            a_target = self.actor_target(s_n, training=True)
            a_target = a_target + noise
            a_target = tf.clip_by_value(a_target, self.action_lower_bound, self.action_upper_bound)

            q_t_1 = self.critic_target_1([s_n, a_target], training=True)
            q_t_2 = self.critic_target_2([s_n, a_target], training=True)

            q_t = tf.math.minimum(q_t_1, q_t_2)
            y = r + self.gamma * q_t
            
            critic_v_1 = self.critic_eval_1([s, a], training=True)
            critic_loss_1 = tf.math.reduce_mean(keras.losses.MSE(y, critic_v_1))
            critic_v_2 = self.critic_eval_2([s, a], training=True)
            critic_loss_2 = tf.math.reduce_mean(keras.losses.MSE(y, critic_v_2))

        critic_grad_1 = tape1.gradient(critic_loss_1, self.critic_eval_1.trainable_variables)
        self.c_opt_1.apply_gradients(zip(critic_grad_1, self.critic_eval_1.trainable_variables))
        critic_grad_2 = tape2.gradient(critic_loss_2, self.critic_eval_2.trainable_variables)
        self.c_opt_2.apply_gradients(zip(critic_grad_2, self.critic_eval_2.trainable_variables))


    @tf.function
    def train_actor(self, s):
        with tf.GradientTape() as tape:
            a_eval = self.actor_eval(s, training=True)
            critic_v_eval = self.critic_eval_1([s, a_eval], training=True)
            actor_loss = -tf.math.reduce_mean(critic_v_eval)

        actor_grad = tape.gradient(actor_loss, self.actor_eval.trainable_variables)
        self.a_opt.apply_gradients(zip(actor_grad, self.actor_eval.trainable_variables))


    def save_network_and_rewards(self, num_eps, ep_reward, avg_reward):
        num_eps = str(num_eps)
        print("\nMESSAGE: save the network and rewards.\n")
        with open('./results/rewards_json/td3/ant_rewards_'+num_eps+'.json', 'w') as file:
            json.dump([{'ep_reward_list': ep, 'avg_reward': avg} 
                    for ep, avg in zip(ep_reward, avg_reward)], file)

        self.actor_eval.save_weights("./results/weights_h5/td3/actor_eval_weight_"+num_eps+".h5")
        self.critic_eval_1.save_weights("./results/weights_h5/td3/critic_eval_1_weight_"+num_eps+".h5")
        self.critic_eval_2.save_weights("./results/weights_h5/td3/critic_eval_2_weight_"+num_eps+".h5")
        self.actor_target.save_weights("./results/weights_h5/td3/actor_target_weight_"+num_eps+".h5")
        self.critic_target_1.save_weights("./results/weights_h5/td3/critic_target_1_weight_"+num_eps+".h5")
        self.critic_target_2.save_weights("./results/weights_h5/td3/critic_target_2_weight_"+num_eps+".h5")


    def td3(self, num_episodes):
        ep_returns = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            reward_total = 0

            count = 0
            while True:
                # Select action
                action = self.policy(state)

                # Execute action, get info
                state_next, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                reward_total += reward

                # Store info into replay buffer
                self.memory.update(state, action, reward, state_next, done)

                if len(self.memory) >= self.batch_size:
                    # Sample memory from buffer, get minibatch
                    s, a, r, s_, done_ = self.memory.sample()
                        # Training critic
                    self.train_critic(s, a, r, s_, done_)
                        
                    if count%self.policy_delay:
                        # Training actor
                        self.train_actor(s)
                        # Soft update network weight
                        self.update_target()
                
                # Reset env if terminal
                if done:
                    break

                state = state_next
                count +=1

            ep_returns.append(reward_total)
            avg_reward = np.mean(ep_returns[-10:])
            print(f"At Episode: {ep}, Esp Reward: {round(reward_total,4)}, Avg Reward: {round(avg_reward,4)}")

        # self.save_network_and_rewards(num_episodes, reward_total, avg_reward)
        
        return ep_returns
    



def plot_agent_results(num_episodes, agent_rewards = None) :

    # Modified Agent
    if (agent_rewards is not None):
        num_agents_mod = len(agent_rewards)
        
        # Average Returns Lists
        modified_agent_average_episode_rewards = []
        for episode in range(0, num_episodes) :
            reward = 0
            for agent in range(0, num_agents_mod) :
                reward += agent_rewards[agent][episode]
            modified_agent_average_episode_rewards.append(reward / num_agents_mod)
            
    
    # Uncropped Learning Curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_episodes), modified_agent_average_episode_rewards, label = "Agent")
    plt.title("Racetrack Average Learning Curve")
    plt.xlabel("Episodes Played")
    plt.ylabel("Average Return")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    env = gym.make('Ant-v4')#, render_mode="human")

    num_agents = 1
    num_episodes = 2000

    td3_agents = []
    td3_agent_rewards = []

    for agent in range(num_agents) :
        episode_rewards = []
        this_agent = TD3_Agent(env, batch_size=128, maxlen=100000)
        episode_rewards = this_agent.td3(num_episodes)

        td3_agents.append(this_agent)
        td3_agent_rewards.append(episode_rewards)

    env.close()
    glfw.terminate()

    plot_agent_results(num_episodes, td3_agent_rewards)