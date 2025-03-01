import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from datetime import datetime

ENV_NAME = 'Ant-v4'

# Get environment information
env = gym.make(ENV_NAME, render_mode="human")
D = env.observation_space.shape[0]  # Input dimension
K = env.action_space.shape[0]       # Output dimension

### Neural Network

# Hyperparameters
M = 300  # Number of neurons in the hidden layer
action_max = env.action_space.high[0]

def relu(x):
    return x * (x > 0)

class ANN:
    def __init__(self, D, M, K, f=relu):
        self.D = D
        self.M = M
        self.K = K
        self.f = f
    def getParam_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        }
    def init(self):
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) / np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M)
        self.b2 = np.zeros(K)

    def forward(self, X):
        Z = self.f(X.dot(self.W1) + self.b1)
        return np.tanh(Z.dot(self.W2) + self.b2) * action_max

    def sampleAct(self, x):
        # Ensure that x is of the correct shape before flattening
        if isinstance(x, (list, tuple)):
            x = np.array(x[0])
        elif not isinstance(x, np.ndarray):
            x = np.array([x[0]])
        X = x.flatten()
        Y = self.forward(X)
        return Y  # No need to flatten the action vector

    def getParam(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def setParams(self, params):
        D, M, K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M:D * M + M]
        self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]

def evolutionStrategy(
    f,
    population_size,
    sigma,
    lr,
    initial_params,
    num_iters):

 
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)

    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)

        R = np.zeros(population_size)
        for j in range(population_size):
            params_try = params + sigma * N[j]
            R[j] = f(params_try)

        m = R.mean()
        s = R.std()
        if s == 0:
            print("Skipping")
            continue

        A = (R - m) / s
        reward_per_iteration[t] = m
        params = params + lr / (population_size * sigma) * np.dot(N.T, A)

        print("Iter:", t, "Avg Reward: %.3f" % m, "Max:", R.max(), "Duration:", (datetime.now() - t0))

    return params, reward_per_iteration

def rewardFunc(params):
    model = ANN(D, M, K)
    model.setParams(params)

    env = gym.make(ENV_NAME)
  
    episode_reward = 0
    episode_length = 0
    done = False
    state = env.reset(seed=15)
    _=False
    #env.render()
    while not done :
        
       
        
        

        action = model.sampleAct(state)

        state, reward, done, _,info = env.step(action)
        if _ == True:
            break
        
        episode_reward += reward
        episode_length += 1

    env.close()
    return episode_reward
    
def dispRew(params, display):
    model = ANN(D, M, K)
    model.setParams(params)

    env = gym.make(ENV_NAME,render_mode = "human")
  
    episode_reward = 0
    episode_length = 0
    
    state = env.reset()
    env.render()
    done = False
    while not KeyboardInterrupt:
        
        #env.render()
        
        

        action = model.sampleAct(state)

        state, reward, done, _,info = env.step(action)
        

        episode_reward += reward
        episode_length += 1

    env.close()
    return episode_reward

if __name__ == '__main__':
    model = ANN(D, M, K)

    
    try:
        saved_params = np.load('es_mujoco_results.npz', allow_pickle=True)
        W1 = saved_params['W1']
        b1 = saved_params['b1']
        W2 = saved_params['W2']
        b2 = saved_params['b2']
        model.setParams(np.concatenate([W1.flatten(), b1, W2.flatten(), b2]))
        print("Loaded saved parameters successfully!")
    except FileNotFoundError:
        print("No saved parameters found. Initializing parameters randomly.")
        model.init()

   
    params = model.getParam()
    best_params, rewards = evolutionStrategy(
        f=rewardFunc,
        population_size=30,
        sigma=0.1,
        lr=0.03,
        initial_params=params,
        num_iters=1000,
    )
    model.setParams(best_params)
    np.savez('saved_parameters.npz', W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)

    # Play test episode
    dispRew(best_params, display=True)
    print("Test:", rewardFunc(best_params))
