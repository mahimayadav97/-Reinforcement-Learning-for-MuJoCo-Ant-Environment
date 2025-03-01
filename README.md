## Overview
This repository contains implementations and evaluations of various reinforcement learning (RL) algorithms for controlling a 3D quadruped robot in the MuJoCo Ant v4 environment. The algorithms implemented include **DDPG**, **DDPG with Prioritized Experience Replay (PER)**, **SAC**, **TD3**, **PPO**, and **Evolution Strategies (ES)**. The goal is to optimize the robot's movement and maximize forward travel distance using continuous action spaces. 

**Agent Performance Demonstration:**  
[Watch the agent in action here](https://drive.google.com/file/d/129ZlGr8QnOHoiwpESmuiXnWffVl8xfxJ/view?usp=sharing)

---

## Implementations 

### **DDPG and DDPG (PER)**
1. Install the required libraries.
2. Open the file using Jupyter Notebook.
3. Adjust `num_episodes` and other hyperparameters listed before the main loop.
4. Run all cells to start training.
5. After each episode, reward and running reward information will be output.

### **SAC**
1. Install the required libraries.
2. Ensure a CUDA-capable GPU and the appropriate drivers are available.
3. Execute the script in a Python environment to train the SAC agent on the "Ant-v4" environment.

### **TD3**
1. Install the required libraries.
2. Adjust the number of episodes (default: 2000) and other hyperparameters.
3. Run the Python script to start training.

### **PPO**
1. Execute the training script to start training the PPO agent in the "Ant-v4" environment.
2. The script runs for a maximum of 5000 episodes, with each episode having up to 1000 steps.
3. The trained model is saved every 500 episodes.

### **Evolution Strategies (ES)**
1. Install the `numpy` library and the `gymnasium` MuJoCo environment.
2. Run the script to train the agent.
3. To visualize the agent, set `render_mode` to `human`.

---

## Results
The performance of each algorithm is summarized below:

| Algorithm       | Average Reward (2000 Episodes) |
|-----------------|--------------------------------|
| DDPG            | 951.16                        |
| DDPG (PER)      | 812.6                         |
| SAC             | 1488.56                       |
| TD3             | 3902.23                       |
| PPO             | 969.51                        |
| ES              | 1027.03                       |

![Average Reward Plot](https://github.com/mahimayadav97/-Reinforcement-Learning-for-MuJoCo-Ant-Environment/blob/main/images/Result.png)  
*Figure 1: Plot of the rolling average reward of 40 episodes for each algorithm implemented, with a comparison to an agent choosing actions randomly.*

**Key Observations:**
- **TD3** achieved the highest average reward of **3902.23**, outperforming all other algorithms.
- **SAC** demonstrated strong performance with an average reward of **1488.56**.
- **DDPG** and **DDPG (PER)** showed similar performance, with DDPG (PER) stabilizing earlier.
- **PPO** exhibited steady improvement over time, with rewards increasing as training progressed.
- **ES** showed consistent improvement but did not fully converge within the training period.

---

## Discussion

### **DDPG and DDPG (PER)**
- Both DDPG and DDPG (PER) converged to an average reward of just under 1000.
- DDPG (PER) stabilized earlier, suggesting faster convergence, but did not significantly outperform standard DDPG.
- The ant policy converged to small actions to maximize `healthy_reward` while minimizing `ctrl_cost`.

### **SAC**
- Initial poor performance reflected the exploration phase.
- By episode 100, the model stabilized, achieving an average reward of **1488.56** by episode 2000.
- Fluctuations were observed due to large actions and episodes where the ant got stuck upside down.

### **TD3**
- Achieved the highest average reward of **3902.23**.
- The reward curve fluctuated significantly due to occasional faulty actions causing the ant to flip over.
- Despite instability, TD3 performed the best among all algorithms.

### **PPO**
- Rewards increased steadily over time, indicating effective learning and adaptation.
- Noise in actions decreased over episodes, transitioning from exploration to exploitation.
- Early variability in rewards smoothed out as training progressed.

### **ES**
- Training completed in 8 hours for 3000 episodes, producing an average reward between 900-1000.
- The algorithm did not fully converge, as rewards continued to increase.
- Larger population sizes (e.g., 30 agents) produced higher rewards but required longer compute times.

---

## Comparison to Baselines
The results are compared to those reported by Fujimoto et al. [9]:

| Algorithm       | Implementation     | Fujimoto et al. [9] |
|-----------------|--------------------|---------------------|
| DDPG            | 951.16             | 1005.3              |
| DDPG (PER)      | 812.6              | -                   |
| SAC             | 1488.56            | 655.35             |
| TD3             | 3902.23            | 4372.44            |
| PPO             | 969.51             | 1083.2             |
| ES              | 1027.03            | -                   |

**Key Points:**
- Results align closely with Fujimoto et al., with larger differences observed for SAC and TD3.
- TD3 exhibited high volatility in rewards, consistent with the high standard deviation reported by Fujimoto et al.
---

## Dependencies
- Python 3.x
- Libraries: `numpy`, `gymnasium`, `torch`, `tensorflow`, `keras`, `mujoco`
- CUDA-capable GPU (recommended for SAC and other GPU-accelerated algorithms)

---

## References
1. [Keras DDPG Pendulum Example](https://keras.io/examples/rl/ddpg_pendulum/)
2. [Prioritized Experience Replay Paper](https://arxiv.org/abs/1511.05952)
3. [Haarnoja's SAC Implementation](https://github.com/haarnoja/sac)
4. [SLM-Lab Pull Request](https://github.com/kengz/SLM-Lab/pull/398)
5. [Pranz24's PyTorch SAC](https://github.com/pranz24/pytorch-soft-actor-critic)
6. [AI Walkers PPO PyTorch](https://github.com/iamvigneshwars/ai-walkers-ppo-pytorch)
7. [Fengredrum's PPO PyTorch](https://github.com/fengredrum/ppo-pytorch)
8. [Lazy Programmer's ES MuJoCo Example](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl3/es_mujoco.py)
9. Scott Fujimoto, Herke van Hoof, and David Meger. Addressing Function Approximation Error in Actor-Critic Methods, October 2018.

---

**Note:** Ensure the necessary MuJoCo license and dependencies are installed to run the environment.
