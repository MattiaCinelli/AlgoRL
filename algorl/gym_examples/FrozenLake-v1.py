#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import gym
from pathlib import Path

# !pip install Box2D
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor 

# Local imports
from utils import eval_env_random_actions
from logs import logging

logger = logging.getLogger("FrozenLake")
# In[1]:
# 1. Set up
"""
Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H) by walking over
the Frozen(F) lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake.

### Action Space
The agent takes a 1-element vector for actions.
The action space is `(dir)`, where `dir` decides direction to move in which can be:
- 0: LEFT
- 1: DOWN
- 2: RIGHT
- 3: UP

### Observation Space
The observation is a value representing the agent's current position as
current_row * nrows + current_col (where both the row and col start at 0).
For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
The number of possible observations is dependent on the size of the map.
For example, the 4x4 map has 16 possible observations.

### Rewards
Reward schedule:
- Reach goal(G): +1
- Reach hole(H): 0
- Reach frozen(F): 0
"""
# Create folder to save models
directory_path = 'models'
Path(directory_path).mkdir(parents=True, exist_ok=True)

# Create environment
env_name = 'FrozenLake-v1'
env = gym.make(env_name)

num_steps = 250_000
model_file_name = Path(directory_path, f'{env_name}_{num_steps}')
logger.info(env.action_space)
logger.info(env.observation_space)

# In[3]:
# 1.1 Test random actions
eval_env_random_actions(env, render=True, logger=logger)

# %%
# 2. Build and Train a model
# for i in range(len(env.P[4][0])):
#     print(env.P[4][0][i])
#     # prob, next_state, reward, done 
# print(len(env.P[4][0]))
# print(env.nS)
env.
# %%
def value_iteration(env, max_iterations=1_000_000, lmbda=0.9):
    stateValue = [0 for _ in range(env.nS)]
    newStateValue = stateValue.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
          action_values = []      
          for action in range(env.nA):
            state_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, reward, done = env.P[state][action][i]
                state_action_value = prob * (reward + lmbda*stateValue[next_state])
                state_value += state_action_value
            action_values.append(state_value)      #the value of each action
            best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
            newStateValue[state] = action_values[best_action]  #update the value of the state
        if i > 1000: 
            if sum(stateValue) - sum(newStateValue) < 1e-04:   # if there is negligible difference break the loop
                break
        else:
            stateValue = newStateValue.copy()
    return stateValue 
stateValue = value_iteration(env)
print(stateValue)

# %%
def get_policy(env, stateValue, lmbda=0.9):
    policy = [0 for _ in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + lmbda * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy
policy = get_policy(env, stateValue)
print(policy)
# %%
def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for _ in range(episodes):
        observation = env.reset()
        steps=0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps+=1
            if done and reward == 1:
            # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
            # print("You fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format((misses/episodes) * 100))
    print('----------------------------------------------')
get_score(env, policy)
# In[4]:
# Instantiate the agent
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=0) # CNNPolicy


# In[5]:
# Train the agent
model.learn(total_timesteps = num_steps)
# We want high explained_variance and 

# In[6]:
# # 3 Save and reload
# Save the agent
model.save(model_file_name)
del model  # delete trained model to demonstrate loading


# In[7]:
# Load the trained agent
# NOTE: if you have loading issue, you can pass `logger.info_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, logger.info_system_info=True)
model = PPO.load(model_file_name, env=env)

# In[12]:

# 3. Evaluate
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model,  env , n_eval_episodes=10)
logger.info(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# In[11]:
# Enjoy trained agent
obs = env.reset()
for _ in range(500):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
env.reset()
env.close()


# %%
