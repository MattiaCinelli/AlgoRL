#!/usr/bin/env python
# coding: utf-8

# # Pendulum-v0
# The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright.
# https://gym.openai.com/envs/Pendulum-v0/
# 
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# 
# ### Action Space
# The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum. Min = -2.0, Max = 2.0
# 
# ### Observation Space
# The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free end and its angular velocity.
# x = cos(theta) (-1.0, 1.0), y = sin(angle) (-1.0, 1.0), Angular Velocity (-8.0, 8.0)
# 
# ### Starting State
# The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.
# 
# ### Episode Termination
# The episode terminates at 200 time steps.
# 
# ## Loading

# In[1]:


# Standard Libraries
import os
from pathlib import Path

# Third party libraries
import gym
import numpy as np
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.ppo.policies import MlpPolicy #  MlpPolicy because the observation of the CartPole task is a feature vector, not images.
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.env_util import make_vec_env

# Local imports
from utils import file_exists, evaluate_model, eval_env_random_actions


# ## Initialization

# In[2]:


env_name = 'Pendulum-v0'
# env = gym.make(env_name)
env = make_vec_env(env_name, n_envs=4, seed=0)

# Create folder to save models
directory_path = 'models'
Path(directory_path).mkdir(parents=True, exist_ok=True)


# In[3]:

print(env.action_space)
print(env.observation_space)

# In[4]:
eval_env_random_actions(env)


# ## Create model

# In[5]:


model = SAC('MlpPolicy', env, train_freq=1, gradient_steps=2, verbose=0)
evaluate_model(model, num_episodes=100)


# Training
# In[6]:
num_steps = 100_000
model.learn(num_steps)

# Evaluate
### Rewards
# The minimum reward that can be obtained is -16.2736044, while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

## Save 
# In[7]:


model_file_name = Path(directory_path, env_name + '_' + str(num_steps))
model.save(model_file_name)

# In[8]:
evaluate_model(model, num_episodes=100)