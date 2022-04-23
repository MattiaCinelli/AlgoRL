#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from pathlib import Path

# !pip install Box2D
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor 

# Local imports
from utils import file_exists, evaluate_model, eval_env_random_actions


# # 1. Set up

# In[2]:


# Create folder to save models
directory_path = 'models'
Path(directory_path).mkdir(parents=True, exist_ok=True)

# Create environment
env_name = 'LunarLander-v2'
env = gym.make(env_name)

num_steps = 250_000
model_file_name = Path(directory_path, f'{env_name}_{num_steps}')
print(env.action_space)
# 0- Do nothing
# 1- Fire left engine
# 2- Fire down engine
# 3- Fire right engine
print(env.observation_space)


# ## 1.1 Test random actions

# In[3]:


eval_env_random_actions(env, render=False)


# # 2. Build and Train a model

# In[4]:


# Instantiate the agent
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=0) # CNNPolicy


# In[5]:


# Train the agent
model.learn(total_timesteps = num_steps)
# We want high explained_variance and 


# # 3 Save and reload

# In[6]:


# Save the agent
model.save(model_file_name)
del model  # delete trained model to demonstrate loading


# ## Load the trained agent

# In[7]:


# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load(model_file_name, env=env)


# # 3. Evaluate

# In[12]:


# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model,  env , n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# In[11]:


# Enjoy trained agent
obs = env.reset()
for _ in range(500):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
env.reset()
env.close()
