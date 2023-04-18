# %% 
import gym
from pathlib import Path

# !pip install Box2D
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# Local imports
from utils import eval_env_random_actions
from logs import logging

logger = logging.getLogger("CarRacing")


# %% Create folder to save models
directory_path = 'models'
Path(directory_path).mkdir(parents=True, exist_ok=True)

# Create environment
env_name = 'CarRacing-v1'
env = gym.make(env_name)

num_steps = 250#_000
model_file_name = Path(directory_path, f'{env_name}_{num_steps}')
# There are 3 actions: steering (-1 is full left, +1 is full right), gas, and breaking.
logger.info(env.action_space)
# State consists of 96x96 pixels
logger.info(env.observation_space)
# Rewards The reward is -0.1 every frame and +1000/N for every track tile visited, -100 for outside the track


# %% 1.1 Test random actions
eval_env_random_actions(env, render=False, logger = logger)


# %% Build and Train a model
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=0) # CNNPolicy


# %% Train the agent
model.learn(total_timesteps = num_steps)
# We want high explained_variance and 

# %% Save and reload
# # Save the agent
# model.save(model_file_name)
# del model  # delete trained model to demonstrate loading
# # model = DQN.load("dqn_lunar", env=env, logger.info_system_info=True)
# model = PPO.load(model_file_name, env=env)

# %% Evaluate
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model,  env , n_eval_episodes=10)
logger.info(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# %% Enjoy trained agent
obs = env.reset()
for _ in range(500):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
env.reset()
env.close()

# %%
