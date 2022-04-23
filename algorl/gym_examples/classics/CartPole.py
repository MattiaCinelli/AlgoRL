# Standard Libraries
import os
from pathlib import Path

# Third party libraries
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy #  MlpPolicy because the observation of the CartPole task is a feature vector, not images.
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor 

# Local imports
from utils import file_exists, eval_env_random_actions
from logs import logging

logger = logging.getLogger("CartPole")
# Initialization
env_name = 'CartPole-v1'

# Create folder to save models
directory_path = 'models'
Path(directory_path).mkdir(parents=True, exist_ok=True)

num_steps = 10#0_000
model_file_name = Path(directory_path, f'{env_name}_{num_steps}')

env = gym.make(env_name)

eval_env_random_actions(env, logger=logger)

logger.debug(env.action_space)
logger.debug(env.observation_space)

## Create model
model = PPO(MlpPolicy, env, verbose=0)

# Training
# If a model trained already exist avoid re-doing it
if not file_exists(model_file_name):
    # Train the agent for n steps
    model = model.learn(total_timesteps=num_steps)
    # Save the model 
    model.save(model_file_name)
    # sample an observation from the environment
    obs = model.env.observation_space.sample()

    # Check prediction before saving
    print(f"pre saved {model.predict(obs, deterministic=True)}")

    del model # delete trained model to demonstrate loading
    loaded_model = PPO.load(model_file_name)
    # Check that the prediction is the same after loading (for the same observation)
    # logger.info(("loaded", loaded_model.predict(obs, deterministic=True))

    # show the save hyperparameters
    logger.info(f"loaded, gamma = {loaded_model.gamma}, num_steps = {loaded_model.n_steps}")
    # as the environment is not serializable, we need to set a new instance of the environment
    loaded_model.set_env(DummyVecEnv([lambda: gym.make(env_name)]))
    # and continue training
    model = loaded_model.learn(num_steps)

# Evaluation
mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=100)
logger.info(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# Set up fake display; otherwise rendering will fail
import os
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'



import base64
from pathlib import Path
from IPython import display as ipythondisplay
def show_videos(video_path='', prefix=''):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay loop controls style="height: 400px;">
        <source src="data:video/mp4;base64,{}" type="video/mp4" />
        </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))



from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env, video_folder=video_folder,
        record_video_trigger=lambda step: step == 0, video_length=video_length,
        name_prefix=prefix)
        
    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
    eval_env.close()