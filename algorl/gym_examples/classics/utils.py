import numpy as np
from pathlib import Path
from logs import logging

# check if file exists
def file_exists(file_path):
    """
    Check if file exists
    :param file_path: (str) path to file
    :return: (bool) True if file exists, False otherwise
    """
    return Path(file_path).is_file()

def eval_env_random_actions(env, episodes = 10, render = False, logger = None):
    all_rewards = []
    for episode in range(1, episodes):
        state = env.reset() # Restart the agent at the beginning
        done = False # If the agent has completed the level
        score = 0 # Called score not return cause it's python

        while not done:
            if render:
                env.render()
            random_action = env.action_space.sample() # Do random actions
            state, reward, done, info = env.step(random_action) 
            score += reward
        logger.info('Episode: {}\tScore: {}'.format(episode, score))
        all_rewards.append(score)
    env.close()
    logger.info(f"Mean reward:{np.mean(all_rewards)} Num episodes:{episodes}")
    return np.mean(all_rewards)
    

def evaluate_model(model, num_episodes=100, logger=None):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    logger.info("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward