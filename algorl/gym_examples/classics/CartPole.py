## Standard Libraries 

## Third party libraries
# import sys
import gym
# import pandas as pd
# import numpy as np
from icecream import ic
# import matplotlib.pyplot as plt
from stable_baselines3 import A2C

## Local imports


"""
# CartPole-v1
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
The system is controlled by applying a force of +1 or -1 to the cart. (action_space)
The pendulum starts upright, and the goal is to prevent it from falling over. 
A reward of +1 is provided for every timestep that the pole remains upright. 
The episode ends when the pole is more than 15 degrees from vertical, 
or the cart moves more than 2.4 units from the center.
"""


class CartPoleClass():
    def __init__(self, enviroment: 'CartPole-v1') -> None:
        self.env = gym.make(enviroment)
        self.init_model()

    # Create the model
    def init_model(self) -> None:
        self.model = A2C('MlpPolicy', self.env, verbose=1)

    # Train the model
    def train_model(self) -> None:
        self.model.learn(total_timesteps=10000)

    # Save the model
    def save_model(self) -> None:
        self.model.save("cartpole_model")

    def predict_model(self) -> None:
        obs = self.env.reset()
        for i in range(1000):
            action, _state = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
                ic('Done at step {}'.format(i))
        self.env.close()


if __name__ == "__main__":
    ic("CartPole-v1")

    env_name = 'CartPole-v1'
    model_name = 'MlpPolicy'

    env = gym.make(env_name)
    model = A2C(model_name, env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            ic('Done at step {}'.format(i))
