"""Simple script to run snips of code"""
# Standard Libraries

# Third party libraries

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import *
from algorl.src.TD import TabuladTD0, Sarsa, QLearning

def test_TabuladTD0(env):
    tdl = TabuladTD0(env)
    tdl.compute_state_value()
    env.drew_statevalue_and_policy(plot_title = 'TD0_Prediction')

def test_Sarsa(env):
    sarsa = Sarsa(env)
    sarsa.compute_state_value()
    env.drew_statevalue_and_policy(plot_title = 'sarsa_Prediction')

def test_QLearning(env):
    qlearn = QLearning(env)
    qlearn.compute_state_value()
    env.drew_statevalue_and_policy(plot_title = 'qlearn_Prediction')
    

if __name__ == "__main__":
    for gridword in GridWorldExamples.__subclasses__():
            test_TabuladTD0(gridword.gridword())
            # test_Sarsa(gridword.gridword())
            # test_QLearning(gridword.gridword())