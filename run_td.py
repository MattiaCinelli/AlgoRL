"""Simple script to run snips of code"""
# Standard Libraries

# Third party libraries

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import *
from algorl.src.TD import *

def test_TabularTD0(env, grid_name):
    tdl = TabularTD0(env)
    tdl.compute_state_value()
    env.drew_statevalue_and_policy(plot_title = f'TD0_{grid_name}')
    
def test_Sarsa(env, grid_name):
    Sarsa(env).compute_state_value(plot_name=f'sarsa_{grid_name}')

def test_QLearning(env, grid_name):
    QLearning(env).compute_state_value(plot_name=f'qlearn{grid_name}')
    
def test_NstepTD(env, grid_name, n_step):
    NstepTD(env, n_step=n_step).compute_state_value()
    env.drew_statevalue_and_policy(plot_title = f'{n_step}-stepTD_{grid_name}')

if __name__ == "__main__":
    for gridword in GridWorldExamples.__subclasses__():
        # test_TabularTD0(gridword.gridword(), grid_name = gridword.__name__)
        # test_Sarsa(gridword.gridword(), grid_name = gridword.__name__)
        # test_QLearning(gridword.gridword(), grid_name = gridword.__name__)
        test_NstepTD(gridword.gridword(), grid_name = gridword.__name__, n_step=3)
        sys.exit()