"""Simple script to run snips of code"""
# Standard Libraries

# Third party libraries

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import Make
from algorl.src.tabular import DP

def print_tab_A():
    env = Make(walls = [(1, 1)])