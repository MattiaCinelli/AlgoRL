"""Simple script to run snips of code"""
# Standard Libraries

# Third party libraries

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import Make
from algorl.src.MC import MCPrediction, MCExploringStarts

def print_tab_A():
    env = Make(walls = [(1, 1)])
    env.render_state_value()
    mc = MCExploringStarts(env)
    mc.compute_state_value()

if __name__ == "__main__":
    print_tab_A()