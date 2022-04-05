"""Simple script to run snips of code"""
# Standard Libraries

# Third party libraries

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import MakeGrid
from algorl.src.MC import MCPrediction, FirstVisitMCPredictions, MCExploringStarts

def print_tab_A():
    env = MakeGrid(
        walls = [(1, 1)], 
        terminal_states = {(0, 3): 1, (1, 3): -10}
        )
    env.render_state_value()
    mc = MCPrediction(env)
    # mc = FirstVisitMCPredictions(env)
    # mc = MCExploringStarts(env)
    mc.compute_state_value()

if __name__ == "__main__":
    print_tab_A()