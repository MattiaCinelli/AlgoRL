"""Simple script to run snips of code"""
# Standard Libraries
import os
os.chdir(os.path.dirname(__file__))

# Third party libraries
import pandas as pd
import numpy as np

# Local imports
from algorl.logs import logging
from algorl.src.grid_environment import *
from algorl.src.MC import MCPrediction, FirstVisitMCPredictions, MCExploringStarts

logger = logging.getLogger(__name__)

def run_MCPrediction(env):
    mc = MCPrediction(env)
    mc.compute_state_value()
    env.drew_statevalue_and_policy(plot_title = 'MC_Prediction')

def run_FirstVisitMCPredictions(env):
    mc = FirstVisitMCPredictions(env)
    mc.compute_state_value()
    env.drew_statevalue_and_policy(plot_title = 'MC_Prediction')

def run_MCExploringStarts(env):
    mc = MCExploringStarts(env)
    mc.compute_state_value()
    env.drew_statevalue_and_policy(plot_title = 'MC_Prediction')


if __name__ == "__main__":
    for gridword in GridWorldExamples.__subclasses__():
        run_MCPrediction(gridword.gridword())
        run_FirstVisitMCPredictions(gridword.gridword())
        run_MCExploringStarts(gridword.gridword())
        sys.exit()
    