import os
import pytest
import pandas as pd
from contextlib import contextmanager


from algorl.src.bandit import Bandits #, RLFunctions

os.chdir(os.path.dirname(__file__))

@contextmanager
def does_not_raise():
    '''A context manager that does not raise an exception.'''
    yield

def test_bandits(setup_teardown):
    bandits = Bandits(number_of_arms = 5)
    assert bandits.number_of_arms == 5
    assert type(bandits.return_bandit_df()) == type(pd.DataFrame())
