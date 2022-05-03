from contextlib import contextmanager
import os
import pytest

from algorl.src.tool_box import create_directory #, RLFunctions

# os.chdir(os.path.dirname(__file__))

@contextmanager
def does_not_raise():
    '''A context manager that does not raise an exception.'''
    yield

def test_create_directory():
    '''
    Tests for create_directory
    '''
    file_path = os.path.join('temp', 'test_dir')
    create_directory(file_path) 
    assert os.path.isdir(file_path)



# def test_RLFunctions():
#     '''
#     Tests for RLFunctions
#     '''
#     rl_sim = RLFunctions()
#     assert rl_sim.get_random_action() == 0
#     assert rl_sim.epsilon_greedy(0) == 0
#     assert rl_sim.greedy(0) == 0

# @pytest.mark.parametrize('a, b', (
#     # Correct inputs and outputs
#     ('', does_not_raise()),
#
#     # Incorrect inputs
#     ('', pytest.raises(ValueError)),
# ))
# def test_example(a, b):
#     '''Tests for function_x'''
#     with expectation:
# #         assert isinstance(function_x(time), datetime.date) == True
