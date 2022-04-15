import os
import glob
import shutil
import pytest
os.chdir(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def setup_teardown():
    print("setup")
    os.makedirs('temp', exist_ok=True)
    
    yield

    print("teardown")
    shutil.rmtree('temp')
    shutil.rmtree('images')
    shutil.rmtree('__pycache__')
