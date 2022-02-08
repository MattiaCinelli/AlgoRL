from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Needed for dependencies
INSTALL_REQUIRES = [
      'wheel',
]

setup(
    name = 'algorl',
    packages = find_packages(),
    version = '0.1.0',
    description = 'A repo of the most common RL algorithms and example of their applications',
    long_description_content_type = 'text/markdown',
    long_description = long_description,
    author='MattiaCinelli',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        ],
    install_requires = INSTALL_REQUIRES,
    python_requires = '>=3.7',
    test_suite='tests',
    zip_safe = False
)
