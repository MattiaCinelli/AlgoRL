##############################
ENVIRONMENT_PREFIX=$(shell pwd)
ENVIRONMENT_NAME=.valgorl
ALTERNATIVE_ENV_NAME=${ENVIRONMENT_NAME}_nb #Name of the env for jupyter notebook. 
# You might want to give the same name env but different for the notebook

# Virtualenv for project
dev: requirements.txt
	echo "Creating virtual environment..."
	python3 -m piptools sync requirements.txt requirements.txt

requirements.txt: venv
	python3 -m piptools compile requirements.in --output-file requirements.txt

venv: requirements.in	
	echo "Compiling requirements..."
	python3 -m venv ${ENVIRONMENT_NAME}
	pip install --upgrade 'pip'
	. ${ENVIRONMENT_NAME}/bin/activate
	pip install 'pip-tools' 'numpy' 'scipy' 'setuptools>=41.0.0'
	# python3 -m ipykernel install --user --name=${ENVIRONMENT_NAME} --display-name "${ALTERNATIVE_ENV_NAME}"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# References
# [1] https://pypi.org/project/pip-tools/