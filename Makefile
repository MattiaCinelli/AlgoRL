# Get the name of the current directory
CURRENT_DIR := $(shell basename "$(PWD)")

# Define the name of the virtual environment
VENV_NAME := .venv_$(CURRENT_DIR)

# Define the path to the Python executable
PYTHON := python3

# Define the path to the virtual environment's bin directory
VENV_BIN := $(CURDIR)/$(VENV_NAME)/bin

# Default target: create the virtual environment
.PHONY: all
all: $(VENV_BIN)/activate

# Target to create the virtual environment
$(VENV_BIN)/activate: requirements.txt
	echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_NAME)
	touch $(VENV_BIN)/activate
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install -e .

# Target to clean up the virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV_NAME)

# Target to install packages
.PHONY: install
install: $(VENV_BIN)/activate requirements.txt
	@echo "Installing packages..."
	@$(VENV_BIN)/pip install -r requirements.txt

# Test
# .PHONY: test
# test:
# 	pytest