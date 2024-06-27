# Variables
VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip
REQUIREMENTS := requirements.txt
LOGS := logs

.PHONY: all setup start stop clean

# Default target
all: setup start

# Setup virtual environment and install requirements
setup: $(VENV_NAME)
	$(PIP) install -r $(REQUIREMENTS)

# Create virtual environment
$(VENV_NAME):
	python3 -m venv $(VENV_NAME)

# Start the application
start:
	$(PYTHON) cli.py

# Stop the application
stop:
	pass

# Clean up
clean:
	rm -rf $(VENV_NAME)
	rm -rf $(LOGS)


