#!/bin/bash

# Check if Pipenv is installed
if ! command -v pipenv &> /dev/null; then
    echo "Pipenv not found, installing..."
    pip install pipenv
fi

echo "Setting up Pipenv environment..."
pipenv install

echo "Setup complete!"
echo "To activate the environment, run: pipenv shell"
echo "Or run commands with: pipenv run <command>" 