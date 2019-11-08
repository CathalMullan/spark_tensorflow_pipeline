#!/usr/bin/env bash
# Create the virtual environment (venv) for Python development.
# Install packages using Poetry,

# Ensure Python 3.7.* is installed and active
REQUIRED_PYTHON=3.7

PYTHON_VERSION="$(python --version 2>&1)"
case $PYTHON_VERSION in
    *"$REQUIRED_PYTHON"*)
        ;;
    *)
        echo "Incorrect Python version - project requires Python $REQUIRED_PYTHON, found $PYTHON_VERSION"
        echo "Recommended installation - https://github.com/pyenv/pyenv"
        exit 1
esac


# cd to project root
cd "$(dirname "${0}")" || exit
cd ../

# Create venv
python -m venv venv
source venv/bin/activate

# Update pip and install poetry
pip install --upgrade pip
pip install poetry

# Install packages
poetry install

# NOTE: Until Tensorflow releases fix - https://github.com/tensorflow/tensorflow/pull/32758
pip install tensorflow
pip install tensorflow_probability
pip install tensorflow-transform
