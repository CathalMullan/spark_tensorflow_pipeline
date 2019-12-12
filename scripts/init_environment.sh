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
poetry update
poetry install

# NOTE: Until Tensorflow releases fix - https://github.com/tensorflow/tensorflow/pull/32758
pip install tf-nightly-2.0-preview==2.0.0-dev20190731 tfp-nightly --upgrade
TFVERSION=$(python -c 'import tensorflow; print(tensorflow.__version__)')
[[ $TFVERSION == '2.0.0-dev20190731' ]] &&
  echo >&2 "Failed to install the most recent TF. Found: ${TFVERSION}."
