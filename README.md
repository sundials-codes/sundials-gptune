# Sundials + GPTune

This repository started with Alex Fish's summer 2022 internship project, using GPTune to build a table of Sundials parameters.

## To install with setup scripts

To start running scripts,
1. Ensure GPTune is built
2. Ensure Sundials is built
3. Edit the first line of `src/setup-env.sh` is accurate to the root directory of your GPTune installation
4. Edit the second line of `src/setup-env.sh` is accurate to the root directory of your Sundials installation
4. Run `. src/setup-env.sh` to get the environment variables and newest python version loaded.

## Installing with Pip and Python virtual environments

python -m pyvenv /path/to/venv
pip install -r requirements.txt
