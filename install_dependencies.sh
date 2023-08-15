#!/bin/bash -e

# python3-opencv and git are needed
sudo apt-get install -y git || exit 1
sudo apt-get install -y python3-opencv || exit 1

pip install -r requirements.txt || exit 1
pip install -r optional.txt || exit 1
