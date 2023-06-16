#!/bin/bash -e

# python3-opencv and git are needed
apt-get install -y git
apt-get install -y python3-opencv

pip install -r requirements.txt || exit 1
pip install -r optional.txt || exit 1