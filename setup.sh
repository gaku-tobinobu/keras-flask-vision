#!/usr/bin/env bash
# make sure you are using recent pip/virtualenv versions
python -m pip install -U pip virtualenv

virtualenv .env
source .env/bin/activate
export PYTHONPATH=`pwd`
pip install -r requirements.txt
