#!/bin/bash

pip install coverage

cd /TPS
coverage run ./TPSModules_unittest.py
coverage report --include */TPS/*
