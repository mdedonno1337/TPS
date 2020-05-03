#!/bin/bash

pip install coverage

cd /TPS
coverage run ./doctester.py
coverage report --include */TPS/*
