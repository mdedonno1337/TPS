#!/bin/bash

pip install coverage

coverage run ./doctester.py
coverage report --include */TPS/*

