#!/usr/bin/env bash
set -e
pip install pybind11 cmake wheel setuptools
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
