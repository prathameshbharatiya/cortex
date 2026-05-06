# setup.py — backward-compatibility shim only.
#
# All real packaging configuration lives in pyproject.toml.
# This file exists solely so `pip install -e .` works on older pip versions
# (< 21.3) that do not support PEP 660 editable installs without setup.py.
#
# Do NOT add configuration here. Edit pyproject.toml instead.

from setuptools import setup

setup()
