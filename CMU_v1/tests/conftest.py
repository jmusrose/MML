"""Pytest configuration for DML_v1/CMU_v1 tests.

Adds the CMU_v1 package root to ``sys.path`` so test files can import the
``utils`` / ``models`` / ``data`` packages directly (e.g. ``from utils.utils
import Averager``) regardless of where pytest is invoked from.
"""
import os
import sys

CMU_V1_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if CMU_V1_ROOT not in sys.path:
    sys.path.insert(0, CMU_V1_ROOT)
