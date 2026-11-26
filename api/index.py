"""Vercel Python serverless WSGI entrypoint for the Rubik's Cube Robot Simulator."""

import os
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from app import app  # noqa: E402, F401
