"""
Simple Football Simulation Package

A process mining-focused football match simulator.
"""

__version__ = "1.0.0"
__author__ = "Simple Football Sim Team"

from .engine.match import MatchEngine
from .scripts.run_sim import simulate_matches

__all__ = ["MatchEngine", "simulate_matches"]
