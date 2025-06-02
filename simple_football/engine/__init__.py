"""
Football simulation engine components.

This module contains the core simulation engine including:
- Pitch geometry and zone management
- Player behavior and tactical roles
- Team coordination and formations
- Expected goals calculation
- Main match simulation engine
"""

from .pitch import Position, PitchZones
from .player import Player, PlayerRole
from .team import Team
from .xg import ExpectedGoalsModel
from .match import MatchEngine

__all__ = ['Position', 'PitchZones', 'Player', 'PlayerRole', 'Team', 'ExpectedGoalsModel', 'MatchEngine']
