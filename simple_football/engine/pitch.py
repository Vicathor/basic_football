"""
Pitch geometry and spatial utilities for football simulation.

Provides coordinate system, zones, and distance calculations for tactical analysis.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Position:
    """2D position on the football pitch."""
    x: float  # 0-105 meters (goal line to goal line)
    y: float  # 0-68 meters (touchline to touchline)
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_to_goal(self, goal_x: float = 105.0) -> float:
        """Calculate angle to goal center in radians."""
        goal_center = Position(goal_x, 34.0)  # Center of goal
        dx = goal_center.x - self.x
        dy = goal_center.y - self.y
        return np.arctan2(dy, dx)


class PitchZones:
    """
    Tactical zones for football pitch analysis.
    
    Zones are used for:
    - Formation positioning
    - Pressure calculations  
    - Tactical pattern recognition
    """
    
    # Pitch dimensions (FIFA standard)
    LENGTH = 105.0  # meters
    WIDTH = 68.0    # meters
    
    # Zone boundaries (x-coordinates)
    DEFENSIVE_THIRD = 35.0
    MIDDLE_THIRD = 70.0
    ATTACKING_THIRD = 105.0
    
    # Penalty areas
    PENALTY_AREA_LENGTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    
    # Goal dimensions
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44
    
    @classmethod
    def get_zone(cls, position: Position) -> str:
        """
        Determine tactical zone for a position.
        
        Returns:
            str: Zone identifier ('def', 'mid', 'att')
        """
        if position.x <= cls.DEFENSIVE_THIRD:
            return 'def'
        elif position.x <= cls.MIDDLE_THIRD:
            return 'mid'
        else:
            return 'att'
    
    @classmethod
    def is_in_penalty_area(cls, position: Position, attacking_goal: bool = True) -> bool:
        """Check if position is in penalty area."""
        if attacking_goal:
            # Attacking penalty area (opponent's)
            return (position.x >= cls.LENGTH - cls.PENALTY_AREA_LENGTH and
                    cls.WIDTH/2 - cls.PENALTY_AREA_WIDTH/2 <= position.y <= cls.WIDTH/2 + cls.PENALTY_AREA_WIDTH/2)
        else:
            # Defensive penalty area (own)
            return (position.x <= cls.PENALTY_AREA_LENGTH and
                    cls.WIDTH/2 - cls.PENALTY_AREA_WIDTH/2 <= position.y <= cls.WIDTH/2 + cls.PENALTY_AREA_WIDTH/2)
    
    @classmethod
    def get_formation_positions(cls, formation: str, team_side: str) -> List[Position]:
        """
        Get standard formation positions for a team.
        
        Args:
            formation: Formation string (e.g., "4-4-2")
            team_side: "home" or "away"
            
        Returns:
            List of 11 positions for outfield players + goalkeeper
        """
        formations = {
            "4-4-2": {
                "home": [
                    Position(5.0, 34.0),    # GK
                    Position(20.0, 10.0),   # LB
                    Position(20.0, 25.0),   # CB
                    Position(20.0, 43.0),   # CB
                    Position(20.0, 58.0),   # RB
                    Position(40.0, 15.0),   # LM
                    Position(40.0, 30.0),   # CM
                    Position(40.0, 38.0),   # CM
                    Position(40.0, 53.0),   # RM
                    Position(75.0, 25.0),   # ST (moved to attacking zone)
                    Position(75.0, 43.0),   # ST (moved to attacking zone)
                ],
                "away": [
                    Position(100.0, 34.0),  # GK
                    Position(85.0, 58.0),   # LB
                    Position(85.0, 43.0),   # CB
                    Position(85.0, 25.0),   # CB
                    Position(85.0, 10.0),   # RB
                    Position(65.0, 53.0),   # LM
                    Position(65.0, 38.0),   # CM
                    Position(65.0, 30.0),   # CM
                    Position(65.0, 15.0),   # RM
                    Position(30.0, 43.0),   # ST (moved to attacking zone)
                    Position(30.0, 25.0),   # ST (moved to attacking zone)
                ]
            }
        }
        
        return formations.get(formation, {}).get(team_side, [])
    
    @classmethod
    def calculate_pressure(cls, player_pos: Position, opponents: List[Position]) -> float:
        """
        Calculate pressure level based on nearest opponents.
        
        Pressure Thresholds:
        - High: < 3m to nearest opponent
        - Medium: 3-8m to nearest opponent  
        - Low: > 8m to nearest opponent
        
        Returns:
            float: Pressure value (0.0 = low, 1.0 = high)
        """
        if not opponents:
            return 0.0
            
        min_distance = min(player_pos.distance_to(opp) for opp in opponents)
        
        if min_distance < 3.0:
            return 1.0  # High pressure
        elif min_distance < 8.0:
            return 0.5  # Medium pressure
        else:
            return 0.0  # Low pressure
