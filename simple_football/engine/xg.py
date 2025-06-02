"""
Expected Goals (xG) calculation for football simulation.

Implements a logistic regression model for shot quality assessment.
"""

import numpy as np
from typing import List
from .pitch import Position, PitchZones


class ExpectedGoalsModel:
    """
    Simple xG model based on shot distance, angle, and defensive pressure.
    
    The model uses a logistic function with empirically-derived coefficients
    to estimate goal probability for each shot attempt.
    """
    
    def __init__(self):
        """Initialize xG model with calibrated coefficients."""
        # Coefficients derived from shot data analysis
        self.intercept = 1.2
        self.distance_coeff = -0.08
        self.angle_coeff = 0.6
        self.pressure_coeff = -0.4
        self.penalty_bonus = 2.0
        
    def expected_goal(self, 
                     shot_position: Position,
                     defenders_positions: List[Position],
                     is_penalty: bool = False) -> float:
        """
        Calculate expected goal probability for a shot.
        
        Args:
            shot_position: Position where shot is taken
            defenders_positions: Positions of defending players
            is_penalty: Whether this is a penalty kick
            
        Returns:
            float: xG value between 0.0 and 1.0
        """
        if is_penalty:
            return 0.76  # Historical penalty conversion rate
            
        # Calculate shot distance to goal center
        goal_center = Position(PitchZones.LENGTH, PitchZones.WIDTH / 2)
        distance = shot_position.distance_to(goal_center)
        
        # Calculate shot angle (wider angle = better chance)
        angle = self._calculate_shot_angle(shot_position)
        
        # Calculate defensive pressure
        pressure = PitchZones.calculate_pressure(shot_position, defenders_positions)
        
        # Logistic regression formula
        logit = (self.intercept + 
                self.distance_coeff * distance +
                self.angle_coeff * angle +
                self.pressure_coeff * pressure)
        
        # Convert to probability using sigmoid
        xg = 1.0 / (1.0 + np.exp(-logit))
        
        # Ensure reasonable bounds
        return np.clip(xg, 0.01, 0.99)
    
    def _calculate_shot_angle(self, shot_position: Position) -> float:
        """
        Calculate effective shot angle in radians.
        
        Wider angles (closer to goal line) provide better scoring opportunities.
        """
        goal_left = Position(PitchZones.LENGTH, PitchZones.WIDTH/2 - PitchZones.GOAL_WIDTH/2)
        goal_right = Position(PitchZones.LENGTH, PitchZones.WIDTH/2 + PitchZones.GOAL_WIDTH/2)
        
        # Vectors from shot position to goal posts
        left_vector = np.array([goal_left.x - shot_position.x, goal_left.y - shot_position.y])
        right_vector = np.array([goal_right.x - shot_position.x, goal_right.y - shot_position.y])
        
        # Calculate angle between vectors
        dot_product = np.dot(left_vector, right_vector)
        norms = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
        
        if norms == 0:
            return 0.0
            
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return angle
    
    def is_good_shot_opportunity(self, xg_value: float) -> bool:
        """
        Determine if a shot represents a good scoring opportunity.
        
        Args:
            xg_value: Expected goal value
            
        Returns:
            bool: True if xG > 0.1 (reasonable chance)
        """
        return xg_value > 0.1
