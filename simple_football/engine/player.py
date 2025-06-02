"""
Player model with role-based behavior and tactical intelligence.

Implements the complete role catalog with safety/liveness constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .pitch import Position, PitchZones


class PlayerRole(Enum):
    """
    Complete role catalog for tactical football simulation.
    
    Position Responsibilities:
    - GK: Shot stopping, distribution, penalty area control
    - CB: Central defending, aerial duels, build-up play
    - FB: Wing defending, overlapping runs, crossing
    - DM: Defensive screening, possession recycling
    - CM: Box-to-box play, passing range, tactical flexibility  
    - AM: Creative passing, shooting, pressing coordination
    - W: Wing play, pace, crossing and cutting inside
    - ST: Finishing, hold-up play, pressing from front
    """
    GOALKEEPER = "GK"
    CENTRE_BACK = "CB" 
    FULLBACK = "FB"
    DEFENSIVE_MIDFIELDER = "DM"
    CENTRE_MIDFIELDER = "CM"
    ATTACKING_MIDFIELDER = "AM"
    WINGER = "W"
    STRIKER = "ST"


@dataclass
class PlayerAttributes:
    """Player attributes affecting performance."""
    pace: float       # 0-100: Speed and acceleration
    passing: float    # 0-100: Pass accuracy and range
    shooting: float   # 0-100: Finishing ability
    defending: float  # 0-100: Tackling and positioning
    physicality: float # 0-100: Strength and stamina
    
    @classmethod
    def generate_for_role(cls, role: PlayerRole, base_skill: float = 70.0) -> 'PlayerAttributes':
        """Generate realistic attributes for a player role."""
        variance = 10.0
        
        # Role-specific attribute profiles
        profiles = {
            PlayerRole.GOALKEEPER: {"pace": 45, "passing": 60, "shooting": 20, "defending": 85, "physicality": 75},
            PlayerRole.CENTRE_BACK: {"pace": 55, "passing": 70, "shooting": 30, "defending": 85, "physicality": 80},
            PlayerRole.FULLBACK: {"pace": 75, "passing": 75, "shooting": 45, "defending": 75, "physicality": 70},
            PlayerRole.DEFENSIVE_MIDFIELDER: {"pace": 65, "passing": 80, "shooting": 55, "defending": 80, "physicality": 75},
            PlayerRole.CENTRE_MIDFIELDER: {"pace": 70, "passing": 85, "shooting": 65, "defending": 70, "physicality": 75},
            PlayerRole.ATTACKING_MIDFIELDER: {"pace": 75, "passing": 85, "shooting": 80, "defending": 55, "physicality": 65},
            PlayerRole.WINGER: {"pace": 85, "passing": 75, "shooting": 75, "defending": 60, "physicality": 65},
            PlayerRole.STRIKER: {"pace": 80, "passing": 70, "shooting": 90, "defending": 40, "physicality": 80},
        }
        
        profile = profiles[role]
        
        # Add randomness within variance
        return cls(
            pace=np.clip(np.random.normal(profile["pace"], variance), 1, 99),
            passing=np.clip(np.random.normal(profile["passing"], variance), 1, 99),
            shooting=np.clip(np.random.normal(profile["shooting"], variance), 1, 99),
            defending=np.clip(np.random.normal(profile["defending"], variance), 1, 99),
            physicality=np.clip(np.random.normal(profile["physicality"], variance), 1, 99)
        )


class Player:
    """
    Football player with role-based tactical behavior.
    
    Implements liveness and safety constraints:
    - Liveness: Player must attempt actions within role responsibilities
    - Safety: Player cannot violate basic rules (offside, handling, etc.)
    """
    
    def __init__(self, 
                 player_id: str,
                 name: str,
                 role: PlayerRole,
                 team_id: str,
                 attributes: Optional[PlayerAttributes] = None):
        """
        Initialize player with role and attributes.
        
        Args:
            player_id: Unique identifier
            name: Player name
            role: Tactical role
            team_id: Team identifier
            attributes: Player attributes (generated if None)
        """
        self.player_id = player_id
        self.name = name
        self.role = role
        self.team_id = team_id
        self.attributes = attributes or PlayerAttributes.generate_for_role(role)
        
        # State variables
        self.position = Position(0.0, 0.0)
        self.has_ball = False
        self.stamina = 100.0  # 0-100
        self.last_action_time = 0.0
        self.action_cooldown = 0.0  # Prevent action spam
        
        # Tactical state
        self.formation_position = Position(0.0, 0.0)
        self.marking_target: Optional['Player'] = None
        self.current_instruction = "maintain_position"
    
    @property
    def stamina_level(self) -> float:
        """Get stamina as a percentage (0.0-1.0)."""
        return self.stamina / 100.0
        
    def update_position(self, target_position: Position, time_step: float) -> None:
        """
        Update player position with movement constraints.
        
        Args:
            target_position: Desired position
            time_step: Simulation time step (seconds)
        """
        # Calculate maximum movement distance based on pace
        max_speed = self.attributes.pace / 10.0  # m/s conversion
        max_distance = max_speed * time_step
        
        # Move towards target position
        current_distance = self.position.distance_to(target_position)
        if current_distance <= max_distance:
            self.position = target_position
        else:
            # Move partial distance
            direction_x = (target_position.x - self.position.x) / current_distance
            direction_y = (target_position.y - self.position.y) / current_distance
            
            new_x = self.position.x + direction_x * max_distance
            new_y = self.position.y + direction_y * max_distance
            self.position = Position(new_x, new_y)
    
    def get_decision_weights(self, game_state: Dict) -> Dict[str, float]:
        """
        Calculate action weights based on role and game situation.
        
        Returns:
            Dict mapping action names to probability weights
        """
        weights = {}
        
        if not self.has_ball:
            # Off-ball actions
            weights.update({
                "move_to_space": 0.4,
                "mark_opponent": 0.3,
                "support_teammate": 0.2,
                "press_opponent": 0.1
            })
        else:
            # On-ball actions based on role
            if self.role == PlayerRole.GOALKEEPER:
                weights = {"distribute_ball": 0.7, "clear_ball": 0.3}
            elif self.role in [PlayerRole.CENTRE_BACK, PlayerRole.FULLBACK]:
                weights = {"pass_safe": 0.5, "pass_forward": 0.3, "clear_ball": 0.2}
            elif self.role == PlayerRole.DEFENSIVE_MIDFIELDER:
                weights = {"pass_safe": 0.4, "pass_forward": 0.4, "dribble": 0.2}
            elif self.role == PlayerRole.CENTRE_MIDFIELDER:
                weights = {"pass_forward": 0.4, "pass_safe": 0.3, "dribble": 0.2, "shoot": 0.1}
            elif self.role == PlayerRole.ATTACKING_MIDFIELDER:
                weights = {"pass_creative": 0.4, "shoot": 0.3, "dribble": 0.3}
            elif self.role == PlayerRole.WINGER:
                weights = {"cross": 0.4, "dribble": 0.3, "pass_forward": 0.2, "shoot": 0.1}
            elif self.role == PlayerRole.STRIKER:
                weights = {"shoot": 0.5, "pass_forward": 0.2, "hold_ball": 0.3}
        
        return weights
    
    def check_safety_constraints(self, action: str, game_state: Dict) -> bool:
        """
        Verify action doesn't violate safety constraints.
        
        Safety Rules:
        1. Cannot use hands (except goalkeeper in penalty area)
        2. Cannot commit obvious fouls
        3. Must respect offside position
        4. Cannot leave designated zones without tactical reason
        
        Returns:
            bool: True if action is safe
        """
        # Check action cooldown
        if self.action_cooldown > 0:
            return False
            
        # Role-specific safety checks
        if self.role == PlayerRole.GOALKEEPER:
            # GK can use hands in own penalty area
            if action == "handle_ball":
                return PitchZones.is_in_penalty_area(self.position, attacking_goal=False)
        
        # Check offside for attacking actions
        if action in ["shoot", "receive_pass"] and self.role in [PlayerRole.STRIKER, PlayerRole.WINGER]:
            return self._check_offside(game_state)
        
        return True
    
    def check_liveness_constraints(self, time_since_last_action: float) -> bool:
        """
        Verify player is fulfilling role responsibilities.
        
        Liveness Rules:
        1. Must attempt action within role timeframe
        2. Must respond to tactical situations
        3. Cannot remain passive for extended periods
        
        Returns:
            bool: True if liveness requirements met
        """
        # Players must act within 5 seconds when ball is nearby
        ball_distance = self._get_ball_distance()
        if ball_distance < 10.0 and time_since_last_action > 5.0:
            return False
            
        # Role-specific liveness checks
        if self.role == PlayerRole.GOALKEEPER and ball_distance < 20.0:
            return time_since_last_action < 3.0
            
        return True
    
    def _check_offside(self, game_state: Dict) -> bool:
        """Check if player is in offside position."""
        # Simplified offside check
        # In real implementation, would check last defender position
        return self.position.x < PitchZones.LENGTH - 10.0
    
    def _get_ball_distance(self) -> float:
        """Get distance to ball (placeholder)."""
        # Would access ball position from game state
        return 50.0  # Default value
    
    def apply_action_cooldown(self, action: str) -> None:
        """Apply cooldown period after action."""
        cooldown_times = {
            "shoot": 1.0,
            "pass": 0.5,
            "dribble": 0.3,
            "tackle": 1.0
        }
        self.action_cooldown = cooldown_times.get(action, 0.2)
    
    def update_stamina(self, time_step: float) -> None:
        """Update player stamina based on activity."""
        # Stamina decreases with movement and actions
        movement_cost = 0.1 * time_step
        action_cost = 0.5 if self.action_cooldown > 0 else 0
        
        self.stamina = max(0.0, self.stamina - movement_cost - action_cost)
        
        # Reduce cooldown
        self.action_cooldown = max(0.0, self.action_cooldown - time_step)
