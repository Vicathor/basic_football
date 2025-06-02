"""
Team model with formation, strategy, and collective behavior.

Implements tactical systems and coordination between players.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .player import Player, PlayerRole
from .pitch import Position, PitchZones


@dataclass
class TeamStrategy:
    """
    Team tactical strategy configuration.
    
    Defines high-level approach including:
    - Formation and player roles
    - Pressing intensity and triggers
    - Possession style preferences
    - Transition behaviors
    """
    formation: str = "4-4-2"
    pressing_intensity: float = 0.5  # 0-1 scale
    possession_style: str = "balanced"  # "defensive", "balanced", "attacking"
    transition_speed: str = "medium"  # "slow", "medium", "fast"
    width_preference: float = 0.7  # 0-1, how wide the team plays
    
    # Tactical triggers
    high_press_threshold: float = 0.3  # When to activate high press
    counter_attack_threshold: float = 0.8  # When to counter-attack


class Team:
    """
    Football team with tactical coordination and strategy implementation.
    
    Manages:
    - 11 players with assigned roles
    - Formation and positioning
    - Collective tactical behaviors
    - Strategy adaptation during match
    """
    
    def __init__(self, 
                 team_id: str,
                 name: str,
                 side: str,  # "home" or "away"
                 strategy: Optional[TeamStrategy] = None):
        """
        Initialize team with players and tactical setup.
        
        Args:
            team_id: Unique team identifier
            name: Team name
            side: Which side of pitch team defends
            strategy: Tactical strategy (default if None)
        """
        self.team_id = team_id
        self.name = name
        self.side = side
        self.strategy = strategy or TeamStrategy()
        
        # Team state
        self.possession = False
        self.formation_positions = []
        self.current_phase = "build_up"  # "build_up", "attack", "defend", "transition"
        
        # Performance metrics
        self.possession_time = 0.0
        self.passes_attempted = 0
        self.passes_completed = 0
        self.shots_taken = 0
        self.goals_scored = 0
        
        # Initialize players
        self.players = self._create_players()
        self._set_formation_positions()
    
    def _create_players(self) -> List[Player]:
        """Create 11 players with appropriate roles for formation."""
        players = []
        
        # Standard 4-4-2 formation roles
        roles = [
            PlayerRole.GOALKEEPER,
            PlayerRole.FULLBACK,      # Left back
            PlayerRole.CENTRE_BACK,   # Centre back 1
            PlayerRole.CENTRE_BACK,   # Centre back 2  
            PlayerRole.FULLBACK,      # Right back
            PlayerRole.CENTRE_MIDFIELDER,  # Left mid
            PlayerRole.CENTRE_MIDFIELDER,  # Centre mid 1
            PlayerRole.CENTRE_MIDFIELDER,  # Centre mid 2
            PlayerRole.CENTRE_MIDFIELDER,  # Right mid
            PlayerRole.STRIKER,       # Striker 1
            PlayerRole.STRIKER,       # Striker 2
        ]
        
        for i, role in enumerate(roles):
            player = Player(
                player_id=f"{self.team_id}_P{i+1:02d}",
                name=f"Player {i+1}",
                role=role,
                team_id=self.team_id
            )
            players.append(player)
            
        return players
    
    def _set_formation_positions(self) -> None:
        """Set formation positions for all players."""
        positions = PitchZones.get_formation_positions(
            self.strategy.formation, 
            self.side
        )
        
        if len(positions) == len(self.players):
            for player, position in zip(self.players, positions):
                player.formation_position = position
                player.position = Position(position.x, position.y)
    
    def update_tactical_phase(self, ball_position: Position, possession_team: str) -> None:
        """
        Update team's tactical phase based on game situation.
        
        Tactical Phases:
        1. Build-up: Controlled possession in own half
        2. Attack: Creating scoring opportunities  
        3. Defend: Preventing opponent scoring
        4. Transition: Switching between attack/defense
        """
        has_possession = (possession_team == self.team_id)
        ball_zone = PitchZones.get_zone(ball_position)
        
        if has_possession:
            if ball_zone == "def":
                self.current_phase = "build_up"
            elif ball_zone == "att":
                self.current_phase = "attack"
            else:
                self.current_phase = "transition"
        else:
            self.current_phase = "defend"
    
    def get_team_instructions(self, game_state: Dict) -> Dict[str, str]:
        """
        Generate tactical instructions for players based on current phase.
        
        Returns:
            Dict mapping player_id to instruction string
        """
        instructions = {}
        
        for player in self.players:
            if self.current_phase == "build_up":
                instructions[player.player_id] = self._get_buildup_instruction(player)
            elif self.current_phase == "attack":
                instructions[player.player_id] = self._get_attack_instruction(player)
            elif self.current_phase == "defend":
                instructions[player.player_id] = self._get_defend_instruction(player)
            else:  # transition
                instructions[player.player_id] = self._get_transition_instruction(player)
        
        return instructions
    
    def _get_buildup_instruction(self, player: Player) -> str:
        """Get build-up phase instruction for player."""
        if player.role == PlayerRole.GOALKEEPER:
            return "distribute_short"
        elif player.role in [PlayerRole.CENTRE_BACK, PlayerRole.FULLBACK]:
            return "support_buildup"
        elif player.role == PlayerRole.DEFENSIVE_MIDFIELDER:
            return "drop_deep"
        else:
            return "create_space"
    
    def _get_attack_instruction(self, player: Player) -> str:
        """Get attacking phase instruction for player."""
        if player.role == PlayerRole.STRIKER:
            return "make_runs"
        elif player.role == PlayerRole.ATTACKING_MIDFIELDER:
            return "create_chances"
        elif player.role == PlayerRole.WINGER:
            return "stretch_play"
        elif player.role in [PlayerRole.FULLBACK]:
            return "overlap"
        else:
            return "support_attack"
    
    def _get_defend_instruction(self, player: Player) -> str:
        """Get defensive phase instruction for player."""
        if player.role == PlayerRole.GOALKEEPER:
            return "organize_defense"
        elif player.role in [PlayerRole.CENTRE_BACK, PlayerRole.FULLBACK]:
            return "maintain_line"
        elif player.role == PlayerRole.DEFENSIVE_MIDFIELDER:
            return "screen_defense"
        else:
            return "press_opponent"
    
    def _get_transition_instruction(self, player: Player) -> str:
        """Get transition phase instruction for player."""
        if self.strategy.transition_speed == "fast":
            return "quick_transition"
        else:
            return "controlled_transition"
    
    def calculate_formation_compactness(self) -> float:
        """
        Calculate how compact the team formation is.
        
        Returns:
            float: Compactness measure (0-1, higher = more compact)
        """
        if len(self.players) < 2:
            return 1.0
            
        positions = [p.position for p in self.players[1:]]  # Exclude goalkeeper
        
        # Calculate average distance between players
        total_distance = 0.0
        pair_count = 0
        
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                total_distance += pos1.distance_to(pos2)
                pair_count += 1
        
        if pair_count == 0:
            return 1.0
            
        avg_distance = total_distance / pair_count
        
        # Normalize to 0-1 scale (assuming max reasonable distance is 50m)
        compactness = max(0.0, 1.0 - (avg_distance / 50.0))
        return compactness
    
    def get_possession_percentage(self, total_time: float) -> float:
        """Calculate possession percentage for the team."""
        if total_time <= 0:
            return 0.0
        return (self.possession_time / total_time) * 100.0
    
    def get_pass_accuracy(self) -> float:
        """Calculate pass completion percentage."""
        if self.passes_attempted == 0:
            return 0.0
        return (self.passes_completed / self.passes_attempted) * 100.0
    
    def update_stats(self, event_type: str, success: bool = True) -> None:
        """Update team statistics based on events."""
        if event_type == "pass":
            self.passes_attempted += 1
            if success:
                self.passes_completed += 1
        elif event_type == "shot":
            self.shots_taken += 1
        elif event_type == "goal":
            self.goals_scored += 1
    
    def adapt_strategy(self, match_time: float, score_difference: int) -> None:
        """
        Adapt strategy based on match situation.
        
        Args:
            match_time: Current match time in minutes
            score_difference: Goal difference (positive if winning)
        """
        # Late in match adaptations
        if match_time > 70:
            if score_difference < 0:  # Losing - more attacking
                self.strategy.pressing_intensity = min(1.0, self.strategy.pressing_intensity + 0.2)
                self.strategy.possession_style = "attacking"
            elif score_difference > 0:  # Winning - more defensive
                self.strategy.pressing_intensity = max(0.2, self.strategy.pressing_intensity - 0.1)
                self.strategy.possession_style = "defensive"
