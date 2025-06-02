"""
Main match engine for football simulation.

Orchestrates time-stepped simulation with event generation and tactical coordination.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .pitch import Position, PitchZones
from .player import Player, PlayerRole
from .team import Team, TeamStrategy  
from .xg import ExpectedGoalsModel
from logger.event_logger import FootballEventLogger


@dataclass
class MatchState:
    """Current state of the football match."""
    time_elapsed: float = 0.0  # seconds
    ball_position: Position = field(default_factory=lambda: Position(52.5, 34.0))  # Center circle
    ball_team: Optional[str] = None
    ball_player: Optional[str] = None
    
    # Match events
    home_score: int = 0
    away_score: int = 0
    half: int = 1
    
    # Flow state
    last_touch_time: float = 0.0
    possession_start_time: float = 0.0
    current_possession_team: Optional[str] = None


class MatchEngine:
    """
    Time-stepped football match simulation engine.
    
    Implements:
    - 90-minute match simulation in 100ms steps
    - Tactical coordination between teams
    - Event generation with process mining logs
    - xG calculation for shots
    - Role-based player behavior
    """
    
    TIME_STEP = 0.1  # 100ms time steps
    MATCH_DURATION = 90 * 60  # 90 minutes in seconds
    
    def __init__(self, 
                 home_team: Team,
                 away_team: Team,
                 random_seed: Optional[int] = None):
        """
        Initialize match engine with two teams.
        
        Args:
            home_team: Home team
            away_team: Away team
            random_seed: Random seed for reproducibility
        """
        self.home_team = home_team
        self.away_team = away_team
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Initialize systems
        self.match_state = MatchState()
        self.xg_model = ExpectedGoalsModel()
        self.event_logger = FootballEventLogger(f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Performance tracking
        self.possession_stats = {home_team.team_id: 0.0, away_team.team_id: 0.0}
        self.shot_stats = {home_team.team_id: [], away_team.team_id: []}
        
    def simulate_match(self) -> Dict:
        """
        Simulate complete 90-minute football match.
        
        Returns:
            Dict containing match result and statistics
        """
        print(f"Starting match: {self.home_team.name} vs {self.away_team.name}")
        
        # Kickoff
        self._kickoff()
        
        # Main simulation loop
        step_count = 0
        max_steps = int(self.MATCH_DURATION / self.TIME_STEP)
        
        while step_count < max_steps:
            self._simulate_step()
            step_count += 1
            
            # Progress reporting
            if step_count % 9000 == 0:  # Every 15 minutes
                minutes = step_count * self.TIME_STEP / 60
                print(f"Match time: {minutes:.0f}min - Score: {self.match_state.home_score}-{self.match_state.away_score}")
        
        # Final whistle
        self._end_match()
        
        return self._generate_match_summary()
    
    def _simulate_step(self) -> None:
        """Simulate single time step of the match."""
        # Update match time
        self.match_state.time_elapsed += self.TIME_STEP
        self.event_logger.advance_time(self.TIME_STEP)
        
        # Update team tactical phases
        self.home_team.update_tactical_phase(
            self.match_state.ball_position, 
            self.match_state.current_possession_team or ""
        )
        self.away_team.update_tactical_phase(
            self.match_state.ball_position,
            self.match_state.current_possession_team or ""
        )
        
        # Update player positions and states
        self._update_players()
        
        # Generate match events
        self._generate_events()
        
        # Update possession statistics
        self._update_possession_stats()
        
        # Check for half-time
        if self.match_state.time_elapsed >= 45 * 60 and self.match_state.half == 1:
            self._half_time()
    
    def _update_players(self) -> None:
        """Update all player positions and states."""
        all_players = self.home_team.players + self.away_team.players
        
        for player in all_players:
            # Update player state
            player.update_stamina(self.TIME_STEP)
            
            # Get target position based on formation and ball position
            target_position = self._get_player_target_position(player)
            
            # Move player towards target
            player.update_position(target_position, self.TIME_STEP)
    
    def _get_player_target_position(self, player: Player) -> Position:
        """Calculate target position for player based on tactics and ball position."""
        # Base formation position
        base_pos = player.formation_position
        
        # Special logic for goalkeepers - keep them in defensive third
        if player.role == PlayerRole.GOALKEEPER:
            # Goalkeepers should stay in their penalty area with minimal movement
            gk_x = 8.0 if player.team_id == self.home_team.team_id else 97.0
            # Minor adjustment based on ball position
            ball_y_adjustment = (self.match_state.ball_position.y - PitchZones.WIDTH/2) * 0.1
            gk_y = np.clip(PitchZones.WIDTH/2 + ball_y_adjustment, 20.0, 48.0)
            return Position(gk_x, gk_y)
        
        # Special logic for strikers - be more aggressive and stay in attacking zones
        if player.role == PlayerRole.STRIKER:
            ball_pos = self.match_state.ball_position
            
            # Strikers should generally stay in attacking zones
            if player.team_id == self.home_team.team_id:
                # HOME strikers should stay in x >= 70 zone (their attacking zone) 
                # More aggressive positioning, especially when team has possession
                if player.team_id == self.match_state.current_possession_team:
                    target_x = min(95.0, max(75.0, base_pos.x + 5.0))
                else:
                    target_x = max(70.0, base_pos.x)  # Ensure in attacking zone (x >= 70)
            else:
                # AWAY strikers should stay in x <= 35 zone (their attacking zone)
                if player.team_id == self.match_state.current_possession_team:
                    target_x = max(10.0, min(30.0, base_pos.x - 5.0))
                else:
                    target_x = min(35.0, base_pos.x)  # Ensure in attacking zone (x <= 35)
                    
            return Position(target_x, base_pos.y)
        
        # Standard positioning for other players
        ball_pos = self.match_state.ball_position
        
        # Adjust based on ball position and tactical phase
        adjustment_factor = 0.1
        
        if player.team_id == self.match_state.current_possession_team:
            # In possession - slight movement towards ball
            dx = (ball_pos.x - base_pos.x) * adjustment_factor
            dy = (ball_pos.y - base_pos.y) * adjustment_factor
        else:
            # Out of possession - maintain defensive shape
            dx = dy = 0.0
        
        target_x = base_pos.x + dx
        target_y = base_pos.y + dy
        
        # Keep within pitch bounds
        target_x = np.clip(target_x, 0, PitchZones.LENGTH)
        target_y = np.clip(target_y, 0, PitchZones.WIDTH)
        
        return Position(target_x, target_y)
    
    def _generate_events(self) -> None:
        """Generate match events based on current state."""
        # Check for referee events first (fouls, cards, offside, etc.)
        if self._should_generate_referee_event():
            self._generate_referee_event()
            return
        
        # Event probability based on ball location and possession
        event_probability = self._calculate_event_probability()
        
        # Occasionally force goalkeeper events to ensure they participate
        force_gk_event = False
        if np.random.random() < 0.005:  # 0.5% chance per step to force GK event (increased from 0.1%)
            ball_zone = PitchZones.get_zone(self.match_state.ball_position)
            if ball_zone == "def":  # Only in defensive zones
                force_gk_event = True
        
        if np.random.random() < event_probability or force_gk_event:
            if force_gk_event:
                event_type = "save"  # Force a save event for goalkeepers
            else:
                event_type = self._determine_event_type()
            self._execute_event(event_type)
    
    def _should_generate_referee_event(self) -> bool:
        """Determine if a referee event should be generated."""
        # Base probability for referee events - need to generate ~20-25 events across 90 minutes
        # With 54000 time steps (90min * 60s * 10 steps/s), need ~0.0004 probability per step
        base_prob = 0.0005  # 0.05% chance per step
        
        # Increase probability in high-intensity areas
        ball_zone = PitchZones.get_zone(self.match_state.ball_position)
        if ball_zone == "att":
            base_prob *= 2.0  # More fouls/cards in attacking areas
        elif ball_zone == "mid":
            base_prob *= 1.5  # Some fouls in midfield
        
        return np.random.random() < base_prob
    
    def _generate_referee_event(self) -> None:
        """Generate a referee event (foul, card, offside, etc.)."""
        # Select event type based on match situation
        referee_events = ["foul", "yellow_card", "offside", "throw_in", "corner", "free_kick"]
        weights = [0.4, 0.15, 0.2, 0.1, 0.1, 0.05]
        
        event_type = np.random.choice(referee_events, p=weights)
        
        # Find a player involved in the incident (for location context)
        all_players = self.home_team.players + self.away_team.players
        # Prefer players near the ball for realism
        distances = [p.position.distance_to(self.match_state.ball_position) for p in all_players]
        nearby_players = [p for i, p in enumerate(all_players) if distances[i] < 15.0]
        
        if nearby_players:
            involved_player = np.random.choice(nearby_players)
        else:
            involved_player = np.random.choice(all_players)
        
        # Get the zone where the incident occurs (based on involved player location)
        zone = self._get_team_relative_zone(involved_player)
        
        # Log the referee event with referee as the actor
        self.event_logger.log_event(
            action=event_type,
            team_id=self.home_team.team_id,  # Referee events assigned to home team by convention
            player_id="referee",
            player_name="Referee",
            player_role="referee",
            position_x=involved_player.position.x,  # Location where incident occurs
            position_y=involved_player.position.y,
            zone=zone,
            possession_team=self.match_state.current_possession_team or involved_player.team_id,
            phase=self._get_team_by_id(involved_player.team_id).current_phase,
            pressure_level=0.0,  # No pressure on referee
            success=True,  # Referee events are facts, not success/failure
            xg_value=0.0,  # No xG for referee events
            pass_accuracy=0.0,  # Not applicable
            opponent_distance=50.0,  # Referee is neutral
            teammate_support=0,  # Referee has no teammates
            stamina_level=1.0  # Referee doesn't tire
        )
        
        # Handle possession changes for some referee events
        if event_type in ["foul", "offside"]:
            # Possession changes to the other team
            self._turnover()
        elif event_type in ["throw_in", "corner", "free_kick"]:
            # Possession stays with current team or gets specific assignment
            # Keep current possession or assign based on context
            pass
    
    def _calculate_event_probability(self) -> float:
        """Calculate probability of event occurring in this time step."""
        # Increased base probability to ensure more player activity
        base_prob = 0.04  # 4% chance per 100ms step (doubled for more events)
        
        # Adjust based on ball zone
        zone = PitchZones.get_zone(self.match_state.ball_position)
        zone_multipliers = {"def": 0.8, "mid": 1.0, "att": 1.5}
        
        return base_prob * zone_multipliers.get(zone, 1.0)
        
        return base_prob * zone_multipliers.get(zone, 1.0)
    
    def _determine_event_type(self) -> str:
        """Determine what type of event should occur."""
        # Get the player who would execute the event to factor in their role
        executing_player = self._get_ball_nearest_player()
        if not executing_player:
            return "pass"
        
        # Special handling for goalkeepers - restrict inappropriate actions
        if executing_player.role == PlayerRole.GOALKEEPER:
            # Goalkeepers should primarily do: pass, clearance, save, distribute_ball
            events = ["pass", "clearance", "save", "distribute_ball", "header"]
            weights = [0.4, 0.25, 0.2, 0.1, 0.05]
            return np.random.choice(events, p=weights)
        
        # Special handling for strikers - they should shoot frequently but intelligently
        if executing_player.role == PlayerRole.STRIKER:
            zone = self._get_team_relative_zone(executing_player)
            
            # Calculate distance to goal to determine shooting intelligently
            if executing_player.team_id == self.home_team.team_id:
                goal_pos = Position(105, 34)  # HOME attacks towards x=105
            else:
                goal_pos = Position(0, 34)    # AWAY attacks towards x=0
            
            distance_to_goal = executing_player.position.distance_to(goal_pos)
            
            # Only shoot when in good positions (close to goal)
            if zone == "att" and distance_to_goal < 25:
                # Close to goal in attacking zone - high shot probability
                events = ["shot", "pass", "dribble", "header"]
                weights = [0.7, 0.15, 0.1, 0.05]
            elif zone == "att":
                # In attacking zone but far from goal - more passing/dribbling
                events = ["pass", "shot", "dribble", "header"]
                weights = [0.4, 0.35, 0.15, 0.1]
            else:
                # Not in attacking zone - focus on getting forward
                events = ["pass", "dribble", "header", "shot"]
                weights = [0.5, 0.3, 0.15, 0.05]
            
            return np.random.choice(events, p=weights)
        
        # Get team-relative zone for other players
        zone = self._get_team_relative_zone(executing_player)
        
        if zone == "att":
            # In attacking third
            if executing_player.role == PlayerRole.WINGER:
                events = ["cross", "dribble", "shot", "pass"]
                weights = [0.4, 0.25, 0.2, 0.15]
            elif executing_player.role == PlayerRole.ATTACKING_MIDFIELDER:
                events = ["shot", "pass", "dribble", "cross"]
                weights = [0.35, 0.35, 0.2, 0.1]
            else:
                events = ["shot", "pass", "cross", "dribble", "tackle"]
                weights = [0.2, 0.4, 0.2, 0.1, 0.1]
        elif zone == "mid":
            # In midfield
            if executing_player.role == PlayerRole.ATTACKING_MIDFIELDER:
                events = ["pass", "shot", "dribble", "tackle"]
                weights = [0.4, 0.3, 0.2, 0.1]
            elif executing_player.role == PlayerRole.CENTRE_MIDFIELDER:
                events = ["pass", "dribble", "tackle", "shot"]
                weights = [0.5, 0.25, 0.2, 0.05]
            elif executing_player.role == PlayerRole.FULLBACK:
                events = ["pass", "cross", "tackle", "dribble"]
                weights = [0.5, 0.25, 0.15, 0.1]
            else:
                events = ["pass", "dribble", "tackle", "interception"]
                weights = [0.5, 0.2, 0.2, 0.1]
        else:  # defensive zone
            # In defensive zone, non-goalkeepers can still do various actions
            if executing_player.role == PlayerRole.CENTRE_BACK:
                events = ["pass", "clearance", "tackle", "header"]
                weights = [0.45, 0.25, 0.2, 0.1]
            elif executing_player.role == PlayerRole.FULLBACK:
                events = ["pass", "clearance", "tackle", "cross"]
                weights = [0.5, 0.25, 0.15, 0.1]
            else:
                events = ["pass", "clearance", "tackle", "interception"]
                weights = [0.4, 0.3, 0.2, 0.1]
        
        return np.random.choice(events, p=weights)
    
    def _get_team_relative_zone(self, player: Player) -> str:
        """Get zone relative to player's team attacking direction."""
        player_x = player.position.x
        
        # Determine which team the player belongs to
        is_home_player = player in self.home_team.players
        
        if is_home_player:
            # HOME team attacks towards x=105
            if player_x >= 70:  # x > 70 is attacking zone for HOME
                return "att"
            elif player_x >= 35:  # 35 <= x <= 70 is midfield for HOME
                return "mid"
            else:  # x < 35 is defensive zone for HOME
                return "def"
        else:
            # AWAY team attacks towards x=0
            if player_x <= 35:  # x < 35 is attacking zone for AWAY
                return "att"
            elif player_x <= 70:  # 35 <= x <= 70 is midfield for AWAY
                return "mid"
            else:  # x > 70 is defensive zone for AWAY
                return "def"
    
    def _execute_event(self, event_type: str) -> None:
        """Execute a specific match event."""
        # Get ball-nearest player to execute event
        executing_player = self._get_ball_nearest_player()
        if not executing_player:
            return
        
        # Safety check: prevent goalkeepers from inappropriate actions
        if executing_player.role == PlayerRole.GOALKEEPER:
            inappropriate_gk_actions = ["dribble", "cross", "shot"]
            if event_type in inappropriate_gk_actions:
                # Replace with appropriate goalkeeper action
                if event_type == "shot":
                    event_type = "save"  # Convert shot to save
                elif event_type == "dribble":
                    event_type = "pass"  # Convert dribble to pass
                elif event_type == "cross":
                    event_type = "clearance"  # Convert cross to clearance
        
        # Special case: if ball is in defensive third and it's a save event,
        # ensure the defending goalkeeper is involved
        ball_zone = PitchZones.get_zone(self.match_state.ball_position)
        if event_type == "save" and ball_zone == "def":
            # Find the appropriate goalkeeper based on ball position and team zones
            ball_x = self.match_state.ball_position.x
            
            # Determine which team's defensive zone the ball is in
            if ball_x < 35:  # Home team's defensive zone (attacking towards x=105)
                gk = next((p for p in self.home_team.players if p.role == PlayerRole.GOALKEEPER), None)
            elif ball_x > 70:  # Away team's defensive zone (attacking towards x=0)  
                gk = next((p for p in self.away_team.players if p.role == PlayerRole.GOALKEEPER), None)
            else:
                # Midfield - choose based on current possession or randomly
                gk = None
            
            if gk:
                executing_player = gk
        
        # Check safety constraints
        game_state = self._get_game_state()
        if not executing_player.check_safety_constraints(event_type, game_state):
            return
        
        # Execute event based on type
        if event_type == "pass":
            self._execute_pass(executing_player)
        elif event_type == "shot":
            self._execute_shot(executing_player)
        elif event_type == "tackle":
            self._execute_tackle(executing_player)
        elif event_type == "cross":
            self._execute_cross(executing_player)
        elif event_type == "dribble":
            self._execute_dribble(executing_player)
        elif event_type == "clearance":
            self._execute_clearance(executing_player)
        elif event_type == "save":
            self._execute_save(executing_player)
        elif event_type == "interception":
            self._execute_interception(executing_player)
        elif event_type == "header":
            self._execute_header(executing_player)
        elif event_type == "distribute_ball":
            self._execute_distribute_ball(executing_player)
        
        # Apply action cooldown
        executing_player.apply_action_cooldown(event_type)
    
    def _log_event_with_zone(self, player: Player, action: str, **kwargs) -> None:
        """Helper method to log event with team-relative zone."""
        zone = self._get_team_relative_zone(player)
        self.event_logger.log_event(
            action=action,
            team_id=player.team_id,
            player_id=player.player_id,
            player_name=player.name,
            player_role=player.role.value,
            position_x=player.position.x,
            position_y=player.position.y,
            zone=zone,
            **kwargs
        )
    
    def _execute_pass(self, player: Player) -> None:
        """Execute a pass attempt."""
        # Find pass target
        teammates = self._get_teammates(player)
        if not teammates:
            return
            
        target_player = self._choose_pass_target(player, teammates)
        if not target_player:
            return
        
        # Calculate pass success probability
        distance = player.position.distance_to(target_player.position)
        pressure = self._calculate_pressure(player)
        success_prob = self._calculate_pass_success(player, distance, pressure)
        success = np.random.random() < success_prob
        
        # Calculate additional metrics
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        
        # Log event
        self._log_event_with_zone(
            player=player,
            action="pass",
            possession_team=player.team_id,  # Fix: Use player's team, not current possession team
            phase=self._get_team_by_id(player.team_id).current_phase,
            pressure_level=pressure,
            success=success,
            pass_accuracy=success_prob,
            xg_value=0.0,  # No xG for passes
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        # Update match state
        if success:
            self._transfer_possession(target_player)
            self._get_team_by_id(player.team_id).update_stats("pass", success=True)
        else:
            self._turnover()
            self._get_team_by_id(player.team_id).update_stats("pass", success=False)
    
    def _execute_shot(self, player: Player) -> None:
        """Execute a shot attempt."""
        # Calculate xG
        defenders = [p.position for p in self._get_opponents(player)]
        xg_value = self.xg_model.expected_goal(player.position, defenders)
        
        # Determine if goal scored
        goal_scored = np.random.random() < xg_value
        
        # Calculate additional metrics
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        # Log shot event
        self._log_event_with_zone(
            player=player,
            action="shot",
            possession_team=player.team_id,  # Fix: Use player's team, not current possession team
            phase="attack",
            pressure_level=pressure,
            success=goal_scored,
            xg_value=xg_value,
            pass_accuracy=0.0,  # Not applicable for shots
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        # Update statistics
        self.shot_stats[player.team_id].append(xg_value)
        self._get_team_by_id(player.team_id).update_stats("shot")
        
        if goal_scored:
            self._score_goal(player.team_id)
        else:
            # Ball goes to opponent (goalkeeper)
            self._turnover()
    
    def _execute_tackle(self, player: Player) -> None:
        """Execute a tackle attempt."""
        # Find opponent to tackle
        opponents = self._get_opponents(player)
        nearby_opponents = [p for p in opponents if player.position.distance_to(p.position) < 3.0]
        
        if not nearby_opponents:
            return
            
        target_opponent = nearby_opponents[0]
        
        # Calculate tackle success
        success_prob = self._calculate_tackle_success(player, target_opponent)
        success = np.random.random() < success_prob
        
        # Calculate additional metrics
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        # Log event
        self._log_event_with_zone(
            player=player,
            action="tackle",
            possession_team=self.match_state.current_possession_team,
            phase="defend",
            pressure_level=pressure,
            success=success,
            xg_value=0.0,  # No xG for tackles
            pass_accuracy=0.0,  # Not applicable for tackles
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        if success:
            self._transfer_possession(player)
    
    def _execute_save(self, player: Player) -> None:
        """Execute a goalkeeper save."""
        # Only goalkeepers can make saves
        if player.role != PlayerRole.GOALKEEPER:
            return
        
        save_prob = player.attributes.defending / 100.0 * 0.8
        success = np.random.random() < save_prob
        
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        self._log_event_with_zone(
            player=player,
            action="save",
            possession_team=player.team_id if success else self.match_state.current_possession_team,
            phase="defend",
            pressure_level=pressure,
            success=success,
            xg_value=0.0,  # No xG for saves
            pass_accuracy=0.0,  # Not applicable
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        if success:
            self._transfer_possession(player)
        else:
            self._turnover()
    
    def _execute_interception(self, player: Player) -> None:
        """Execute an interception."""
        success_prob = player.attributes.defending / 100.0 * 0.6
        success = np.random.random() < success_prob
        
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        self._log_event_with_zone(
            player=player,
            action="interception",
            possession_team=player.team_id if success else self.match_state.current_possession_team,
            phase="defend",
            pressure_level=pressure,
            success=success,
            xg_value=0.0,  # No xG for interceptions
            pass_accuracy=0.0,  # Not applicable
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        if success:
            self._transfer_possession(player)
    
    def _calculate_pass_success(self, player: Player, distance: float, pressure: float) -> float:
        """Calculate pass success probability."""
        base_success = player.attributes.passing / 100.0
        
        # Distance penalty
        distance_penalty = min(0.3, distance / 50.0)
        
        # Pressure penalty
        pressure_penalty = pressure * 0.2
        
        success_prob = base_success - distance_penalty - pressure_penalty
        return np.clip(success_prob, 0.1, 0.95)
    
    def _calculate_tackle_success(self, tackler: Player, target: Player) -> float:
        """Calculate tackle success probability."""
        tackle_skill = tackler.attributes.defending / 100.0
        dribble_skill = target.attributes.pace / 100.0
        
        success_prob = tackle_skill - (dribble_skill * 0.5)
        return np.clip(success_prob, 0.2, 0.8)
    
    def _get_ball_nearest_player(self) -> Optional[Player]:
        """Get player closest to ball, with enhanced randomness for better activity distribution."""
        all_players = self.home_team.players + self.away_team.players
        
        if not all_players:
            return None
            
        distances = [p.position.distance_to(self.match_state.ball_position) for p in all_players]
        
        # Track player activity levels (simplified using time since last action)
        current_time = self.match_state.time_elapsed
        player_activity_scores = []
        
        for i, player in enumerate(all_players):
            distance_score = 1.0 / (distances[i] + 1.0)  # Closer = higher score
            
            # Boost score for less active players (using last action time)
            time_since_action = current_time - getattr(player, 'last_action_time', 0.0)
            activity_boost = min(2.0, time_since_action / 60.0)  # Up to 2x boost after 60s
            
            # Role-based activity requirement boost
            role_boost = 1.0
            if player.role.value in ['goalkeeper']:
                role_boost = 1.5  # GKs need more activity
            elif player.role.value in ['striker', 'winger']:
                role_boost = 1.2  # Attackers should be active
            
            total_score = distance_score * (1.0 + activity_boost) * role_boost
            player_activity_scores.append(total_score)
        
        # 30% of the time, select purely by distance (tactical realism)
        # 70% of the time, use activity-balanced selection
        if np.random.random() < 0.3:
            nearest_idx = np.argmin(distances)
            selected_player = all_players[nearest_idx]
        else:
            # Select using activity-balanced scores
            total_score = sum(player_activity_scores)
            if total_score > 0:
                probabilities = [score / total_score for score in player_activity_scores]
                selected_idx = np.random.choice(len(all_players), p=probabilities)
                selected_player = all_players[selected_idx]
            else:
                # Fallback to nearest
                nearest_idx = np.argmin(distances)
                selected_player = all_players[nearest_idx]
        
        # Update last action time
        selected_player.last_action_time = current_time
        return selected_player
    
    def _get_teammates(self, player: Player) -> List[Player]:
        """Get teammates of a player."""
        team = self._get_team_by_id(player.team_id)
        return [p for p in team.players if p.player_id != player.player_id]
    
    def _get_opponents(self, player: Player) -> List[Player]:
        """Get opponents of a player."""
        if player.team_id == self.home_team.team_id:
            return self.away_team.players
        else:
            return self.home_team.players
    
    def _get_team_by_id(self, team_id: str) -> Team:
        """Get team by ID."""
        if team_id == self.home_team.team_id:
            return self.home_team
        else:
            return self.away_team
    
    def _transfer_possession(self, new_player: Player) -> None:
        """Transfer ball possession to a new player."""
        # Clear previous possession
        for team in [self.home_team, self.away_team]:
            for player in team.players:
                player.has_ball = False
        
        # Set new possession
        new_player.has_ball = True
        self.match_state.ball_player = new_player.player_id
        self.match_state.ball_team = new_player.team_id
        self.match_state.ball_position = new_player.position
        
        # Update possession team if changed
        if self.match_state.current_possession_team != new_player.team_id:
            self.match_state.current_possession_team = new_player.team_id
            self.match_state.possession_start_time = self.match_state.time_elapsed
    
    def _turnover(self) -> None:
        """Handle possession turnover."""
        # Find nearest opponent to give possession
        current_team = self.match_state.current_possession_team
        
        if current_team == self.home_team.team_id:
            new_team_players = self.away_team.players
        else:
            new_team_players = self.home_team.players
        
        # Give to nearest opponent
        distances = [p.position.distance_to(self.match_state.ball_position) for p in new_team_players]
        nearest_idx = np.argmin(distances)
        new_player = new_team_players[nearest_idx]
        
        self._transfer_possession(new_player)
    
    def _score_goal(self, scoring_team: str) -> None:
        """Handle goal scoring."""
        if scoring_team == self.home_team.team_id:
            self.match_state.home_score += 1
        else:
            self.match_state.away_score += 1
        
        # Log goal event with complete data
        self.event_logger.log_event(
            action="goal",
            team_id=scoring_team,
            player_id="team",
            player_name="Team",
            player_role="team",
            position_x=self.match_state.ball_position.x,
            position_y=self.match_state.ball_position.y,
            possession_team=scoring_team,
            phase="attack",
            xg_value=0.0,  # Goals don't have separate xG
            pass_accuracy=0.0,  # Not applicable
            opponent_distance=5.0,  # Goal line
            teammate_support=5,  # Assume good support for goal
            stamina_level=0.8,
            zone="att"  # Goals are scored in attacking zone
        )
        
        # Update team stats
        self._get_team_by_id(scoring_team).update_stats("goal")
        
        # Restart with kickoff
        self._kickoff(receiving_team=scoring_team)
    
    def _kickoff(self, receiving_team: Optional[str] = None) -> None:
        """Initialize or restart with kickoff."""
        # Reset ball to center
        self.match_state.ball_position = Position(52.5, 34.0)
        
        # Determine kickoff team
        if receiving_team is None:
            # Match start - random kickoff
            kickoff_team = np.random.choice([self.home_team.team_id, self.away_team.team_id])
        else:
            # After goal - non-scoring team kicks off
            kickoff_team = self.away_team.team_id if receiving_team == self.home_team.team_id else self.home_team.team_id
        
        # Give possession to kickoff team
        kickoff_player = self._get_team_by_id(kickoff_team).players[9]  # Striker
        self._transfer_possession(kickoff_player)
        
        # Log kickoff
        self._log_event_with_zone(
            player=kickoff_player,
            action="kickoff",
            possession_team=kickoff_team,
            phase="build_up",
            xg_value=0.0,  # No xG for kickoffs
            pass_accuracy=0.9,  # Kickoffs are usually accurate
            opponent_distance=10.0,  # Center circle
            teammate_support=5,
            stamina_level=1.0
        )
    
    def _half_time(self) -> None:
        """Handle half-time."""
        self.match_state.half = 2
        
        # Log half-time with home team for better balance
        self.event_logger.log_event(
            action="half_time",
            team_id=self.home_team.team_id,
            player_id="referee",
            player_name="Referee",
            player_role="referee",
            position_x=52.5,
            position_y=34.0,
            phase="transition",
            xg_value=0.0,  # No xG for referee events
            pass_accuracy=0.0,  # Not applicable
            opponent_distance=50.0,
            teammate_support=0,
            stamina_level=1.0,
            zone="mid"  # Center circle is midfield
        )
        
        print(f"Half-time: {self.home_team.name} {self.match_state.home_score}-{self.match_state.away_score} {self.away_team.name}")
    
    def _end_match(self) -> None:
        """Handle full-time."""
        self.event_logger.log_event(
            action="full_time",
            team_id=self.home_team.team_id,
            player_id="referee", 
            player_name="Referee",
            player_role="referee",
            position_x=52.5,
            position_y=34.0,
            phase="end",
            xg_value=0.0,  # No xG for referee events
            pass_accuracy=0.0,  # Not applicable
            opponent_distance=50.0,
            teammate_support=0,
            stamina_level=1.0,
            zone="mid"  # Center circle is midfield
        )
        
        print(f"Full-time: {self.home_team.name} {self.match_state.home_score}-{self.match_state.away_score} {self.away_team.name}")
    
    def _update_possession_stats(self) -> None:
        """Update possession time statistics."""
        if self.match_state.current_possession_team:
            team = self._get_team_by_id(self.match_state.current_possession_team)
            team.possession_time += self.TIME_STEP
            self.possession_stats[self.match_state.current_possession_team] += self.TIME_STEP
    
    def _get_game_state(self) -> Dict:
        """Get current game state for player decision making."""
        return {
            "time_elapsed": self.match_state.time_elapsed,
            "ball_position": self.match_state.ball_position,
            "possession_team": self.match_state.current_possession_team,
            "home_score": self.match_state.home_score,
            "away_score": self.match_state.away_score
        }
    
    def _generate_match_summary(self) -> Dict:
        """Generate comprehensive match summary."""
        total_time = self.match_state.time_elapsed
        
        return {
            "match_id": self.event_logger.match_id,
            "final_score": {
                "home": self.match_state.home_score,
                "away": self.match_state.away_score
            },
            "teams": {
                "home": self.home_team.name,
                "away": self.away_team.name
            },
            "possession": {
                "home": self.home_team.get_possession_percentage(total_time),
                "away": self.away_team.get_possession_percentage(total_time)
            },
            "shots": {
                "home": len(self.shot_stats[self.home_team.team_id]),
                "away": len(self.shot_stats[self.away_team.team_id])
            },
            "xg": {
                "home": sum(self.shot_stats[self.home_team.team_id]),
                "away": sum(self.shot_stats[self.away_team.team_id])
            },
            "pass_accuracy": {
                "home": self.home_team.get_pass_accuracy(),
                "away": self.away_team.get_pass_accuracy()
            },
            "event_log_stats": self.event_logger.get_summary_stats(),
            "tactical_patterns": self.event_logger.analyze_tactical_patterns()
        }
    
    def export_logs(self, output_dir: str = "logs") -> Tuple[str, str]:
        """
        Export match logs to CSV and XES formats.
        
        Returns:
            Tuple of (csv_path, xes_path)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f"match_{timestamp}.csv")
        xes_path = os.path.join(output_dir, f"match_{timestamp}.xes")
        
        self.event_logger.export_to_csv(csv_path)
        self.event_logger.export_to_xes(xes_path)
        
        return csv_path, xes_path
    
    # Additional event handlers for completeness
    def _execute_cross(self, player: Player) -> None:
        """Execute a cross attempt."""
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        self.event_logger.log_event(
            action="cross",
            team_id=player.team_id,
            player_id=player.player_id,
            player_name=player.name,
            player_role=player.role.value,
            position_x=player.position.x,
            position_y=player.position.y,
            possession_team=player.team_id,
            phase="attack",
            pressure_level=pressure,
            xg_value=0.0,  # No xG for crosses
            pass_accuracy=0.7,  # Default cross accuracy
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        # Simplified: 50% chance of maintaining possession
        if np.random.random() < 0.5:
            # Find teammate in box
            teammates = self._get_teammates(player)
            if teammates:
                target = np.random.choice(teammates)
                self._transfer_possession(target)
        else:
            self._turnover()
    
    def _execute_dribble(self, player: Player) -> None:
        """Execute a dribble attempt."""
        success_prob = player.attributes.pace / 100.0 * 0.7
        success = np.random.random() < success_prob
        
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        self._log_event_with_zone(
            player=player,
            action="dribble",
            possession_team=player.team_id,  # Fix: Use player's team consistently
            phase=self._get_team_by_id(player.team_id).current_phase,
            pressure_level=pressure,
            success=success,
            xg_value=0.0,  # No xG for dribbles
            pass_accuracy=0.0,  # Not applicable for dribbles
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        if not success:
            self._turnover()
    
    def _execute_clearance(self, player: Player) -> None:
        """Execute a clearance."""
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        self._log_event_with_zone(
            player=player,
            action="clearance",
            possession_team=self.match_state.current_possession_team,
            phase="defend",
            pressure_level=pressure,
            xg_value=0.0,  # No xG for clearances
            pass_accuracy=0.0,  # Not applicable for clearances
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        # Ball goes to random area upfield
        self._turnover()
    
    def _calculate_nearest_opponent_distance(self, player: Player) -> float:
        """Calculate distance to nearest opponent."""
        opponents = self._get_opponents(player)
        if not opponents:
            return 50.0  # Default large distance
        
        distances = [player.position.distance_to(opp.position) for opp in opponents]
        return min(distances)
    
    def _calculate_teammate_support(self, player: Player) -> int:
        """Calculate number of nearby teammates for support."""
        teammates = self._get_teammates(player)
        if not teammates:
            return 0
        
        nearby_teammates = [
            t for t in teammates 
            if player.position.distance_to(t.position) < 10.0
        ]
        return len(nearby_teammates)
    
    def _calculate_stamina_level(self, player: Player) -> float:
        """Calculate current stamina level as percentage."""
        return player.stamina_level
    
    def _calculate_pressure(self, player: Player) -> float:
        """Calculate pressure level on player."""
        opponents = self._get_opponents(player)
        opponent_positions = [opp.position for opp in opponents]
        return PitchZones.calculate_pressure(player.position, opponent_positions)
    
    def _choose_pass_target(self, player: Player, teammates: List[Player]) -> Optional[Player]:
        """Choose the best pass target from available teammates."""
        if not teammates:
            return None
        
        # Calculate scores for each teammate based on distance and position
        teammate_scores = []
        for teammate in teammates:
            distance = player.position.distance_to(teammate.position)
            
            # Distance score (prefer medium distances)
            if 3.0 <= distance <= 15.0:
                distance_score = 1.0
            elif distance <= 3.0:
                distance_score = 0.3  # Too close
            elif distance <= 25.0:
                distance_score = 0.7
            else:
                distance_score = 0.2  # Too far
            
            # Position score (prefer teammates ahead or in good positions)
            position_score = 0.5
            if player.team_id == self.home_team.team_id:
                # HOME attacks towards x=105
                if teammate.position.x > player.position.x:
                    position_score = 0.8  # Teammate is ahead
            else:
                # AWAY attacks towards x=0  
                if teammate.position.x < player.position.x:
                    position_score = 0.8  # Teammate is ahead
            
            total_score = distance_score * position_score
            teammate_scores.append((teammate, total_score))
        
        # Sort by score and select from top options with some randomness
        teammate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 3 options with weighted probability
        top_options = teammate_scores[:3]
        if top_options:
            teammates_list = [t[0] for t in top_options]
            weights = [t[1] for t in top_options]
            weights = np.array(weights) / sum(weights)  # Normalize
            return np.random.choice(teammates_list, p=weights)
        
        return teammates[0]  # Fallback
    
    def _execute_header(self, player: Player) -> None:
        """Execute a header attempt."""
        # Headers can be both defensive and offensive
        zone = self._get_team_relative_zone(player)
        success_prob = player.attributes.heading / 100.0 if hasattr(player.attributes, 'heading') else 0.6
        success = np.random.random() < success_prob
        
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        # Headers in attacking zone might be shots
        if zone == "att" and player.role == PlayerRole.STRIKER:
            # Treat as a shot attempt
            xg_value = self.xg_model.expected_goal(player.position, [p.position for p in self._get_opponents(player)])
            goal_scored = np.random.random() < xg_value
            
            self._log_event_with_zone(
                player=player,
                action="header",
                possession_team=self.match_state.current_possession_team,
                phase="attack",
                pressure_level=pressure,
                success=goal_scored,
                xg_value=xg_value,
                pass_accuracy=0.0,
                opponent_distance=opponent_distance,
                teammate_support=teammate_support,
                stamina_level=stamina_level
            )
            
            if goal_scored:
                self._score_goal(player.team_id)
            else:
                self._turnover()
        else:
            # Regular header (defensive or midfield)
            self._log_event_with_zone(
                player=player,
                action="header",
                possession_team=player.team_id if success else self.match_state.current_possession_team,
                phase=self._get_team_by_id(player.team_id).current_phase,
                pressure_level=pressure,
                success=success,
                xg_value=0.0,
                pass_accuracy=0.0,
                opponent_distance=opponent_distance,
                teammate_support=teammate_support,
                stamina_level=stamina_level
            )
            
            if not success:
                self._turnover()
    
    def _execute_distribute_ball(self, player: Player) -> None:
        """Execute a ball distribution (typically by goalkeeper)."""
        success_prob = 0.85  # Goalkeepers usually accurate with distribution
        success = np.random.random() < success_prob
        
        opponent_distance = self._calculate_nearest_opponent_distance(player)
        teammate_support = self._calculate_teammate_support(player)
        stamina_level = self._calculate_stamina_level(player)
        pressure = self._calculate_pressure(player)
        
        self._log_event_with_zone(
            player=player,
            action="distribute_ball",
            possession_team=player.team_id,
            phase="build_up",
            pressure_level=pressure,
            success=success,
            xg_value=0.0,
            pass_accuracy=success_prob,
            opponent_distance=opponent_distance,
            teammate_support=teammate_support,
            stamina_level=stamina_level
        )
        
        if success:
            # Find a teammate to distribute to
            teammates = self._get_teammates(player)
            if teammates:
                target = np.random.choice(teammates)
                self._transfer_possession(target)
        else:
            self._turnover()
