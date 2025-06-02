"""
Football event logging system for process mining analysis.

Generates PM4Py-compatible event logs with complete tactical context.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pm4py
from dataclasses import dataclass, asdict


@dataclass
class FootballEvent:
    """
    Single football event with complete contextual information.
    
    Conforms to PM4Py event log schema with football-specific extensions.
    """
    # PM4Py required fields
    timestamp: datetime
    case_id: str  # match_id
    activity: str  # action type
    
    # Football-specific context
    match_id: str
    team_id: str
    player_id: str
    player_name: str
    player_role: str
    
    # Spatial context
    position_x: float
    position_y: float
    zone: str  # def/mid/att
    
    # Tactical context
    possession_team: str
    phase: str  # build_up/attack/defend/transition
    pressure_level: float  # 0-1
    
    # Performance metrics
    xg_value: Optional[float] = None
    pass_accuracy: Optional[float] = None
    success: bool = True
    
    # Sequence tracking
    possession_chain_id: Optional[str] = None
    sequence_number: int = 0
    
    # Additional attributes for analysis
    opponent_distance: Optional[float] = None
    teammate_support: Optional[int] = None
    stamina_level: Optional[float] = None


class FootballEventLogger:
    """
    Event logger for football match simulation.
    
    Captures all match events in PM4Py-compatible format with tactical enrichment.
    Exports to both CSV and XES formats for process mining analysis.
    """
    
    def __init__(self, match_id: str):
        """
        Initialize event logger for a match.
        
        Args:
            match_id: Unique identifier for the match
        """
        self.match_id = match_id
        self.events: List[FootballEvent] = []
        self.current_possession_chain = None
        self.sequence_counter = 0
        
        # Match metadata
        self.match_start_time = datetime.now()
        self.current_time = self.match_start_time
        
    def log_event(self, 
                  action: str,
                  team_id: str,
                  player_id: str,
                  player_name: str,
                  player_role: str,
                  position_x: float,
                  position_y: float,
                  zone: Optional[str] = None,
                  **kwargs) -> None:
        """
        Log a single football event with tactical context.
        
        Args:
            action: Type of action (pass, shot, tackle, etc.)
            team_id: Team performing action
            player_id: Player performing action
            player_name: Player name
            player_role: Player tactical role
            position_x: X coordinate on pitch
            position_y: Y coordinate on pitch
            zone: Optional team-relative zone (def/mid/att), calculated if not provided
            **kwargs: Additional event attributes
        """
        # Determine zone from position if not provided
        if zone is None:
            from engine.pitch import PitchZones, Position
            zone = PitchZones.get_zone(Position(position_x, position_y))
        
        # Update possession chain if needed (before creating event)
        self._update_possession_chain(action, team_id)
        
        # Create event
        event = FootballEvent(
            timestamp=self.current_time,
            case_id=self.match_id,
            activity=action,
            match_id=self.match_id,
            team_id=team_id,
            player_id=player_id,
            player_name=player_name,
            player_role=player_role,
            position_x=position_x,
            position_y=position_y,
            zone=zone,
            possession_team=kwargs.get('possession_team', team_id),
            phase=kwargs.get('phase', 'unknown'),
            pressure_level=kwargs.get('pressure_level', 0.0),
            xg_value=kwargs.get('xg_value'),
            pass_accuracy=kwargs.get('pass_accuracy'),
            success=kwargs.get('success', True),
            possession_chain_id=self.current_possession_chain,
            sequence_number=self.sequence_counter,
            opponent_distance=kwargs.get('opponent_distance'),
            teammate_support=kwargs.get('teammate_support'),
            stamina_level=kwargs.get('stamina_level')
        )
        
        self.events.append(event)
        self.sequence_counter += 1
    
    def _update_possession_chain(self, action: str, team_id: str) -> None:
        """Update possession chain tracking."""
        # Start new possession chain on certain events
        if action in ['kickoff', 'throw_in', 'corner', 'free_kick', 'goal_kick']:
            self.current_possession_chain = f"{self.match_id}_{team_id}_{len(self.events)}"
        
        # End possession chain on turnovers
        elif action in ['interception', 'tackle_won', 'clearance']:
            if self.current_possession_chain and not self.current_possession_chain.endswith(team_id):
                self.current_possession_chain = f"{self.match_id}_{team_id}_{len(self.events)}"
    
    def advance_time(self, time_step: float) -> None:
        """
        Advance simulation time.
        
        Args:
            time_step: Time step in seconds
        """
        self.current_time += timedelta(seconds=time_step)
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export event log to CSV format.
        
        Args:
            filepath: Output file path
        """
        if not self.events:
            print("No events to export")
            return
            
        # Convert events to dictionaries
        event_dicts = [asdict(event) for event in self.events]
        
        # Create DataFrame
        df = pd.DataFrame(event_dicts)
        
        # Ensure required PM4Py columns
        df['case:concept:name'] = df['case_id']
        df['concept:name'] = df['activity']
        df['time:timestamp'] = df['timestamp']
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        print(f"Event log exported to {filepath}")
        
        # Validate export
        self._validate_export(df)
    
    def export_to_xes(self, filepath: str) -> None:
        """
        Export event log to XES format using PM4Py.
        
        Args:
            filepath: Output file path (.xes)
        """
        if not self.events:
            print("No events to export")
            return
            
        # Convert to DataFrame first
        event_dicts = [asdict(event) for event in self.events]
        df = pd.DataFrame(event_dicts)
        
        # Format for PM4Py
        df['case:concept:name'] = df['case_id']
        df['concept:name'] = df['activity'] 
        df['time:timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert to PM4Py event log
        event_log = pm4py.format_dataframe(
            df, 
            case_id='case:concept:name',
            activity_key='concept:name', 
            timestamp_key='time:timestamp'
        )
        
        # Export to XES
        pm4py.write_xes(event_log, filepath)
        print(f"XES event log exported to {filepath}")
    
    def _validate_export(self, df: pd.DataFrame) -> None:
        """Validate exported event log quality."""
        # Check completeness
        required_columns = ['timestamp', 'case_id', 'activity', 'team_id', 'player_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_percentage[missing_percentage > 1.0]
        
        if not high_missing.empty:
            print(f"WARNING: High missing values in columns: {high_missing.to_dict()}")
        
        # Check timestamp ordering
        if not df['timestamp'].is_monotonic_increasing:
            print("WARNING: Timestamps are not in chronological order")
        
        # Success metrics
        completeness = ((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns))) * 100
        print(f"Event log completeness: {completeness:.1f}%")
        
        if completeness >= 99.0:
            print("✓ Schema completeness requirement met (>99%)")
        else:
            print("✗ Schema completeness requirement failed (<99%)")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the event log."""
        if not self.events:
            return {}
            
        df = pd.DataFrame([asdict(event) for event in self.events])
        
        return {
            'total_events': len(self.events),
            'unique_activities': df['activity'].nunique(),
            'match_duration_minutes': (self.current_time - self.match_start_time).total_seconds() / 60,
            'events_per_minute': len(self.events) / max(1, (self.current_time - self.match_start_time).total_seconds() / 60),
            'possession_chains': df['possession_chain_id'].nunique(),
            'avg_chain_length': df.groupby('possession_chain_id').size().mean() if 'possession_chain_id' in df.columns else 0,
            'shots_logged': len(df[df['activity'] == 'shot']),
            'passes_logged': len(df[df['activity'] == 'pass']),
            'tackles_logged': len(df[df['activity'] == 'tackle'])
        }
    
    def analyze_tactical_patterns(self) -> Dict[str, float]:
        """
        Analyze tactical patterns in the event log.
        
        Returns:
            Dict with pattern frequencies and metrics
        """
        if not self.events:
            return {}
            
        df = pd.DataFrame([asdict(event) for event in self.events])
        
        # Pattern analysis
        patterns = {}
        
        # Build-up patterns
        buildup_events = df[df['phase'] == 'build_up']
        if len(buildup_events) > 0:
            patterns['buildup_pass_frequency'] = len(buildup_events[buildup_events['activity'] == 'pass']) / len(buildup_events)
        
        # Attacking patterns  
        attack_events = df[df['phase'] == 'attack']
        if len(attack_events) > 0:
            patterns['attack_shot_frequency'] = len(attack_events[attack_events['activity'] == 'shot']) / len(attack_events)
            patterns['attack_cross_frequency'] = len(attack_events[attack_events['activity'] == 'cross']) / len(attack_events)
        
        # Defensive patterns
        defend_events = df[df['phase'] == 'defend']
        if len(defend_events) > 0:
            patterns['defend_tackle_frequency'] = len(defend_events[defend_events['activity'] == 'tackle']) / len(defend_events)
            patterns['defend_clearance_frequency'] = len(defend_events[defend_events['activity'] == 'clearance']) / len(defend_events)
        
        # Transition patterns
        transition_events = df[df['phase'] == 'transition']
        if len(transition_events) > 0:
            patterns['transition_counter_frequency'] = len(transition_events[transition_events['activity'] == 'counter_attack']) / len(transition_events)
        
        return patterns
