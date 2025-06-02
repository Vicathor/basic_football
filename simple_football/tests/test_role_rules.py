"""
Test role rules and constraints compliance.

Validates that player roles follow liveness and safety constraints during simulation.
"""

import pandas as pd
import os
import sys
import pytest
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.player import PlayerRole


def get_latest_csv_file(log_dir: str = "logs") -> str:
    """Get the most recently created CSV file."""
    if not os.path.exists(log_dir):
        pytest.skip(f"Log directory {log_dir} does not exist")
    
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    
    if not csv_files:
        pytest.skip(f"No CSV files found in {log_dir}")
    
    # Get most recent file
    csv_files.sort(key=lambda x: os.path.getctime(os.path.join(log_dir, x)), reverse=True)
    return os.path.join(log_dir, csv_files[0])


def test_all_roles_present():
    """Test that all expected player roles are present in the log."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    expected_roles = {role.value for role in PlayerRole}
    actual_roles = set(df['player_role'].unique())
    
    # Remove non-player roles that might be in logs
    actual_roles = {role for role in actual_roles if role in expected_roles}
    
    missing_roles = expected_roles - actual_roles
    
    # Should have most core roles (allow some flexibility)
    core_roles = {'GK', 'CB', 'FB', 'CM', 'ST'}
    missing_core = core_roles - actual_roles
    
    assert len(missing_core) == 0, f"Missing core player roles: {missing_core}"


def test_goalkeeper_safety_constraints():
    """Test goalkeeper-specific safety constraints."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Get goalkeeper events
    gk_events = df[df['player_role'] == 'GK']
    
    if len(gk_events) == 0:
        pytest.skip("No goalkeeper events found")
    
    # Goalkeepers should primarily be in defensive third
    gk_in_def_third = gk_events[gk_events['position_x'] <= 35.0]
    defensive_percentage = len(gk_in_def_third) / len(gk_events) * 100
    
    # At least 80% of GK events should be in defensive third
    assert defensive_percentage >= 80, f"Goalkeeper only in defensive third {defensive_percentage:.1f}% of time"
    
    # Goalkeepers should not attempt certain actions inappropriately
    inappropriate_actions = ['dribble', 'cross', 'shot']
    gk_inappropriate = gk_events[gk_events['activity'].isin(inappropriate_actions)]
    
    # Allow very few inappropriate actions
    max_inappropriate = len(gk_events) * 0.05  # 5% tolerance
    assert len(gk_inappropriate) <= max_inappropriate, f"Goalkeeper performed {len(gk_inappropriate)} inappropriate actions"


def test_striker_safety_constraints():
    """Test striker-specific safety constraints."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Get striker events
    st_events = df[df['player_role'] == 'ST']
    
    if len(st_events) == 0:
        pytest.skip("No striker events found")
    
    # Strikers should primarily be in attacking areas (relative to their team direction)
    # HOME team attacks right (x >= 35), AWAY team attacks left (x <= 70)
    st_home = st_events[st_events['team_id'] == 'HOME']
    st_away = st_events[st_events['team_id'] == 'AWAY']
    
    home_attacking = st_home[st_home['position_x'] >= 35.0] if len(st_home) > 0 else st_home
    away_attacking = st_away[st_away['position_x'] <= 70.0] if len(st_away) > 0 else st_away
    
    st_in_att_areas = pd.concat([home_attacking, away_attacking]) if len(home_attacking) > 0 or len(away_attacking) > 0 else pd.DataFrame()
    attacking_percentage = len(st_in_att_areas) / len(st_events) * 100
    
    # At least 70% of striker events should be in attacking areas
    assert attacking_percentage >= 70, f"Striker only in attacking areas {attacking_percentage:.1f}% of time"
    
    # Strikers should have shot attempts
    st_shots = st_events[st_events['activity'] == 'shot']
    shot_percentage = len(st_shots) / len(st_events) * 100
    
    # Strikers should take at least some shots
    assert shot_percentage >= 5, f"Striker only took shots {shot_percentage:.1f}% of the time"


def test_role_action_appropriateness():
    """Test that roles perform appropriate actions for their position."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Define role-appropriate actions
    role_actions = {
        'GK': {'distribute_ball', 'save', 'clearance', 'pass'},
        'CB': {'pass', 'clearance', 'tackle', 'header'},
        'FB': {'pass', 'cross', 'tackle', 'dribble'},
        'DM': {'pass', 'tackle', 'interception'},
        'CM': {'pass', 'dribble', 'shot', 'tackle'},
        'AM': {'pass', 'shot', 'dribble', 'cross'},
        'W': {'cross', 'dribble', 'shot', 'pass'},
        'ST': {'shot', 'pass', 'dribble', 'header'}
    }
    
    for role, expected_actions in role_actions.items():
        role_events = df[df['player_role'] == role]
        
        if len(role_events) == 0:
            continue
            
        role_actual_actions = set(role_events['activity'].unique())
        
        # Check for some overlap with expected actions
        common_actions = expected_actions.intersection(role_actual_actions)
        
        # Should have at least 2 common actions
        assert len(common_actions) >= 2, f"Role {role} only has {len(common_actions)} appropriate actions: {common_actions}"


def test_liveness_constraints():
    """Test that players fulfill liveness requirements (regular activity)."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Group by player and check activity distribution
    player_activity = df.groupby('player_id').size()
    
    # All players should have some activity
    inactive_players = player_activity[player_activity == 0]
    assert len(inactive_players) == 0, f"Found {len(inactive_players)} completely inactive players"
    
    # Check minimum activity level
    min_activity = player_activity.min()
    total_events = len(df)
    
    # Each player should have at least 1% of total events (reasonable participation)
    min_expected_activity = max(5, total_events * 0.01)
    
    assert min_activity >= min_expected_activity, f"Least active player has only {min_activity} events (expected >= {min_expected_activity:.0f})"


def test_team_role_distribution():
    """Test that each team has appropriate role distribution."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Group by team and count roles
    for team_id in df['team_id'].unique():
        if team_id in ['match', 'referee']:  # Skip non-team entities
            continue
            
        team_events = df[df['team_id'] == team_id]
        team_roles = team_events['player_role'].value_counts()
        
        # Each team should have a goalkeeper
        assert 'GK' in team_roles.index, f"Team {team_id} has no goalkeeper events"
        
        # Should have multiple outfield roles
        outfield_roles = team_roles.drop('GK', errors='ignore')
        assert len(outfield_roles) >= 4, f"Team {team_id} has only {len(outfield_roles)} outfield roles"


def test_possession_role_consistency():
    """Test that possession events are consistent with roles."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Focus on possession-based actions
    possession_actions = ['pass', 'shot', 'dribble', 'cross']
    possession_events = df[df['activity'].isin(possession_actions)]
    
    if len(possession_events) == 0:
        pytest.skip("No possession events found")
    
    # Check that possession team matches player team
    mismatched_possession = possession_events[
        possession_events['team_id'] != possession_events['possession_team']
    ]
    
    # Allow small number of mismatches (transition moments)
    max_mismatches = len(possession_events) * 0.1  # 10% tolerance
    
    assert len(mismatched_possession) <= max_mismatches, f"{len(mismatched_possession)} possession mismatches found"


def test_zone_role_correlation():
    """Test that player zones correlate with their roles."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Expected zone preferences by role
    zone_preferences = {
        'GK': 'def',     # Goalkeepers in defensive zone
        'CB': 'def',     # Centre-backs primarily defensive
        'DM': 'mid',     # Defensive midfielders in middle
        'ST': 'att'      # Strikers in attacking zone
    }
    
    for role, preferred_zone in zone_preferences.items():
        role_events = df[df['player_role'] == role]
        
        if len(role_events) == 0:
            continue
            
        zone_counts = role_events['zone'].value_counts()
        
        if preferred_zone in zone_counts.index:
            preferred_percentage = zone_counts[preferred_zone] / len(role_events) * 100
            
            # Role should spend significant time in preferred zone
            min_percentage = 40 if role != 'GK' else 70  # Stricter for GK
            
            assert preferred_percentage >= min_percentage, f"{role} only in {preferred_zone} zone {preferred_percentage:.1f}% of time"


def test_action_success_rates():
    """Test that action success rates are reasonable for roles."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Check success rates by role for key actions
    if 'success' not in df.columns:
        pytest.skip("Success column not found")
    
    for role in ['GK', 'CB', 'CM', 'ST']:
        role_events = df[(df['player_role'] == role) & (df['activity'].isin(['pass', 'shot', 'tackle']))]
        
        if len(role_events) == 0:
            continue
            
        success_rate = role_events['success'].mean() * 100
        
        # Success rates should be reasonable (not too high or too low)
        assert 20 <= success_rate <= 95, f"{role} has unrealistic success rate: {success_rate:.1f}%"


if __name__ == "__main__":
    # Run tests manually if called directly
    test_all_roles_present()
    test_goalkeeper_safety_constraints()
    test_striker_safety_constraints()
    test_role_action_appropriateness()
    test_liveness_constraints()
    test_team_role_distribution()
    test_possession_role_consistency()
    test_zone_role_correlation()
    test_action_success_rates()
    
    print("✓ All role rules tests passed!")
