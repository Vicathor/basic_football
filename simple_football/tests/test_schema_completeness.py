"""
Test schema completeness for event logs.

Validates that exported CSV files contain all required columns with minimal missing data.
"""

import pandas as pd
import os
import sys
import pytest
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def test_required_columns_present():
    """Test that all required PM4Py columns are present."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    required_columns = [
        'timestamp', 'case_id', 'activity', 'match_id', 'team_id', 
        'player_id', 'player_name', 'player_role', 'position_x', 
        'position_y', 'zone', 'possession_team', 'phase'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"


def test_schema_completeness():
    """Test that missing data is below 1% threshold."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Calculate completeness for each column
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    print(f"Schema completeness: {completeness:.2f}%")
    
    # Must be >= 99% complete
    assert completeness >= 99.0, f"Schema completeness {completeness:.2f}% below 99% threshold"


def test_critical_columns_no_nulls():
    """Test that critical columns have no null values."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    critical_columns = ['timestamp', 'case_id', 'activity', 'team_id', 'player_id']
    
    for col in critical_columns:
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Critical column '{col}' has {null_count} null values"


def test_data_types_correct():
    """Test that data types are appropriate."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Test numeric columns
    numeric_columns = ['position_x', 'position_y', 'pressure_level', 'sequence_number']
    for col in numeric_columns:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' should be numeric"
    
    # Test string columns
    string_columns = ['activity', 'team_id', 'player_role', 'zone', 'phase']
    for col in string_columns:
        if col in df.columns:
            assert pd.api.types.is_object_dtype(df[col]), f"Column '{col}' should be string/object type"


def test_coordinate_ranges():
    """Test that coordinates are within pitch boundaries."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Check x coordinates (0-105 meters)
    x_out_of_bounds = ((df['position_x'] < 0) | (df['position_x'] > 105)).sum()
    assert x_out_of_bounds == 0, f"{x_out_of_bounds} events have x coordinates outside pitch (0-105)"
    
    # Check y coordinates (0-68 meters)  
    y_out_of_bounds = ((df['position_y'] < 0) | (df['position_y'] > 68)).sum()
    assert y_out_of_bounds == 0, f"{y_out_of_bounds} events have y coordinates outside pitch (0-68)"


def test_minimum_event_count():
    """Test that sufficient events were logged."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Should have at least 200 events for a 90-minute match
    min_events = 200
    actual_events = len(df)
    
    assert actual_events >= min_events, f"Only {actual_events} events logged, expected at least {min_events}"


def test_team_balance():
    """Test that both teams have reasonable event representation."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    team_counts = df['team_id'].value_counts()
    
    # Should have at least 2 teams
    assert len(team_counts) >= 2, f"Only {len(team_counts)} teams found in events"
    
    # No team should have less than 20% of events (too imbalanced)
    min_percentage = 20.0
    for team, count in team_counts.items():
        percentage = (count / len(df)) * 100
        assert percentage >= min_percentage, f"Team {team} only has {percentage:.1f}% of events (< {min_percentage}%)"


if __name__ == "__main__":
    # Run tests manually if called directly
    test_required_columns_present()
    test_schema_completeness()
    test_critical_columns_no_nulls()
    test_data_types_correct()
    test_coordinate_ranges()
    test_minimum_event_count()
    test_team_balance()
    
    print("✓ All schema completeness tests passed!")
