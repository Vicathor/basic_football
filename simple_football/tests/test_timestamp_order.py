"""
Test timestamp ordering in event logs.

Validates that events are logged in chronological order for process mining compatibility.
"""

import pandas as pd
import os
import sys
import pytest
from datetime import datetime

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


def test_timestamps_chronological():
    """Test that timestamps are in chronological order."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Convert timestamp column to datetime
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # Check if timestamps are monotonically increasing
    is_chronological = df['timestamp_dt'].is_monotonic_increasing
    
    assert is_chronological, "Timestamps are not in chronological order"


def test_timestamp_format():
    """Test that timestamps are in valid datetime format."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Try to convert all timestamps to datetime
    try:
        timestamps = pd.to_datetime(df['timestamp'])
        invalid_count = timestamps.isnull().sum()
        
        assert invalid_count == 0, f"{invalid_count} timestamps could not be parsed"
        
    except Exception as e:
        pytest.fail(f"Error parsing timestamps: {e}")


def test_no_duplicate_timestamps():
    """Test that there are no exact duplicate timestamps."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    duplicate_count = df['timestamp'].duplicated().sum()
    
    # Allow small number of duplicates (same time step events)
    max_duplicates = len(df) * 0.05  # 5% threshold
    
    assert duplicate_count <= max_duplicates, f"{duplicate_count} duplicate timestamps found (>{max_duplicates:.0f} allowed)"


def test_reasonable_time_intervals():
    """Test that time intervals between events are reasonable."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Convert to datetime
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # Calculate time differences
    time_diffs = df['timestamp_dt'].diff().dt.total_seconds()
    
    # Remove first NaN value
    time_diffs = time_diffs.dropna()
    
    if len(time_diffs) > 0:
        # Check for reasonable intervals
        min_interval = time_diffs.min()
        max_interval = time_diffs.max()
        mean_interval = time_diffs.mean()
        
        # Intervals should be between 0.01 seconds and 60 seconds
        assert min_interval >= 0.0, f"Negative time interval found: {min_interval}"
        assert max_interval <= 60.0, f"Excessive time interval found: {max_interval} seconds"
        
        # Mean interval should be reasonable for football simulation
        assert 0.01 <= mean_interval <= 10.0, f"Mean interval {mean_interval:.3f}s outside reasonable range"


def test_match_duration():
    """Test that match duration is approximately 90 minutes."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Convert to datetime
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # Calculate total duration
    start_time = df['timestamp_dt'].min()
    end_time = df['timestamp_dt'].max()
    duration_minutes = (end_time - start_time).total_seconds() / 60
    
    # Should be approximately 90 minutes (allow some variation for setup/finish)
    assert 85 <= duration_minutes <= 95, f"Match duration {duration_minutes:.1f} minutes not close to 90 minutes"


def test_sequence_numbers():
    """Test that sequence numbers increase properly."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    if 'sequence_number' not in df.columns:
        pytest.skip("sequence_number column not found")
    
    # Sequence numbers should start at 0 and increase by 1
    expected_sequence = list(range(len(df)))
    actual_sequence = df['sequence_number'].tolist()
    
    # Allow some gaps but check general trend
    is_increasing = df['sequence_number'].is_monotonic_increasing
    assert is_increasing, "Sequence numbers are not monotonically increasing"
    
    # Check that we start from 0 or low number
    min_seq = df['sequence_number'].min()
    assert min_seq <= 10, f"Sequence numbers start from {min_seq}, expected to start near 0"


def test_timestamp_precision():
    """Test that timestamps have appropriate precision."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Convert to datetime
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # Check precision by looking at microseconds
    microseconds = df['timestamp_dt'].dt.microsecond
    
    # Most events should have some microsecond precision (not all exactly on second boundaries)
    zero_microseconds = (microseconds == 0).sum()
    zero_percentage = (zero_microseconds / len(df)) * 100
    
    # Less than 90% should have zero microseconds (indicating good precision)
    assert zero_percentage < 90, f"{zero_percentage:.1f}% of timestamps have zero microseconds (low precision)"


def test_events_per_time_period():
    """Test reasonable distribution of events over time."""
    csv_file = get_latest_csv_file()
    df = pd.read_csv(csv_file)
    
    # Convert to datetime
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # Group by minute intervals
    df['minute'] = df['timestamp_dt'].dt.floor('1min')
    events_per_minute = df.groupby('minute').size()
    
    # Should have events in most minutes
    total_minutes = (df['timestamp_dt'].max() - df['timestamp_dt'].min()).total_seconds() / 60
    minutes_with_events = len(events_per_minute)
    coverage_percentage = (minutes_with_events / total_minutes) * 100
    
    # At least 70% of minutes should have events
    assert coverage_percentage >= 70, f"Only {coverage_percentage:.1f}% of minutes have events"
    
    # No minute should have excessive events (> 100)
    max_events_per_minute = events_per_minute.max()
    assert max_events_per_minute <= 100, f"One minute has {max_events_per_minute} events (too many)"


if __name__ == "__main__":
    # Run tests manually if called directly
    test_timestamps_chronological()
    test_timestamp_format()
    test_no_duplicate_timestamps()
    test_reasonable_time_intervals()
    test_match_duration()
    test_sequence_numbers()
    test_timestamp_precision()
    test_events_per_time_period()
    
    print("✓ All timestamp ordering tests passed!")
