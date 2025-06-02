# Simple Football Simulation

A Python package for simulating 11-a-side football matches with process mining capabilities.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single match simulation
python scripts/run_sim.py --matches 1 --seed 42

# Run multiple matches
python scripts/run_sim.py --matches 10 --seed 42 --output-dir logs
```

## Usage in Code

```python
from simple_football_sim import simulate_matches

# Simulate 3 matches with reproducible results
simulate_matches(n_matches=3, random_seed=7, out_dir="logs")
```

## Process Mining Analysis

The simulation generates event logs compatible with PM4Py:

```python
import pm4py
import pandas as pd

# Load match data
df = pd.read_csv("logs/match_20250601_120000.csv")
event_log = pm4py.format_dataframe(df, case_id='match_id', activity_key='action', timestamp_key='timestamp')

# Discover process model
process_model = pm4py.discover_petri_net_alpha(event_log)
pm4py.view_petri_net(process_model[0], process_model[1], process_model[2])
```

## Architecture

- `engine/` - Core simulation components
- `logger/` - Event logging and PM4Py integration  
- `scripts/` - CLI tools and utilities
- `tests/` - Quality assurance and validation
- `notebooks/` - Analysis examples

## Key Features

- **Tactical Realism**: 11 distinct player roles with formation-based positioning
- **Process Mining Ready**: Events conform to PM4Py standards
- **Performance**: <2 minutes per 90-minute match on laptop
- **Extensible**: Clean interfaces for custom tactics and analytics
- **Quality Gates**: Built-in validation for schema completeness and rule compliance

## Output Files

Each match generates:
- `match_{timestamp}.csv` - Event log in PM4Py format
- `match_{timestamp}.xes` - XES format for advanced process mining tools

## Assumptions & Simplifications

1. Discrete 100ms time steps for computational efficiency
2. Simplified physics (no ball trajectory modeling)
3. Basic xG model based on distance, angle, and pressure
4. Deterministic formations with stochastic individual decisions
5. No injuries, substitutions, or weather effects
# simple
# simple
