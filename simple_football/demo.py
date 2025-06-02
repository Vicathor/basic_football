#!/usr/bin/env python3
"""
🏈 Simple Football Simulation Demo
==================================

This demonstrates the complete football simulation and process mining pipeline.

Usage:
    python demo.py

Features showcased:
- Tactical 11v11 football simulation
- PM4Py-compatible event logs
- Process mining ready exports (CSV + XES)
- Role-based behavior modeling
- Expected Goals (xG) calculation
"""

import os
import sys
import time
import pandas as pd
import pm4py
from datetime import datetime

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.run_sim import simulate_single_match

def main():
    """Run demonstration simulation and analysis."""
    
    print("🏈 Simple Football Simulation - Process Mining Demo")
    print("=" * 60)
    
    # Setup
    demo_dir = "demo_output"
    os.makedirs(demo_dir, exist_ok=True)
    
    print(f"📁 Output directory: {demo_dir}")
    print("⚽ Starting match simulation...")
    
    # Run simulation
    start_time = time.time()
    
    try:
        # Simulate a single match
        result = simulate_single_match(
            random_seed=12345,
            verbose=True
        )
        
        simulation_time = time.time() - start_time
        
        print(f"\n✅ Simulation completed in {simulation_time:.2f} seconds")
        print(f"📊 Final Score: {result['teams']['home']} {result['final_score']['home']}-{result['final_score']['away']} {result['teams']['away']}")
        
        # Export logs manually since simulate_single_match doesn't do file export
        engine = result['engine']
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        match_id = f"match_{timestamp}"
        
        csv_file = os.path.join(demo_dir, f"{match_id}.csv")
        xes_file = os.path.join(demo_dir, f"{match_id}.xes")
        
        print(f"\n📁 Exporting event logs...")
        
        # Export CSV
        engine.event_logger.export_to_csv(csv_file)
        print(f"✅ CSV exported: {os.path.basename(csv_file)}")
        
        # Export XES  
        engine.event_logger.export_to_xes(xes_file)
        print(f"✅ XES exported: {os.path.basename(xes_file)}")
        
        print(f"\n📈 Analyzing event log: {os.path.basename(csv_file)}")
        
        # Load and analyze CSV
        df = pd.read_csv(csv_file)
        
        print(f"\n=== MATCH STATISTICS ===")
        print(f"Total events logged: {len(df):,}")
        print(f"Event types: {df['activity'].nunique()}")
        print(f"Players involved: {df['player_id'].nunique()}")
        print(f"Possession changes: {df['possession_chain_id'].nunique()}")
        
        print(f"\n=== EVENT BREAKDOWN ===")
        event_counts = df['activity'].value_counts().head(8)
        for event, count in event_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{event:12}: {count:3} ({percentage:4.1f}%)")
        
        print(f"\n=== TEAM PERFORMANCE ===")
        home_events = df[df['team_id'] == 'HOME']
        away_events = df[df['team_id'] == 'AWAY']
        
        home_possession = len(home_events) / (len(home_events) + len(away_events)) * 100
        away_possession = 100 - home_possession
        
        print(f"HOME possession: {home_possession:.1f}%")
        print(f"AWAY possession: {away_possession:.1f}%")
        
        # Shot analysis
        shots_df = df[df['activity'] == 'shot']
        if len(shots_df) > 0:
            home_shots = len(shots_df[shots_df['team_id'] == 'HOME'])
            away_shots = len(shots_df[shots_df['team_id'] == 'AWAY'])
            
            print(f"HOME shots: {home_shots}")
            print(f"AWAY shots: {away_shots}")
            
            # xG analysis
            if 'xg_value' in shots_df.columns:
                home_xg = shots_df[shots_df['team_id'] == 'HOME']['xg_value'].sum()
                away_xg = shots_df[shots_df['team_id'] == 'AWAY']['xg_value'].sum()
                print(f"HOME xG: {home_xg:.2f}")
                print(f"AWAY xG: {away_xg:.2f}")
        
        print(f"\n=== PROCESS MINING READY ===")
        print(f"✅ CSV export: {os.path.basename(csv_file)}")
        print(f"✅ XES export: {os.path.basename(xes_file)}")
        
        # Validate PM4Py compatibility
        try:
            event_log = pm4py.read_xes(xes_file)
            trace_count = len(event_log)
            
            if trace_count > 0:
                avg_events = sum(len(trace) for trace in event_log) / trace_count
                print(f"✅ PM4Py validation: {trace_count} trace(s), avg {avg_events:.1f} events/trace")
            else:
                print("⚠️  PM4Py validation: No traces found")
                
        except Exception as e:
            print(f"⚠️  PM4Py validation failed: {e}")
        
        print(f"\n=== PROCESS MINING CAPABILITIES ===")
        print("📊 Available analyses:")
        print("   • Tactical pattern discovery (formations, plays)")
        print("   • Bottleneck analysis (possession chains)")
        print("   • Player performance optimization")
        print("   • Transition risk assessment")
        print("   • xG-based decision effectiveness")
        
        print(f"\n=== DEMO COMPLETE ===")
        print(f"🎯 Simulation: {simulation_time:.1f}s (target: <2min)")
        print(f"📁 Files saved to: {os.path.abspath(demo_dir)}")
        print(f"📚 Ready for process mining analysis!")
        
        # Show sample data
        print(f"\n=== SAMPLE EVENT LOG ===")
        sample_cols = ['timestamp', 'activity', 'team_id', 'player_role', 'zone', 'success']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(5).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
