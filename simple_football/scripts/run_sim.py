"""
Main simulation script for running football matches.

Provides CLI interface and batch simulation capabilities.
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Any
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.match import MatchEngine
from engine.team import Team, TeamStrategy


def create_default_teams() -> tuple[Team, Team]:
    """Create default home and away teams for simulation."""
    # Home team
    home_strategy = TeamStrategy(
        formation="4-4-2",
        pressing_intensity=0.6,
        possession_style="balanced",
        transition_speed="medium"
    )
    
    home_team = Team(
        team_id="HOME", 
        name="Home United",
        side="home",
        strategy=home_strategy
    )
    
    # Away team  
    away_strategy = TeamStrategy(
        formation="4-4-2",
        pressing_intensity=0.5,
        possession_style="defensive", 
        transition_speed="fast"
    )
    
    away_team = Team(
        team_id="AWAY",
        name="Away City", 
        side="away",
        strategy=away_strategy
    )
    
    return home_team, away_team


def simulate_single_match(random_seed: int = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Simulate a single football match.
    
    Args:
        random_seed: Random seed for reproducibility
        verbose: Print detailed output
        
    Returns:
        Dict containing match results and statistics
    """
    if verbose:
        print(f"Setting up match with seed: {random_seed}")
    
    # Create teams
    home_team, away_team = create_default_teams()
    
    # Create and run match
    engine = MatchEngine(home_team, away_team, random_seed=random_seed)
    
    start_time = time.time()
    match_result = engine.simulate_match()
    simulation_time = time.time() - start_time
    
    if verbose:
        print(f"Simulation completed in {simulation_time:.2f} seconds")
        print_match_summary(match_result)
    
    return {
        **match_result,
        "simulation_time_seconds": simulation_time,
        "engine": engine  # For log export
    }


def simulate_matches(n_matches: int = 1, 
                    random_seed: int = None,
                    out_dir: str = "logs",
                    verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Simulate multiple football matches.
    
    Args:
        n_matches: Number of matches to simulate
        random_seed: Base random seed
        out_dir: Output directory for logs
        verbose: Print detailed output
        
    Returns:
        List of match results
    """
    if verbose:
        print(f"Starting simulation of {n_matches} matches")
        print(f"Output directory: {out_dir}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    total_start_time = time.time()
    
    for i in range(n_matches):
        if verbose:
            print(f"\n--- Match {i+1}/{n_matches} ---")
        
        # Use different seed for each match if base seed provided
        match_seed = random_seed + i if random_seed is not None else None
        
        # Simulate match
        result = simulate_single_match(random_seed=match_seed, verbose=verbose)
        
        # Export logs
        try:
            csv_path, xes_path = result["engine"].export_logs(out_dir)
            result["log_files"] = {"csv": csv_path, "xes": xes_path}
            
            if verbose:
                print(f"Logs exported to: {csv_path}, {xes_path}")
                
        except Exception as e:
            print(f"Warning: Failed to export logs for match {i+1}: {e}")
            result["log_files"] = {"error": str(e)}
        
        # Remove engine from result to avoid serialization issues
        del result["engine"]
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    if verbose:
        print(f"\n=== SIMULATION SUMMARY ===")
        print(f"Total matches: {n_matches}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per match: {total_time/n_matches:.2f} seconds")
        print_batch_summary(results)
    
    return results


def print_match_summary(result: Dict[str, Any]) -> None:
    """Print formatted match summary."""
    print(f"\n=== MATCH SUMMARY ===")
    print(f"Final Score: {result['teams']['home']} {result['final_score']['home']}-{result['final_score']['away']} {result['teams']['away']}")
    
    print(f"\nPossession:")
    print(f"  {result['teams']['home']}: {result['possession']['home']:.1f}%")
    print(f"  {result['teams']['away']}: {result['possession']['away']:.1f}%")
    
    print(f"\nShots:")
    print(f"  {result['teams']['home']}: {result['shots']['home']}")
    print(f"  {result['teams']['away']}: {result['shots']['away']}")
    
    print(f"\nExpected Goals (xG):")
    print(f"  {result['teams']['home']}: {result['xg']['home']:.2f}")
    print(f"  {result['teams']['away']}: {result['xg']['away']:.2f}")
    
    print(f"\nPass Accuracy:")
    print(f"  {result['teams']['home']}: {result['pass_accuracy']['home']:.1f}%")
    print(f"  {result['teams']['away']}: {result['pass_accuracy']['away']:.1f}%")
    
    if 'event_log_stats' in result:
        stats = result['event_log_stats']
        print(f"\nEvent Log Stats:")
        print(f"  Total events: {stats.get('total_events', 0)}")
        print(f"  Events per minute: {stats.get('events_per_minute', 0):.1f}")
        print(f"  Possession chains: {stats.get('possession_chains', 0)}")


def print_batch_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics across multiple matches."""
    if not results:
        return
        
    # Aggregate statistics
    total_goals = sum(r['final_score']['home'] + r['final_score']['away'] for r in results)
    avg_goals_per_match = total_goals / len(results)
    
    home_wins = sum(1 for r in results if r['final_score']['home'] > r['final_score']['away'])
    away_wins = sum(1 for r in results if r['final_score']['away'] > r['final_score']['home'])
    draws = len(results) - home_wins - away_wins
    
    avg_possession_home = sum(r['possession']['home'] for r in results) / len(results)
    avg_xg_home = sum(r['xg']['home'] for r in results) / len(results)
    avg_xg_away = sum(r['xg']['away'] for r in results) / len(results)
    
    print(f"\nResults Distribution:")
    print(f"  Home wins: {home_wins} ({home_wins/len(results)*100:.1f}%)")
    print(f"  Away wins: {away_wins} ({away_wins/len(results)*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/len(results)*100:.1f}%)")
    
    print(f"\nAverages per match:")
    print(f"  Goals: {avg_goals_per_match:.2f}")
    print(f"  Home possession: {avg_possession_home:.1f}%")
    print(f"  Home xG: {avg_xg_home:.2f}")
    print(f"  Away xG: {avg_xg_away:.2f}")


def validate_simulation_output(results: List[Dict[str, Any]]) -> bool:
    """
    Validate simulation meets quality requirements.
    
    Returns:
        bool: True if all quality gates passed
    """
    print("\n=== QUALITY VALIDATION ===")
    
    passed_checks = 0
    total_checks = 5
    
    # Check 1: All matches completed
    if len(results) > 0 and all('final_score' in r for r in results):
        print("✓ All matches completed successfully")
        passed_checks += 1
    else:
        print("✗ Some matches failed to complete")
    
    # Check 2: Reasonable simulation time (< 2 minutes per match)
    avg_sim_time = sum(r.get('simulation_time_seconds', 0) for r in results) / max(1, len(results))
    if avg_sim_time < 120:  # 2 minutes
        print(f"✓ Average simulation time: {avg_sim_time:.2f}s (< 2 min requirement)")
        passed_checks += 1
    else:
        print(f"✗ Average simulation time: {avg_sim_time:.2f}s (> 2 min requirement)")
    
    # Check 3: Event logs generated
    logs_generated = sum(1 for r in results if 'log_files' in r and 'csv' in r['log_files'])
    if logs_generated == len(results):
        print(f"✓ Event logs generated for all {len(results)} matches")
        passed_checks += 1
    else:
        print(f"✗ Event logs missing for {len(results) - logs_generated} matches")
    
    # Check 4: Reasonable event counts
    avg_events = 0
    if results and 'event_log_stats' in results[0]:
        avg_events = sum(r['event_log_stats'].get('total_events', 0) for r in results) / len(results)
        if 200 <= avg_events <= 2000:  # Reasonable range
            print(f"✓ Average events per match: {avg_events:.0f} (reasonable range)")
            passed_checks += 1
        else:
            print(f"✗ Average events per match: {avg_events:.0f} (outside reasonable range)")
    else:
        print("✗ Event statistics not available")
    
    # Check 5: xG values are reasonable
    avg_total_xg = sum(r['xg']['home'] + r['xg']['away'] for r in results) / max(1, len(results))
    if 1.0 <= avg_total_xg <= 4.0:  # Reasonable xG per match
        print(f"✓ Average total xG per match: {avg_total_xg:.2f} (reasonable range)")
        passed_checks += 1
    else:
        print(f"✗ Average total xG per match: {avg_total_xg:.2f} (outside reasonable range)")
    
    # Overall result
    success_rate = passed_checks / total_checks
    print(f"\nValidation Result: {passed_checks}/{total_checks} checks passed ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.8:
        print("✓ Quality requirements met")
        return True
    else:
        print("✗ Quality requirements not met")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Football Match Simulation')
    parser.add_argument('--matches', type=int, default=1, help='Number of matches to simulate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='logs', help='Output directory for logs')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--validate', action='store_true', help='Run quality validation')
    
    args = parser.parse_args()
    
    try:
        # Run simulation
        results = simulate_matches(
            n_matches=args.matches,
            random_seed=args.seed,
            out_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        # Run validation if requested
        if args.validate:
            validation_passed = validate_simulation_output(results)
            if not validation_passed:
                sys.exit(1)
        
        print(f"\nSimulation completed successfully!")
        print(f"Logs saved to: {os.path.abspath(args.output_dir)}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
