"""
Leaderboard Update Script for GNN Molecular Graph Classification Challenge
==========================================================================

This script updates the leaderboard.md file with new submission scores
and optional efficiency metrics.

Usage:
    python update_leaderboard.py <submission_file> [--efficiency <score>] [--params <count>] [--time <ms>]
    
Environment Variables:
    ACTOR: GitHub username of the participant (set by GitHub Actions)
    
The script:
1. Computes the score for the submission
2. Updates the participant's entry in leaderboard.md (keeping best score)
3. Optionally records efficiency metrics (inference time, parameters)
4. Sorts the leaderboard by score (descending)

Leaderboard columns:
    - Rank: Position based on Macro F1 Score
    - Participant: GitHub username or team name
    - Macro-F1: Primary metric for ranking
    - Efficiency: F1² / (log₁₀(time_ms) × log₁₀(params)) 
    - Params: Model parameter count
    - Time (ms): Average inference time per batch
    - Last Updated: Submission date
"""

import os
import sys
import re
import math
import argparse
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score


def load_leaderboard(leaderboard_path: str) -> List[Dict[str, Any]]:
    """
    Load the current leaderboard from markdown file.
    
    Supports both old format (4 columns) and new format (7 columns with efficiency).
    
    Returns:
        list: List of entry dictionaries
    """
    entries = []
    
    if not os.path.exists(leaderboard_path):
        return entries
    
    with open(leaderboard_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (title, description, table header)
    in_table = False
    num_columns = 0
    
    for line in lines:
        line = line.strip()
        
        # Detect table start and column count
        if line.startswith('| Rank'):
            in_table = True
            num_columns = len([p for p in line.split('|') if p.strip()])
            continue
        
        # Skip separator line
        if line.startswith('|---') or line.startswith('| ---'):
            continue
        
        # Parse table rows
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 3:
                try:
                    participant = parts[1].strip().strip('*')  # Remove italic markers
                    score = float(parts[2].strip())
                    
                    entry = {
                        'participant': participant,
                        'score': score,
                        'efficiency': None,
                        'params': None,
                        'time_ms': None,
                        'cliff_accuracy': None,
                        'date': ''
                    }
                    
                    # Parse extended format (with efficiency metrics)
                    if num_columns >= 7 and len(parts) >= 6:
                        # Format: Rank | Participant | Macro-F1 | Efficiency | Params | Time | Cliff Acc | Date
                        eff_str = parts[3].strip()
                        if eff_str and eff_str != '-':
                            entry['efficiency'] = float(eff_str)
                        
                        params_str = parts[4].strip().replace(',', '').replace('K', '000').replace('M', '000000')
                        if params_str and params_str != '-':
                            entry['params'] = int(float(params_str))
                        
                        time_str = parts[5].strip()
                        if time_str and time_str != '-':
                            entry['time_ms'] = float(time_str)
                        
                        # Cliff accuracy column (index 6) if present
                        if len(parts) > 6:
                            cliff_str = parts[6].strip()
                            if cliff_str and cliff_str != '-':
                                try:
                                    entry['cliff_accuracy'] = float(cliff_str)
                                except ValueError:
                                    pass
                        
                        entry['date'] = parts[7].strip() if len(parts) > 7 else (parts[6].strip() if len(parts) > 6 and not parts[6].strip().replace('.', '').isdigit() else '')
                    else:
                        # Old format: Rank | Participant | Score | Date
                        entry['date'] = parts[3].strip() if len(parts) > 3 else ''
                    
                    entries.append(entry)
                except (ValueError, IndexError):
                    continue
    
    return entries


def format_params(params: Optional[int]) -> str:
    """Format parameter count with K/M suffix."""
    if params is None:
        return '-'
    if params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    elif params >= 1_000:
        return f"{params / 1_000:.1f}K"
    return str(params)


def save_leaderboard(leaderboard_path: str, entries: List[Dict[str, Any]]) -> None:
    """
    Save the leaderboard to markdown file with extended format.
    
    Args:
        leaderboard_path: Path to leaderboard.md
        entries: List of entry dictionaries
    """
    # Sort by score (descending)
    entries.sort(key=lambda x: x['score'], reverse=True)
    
    with open(leaderboard_path, 'w') as f:
        # Header
        f.write("# 🏆 Leaderboard\n\n")
        f.write("Competition: **GNN Molecular Graph Classification Challenge**\n\n")
        f.write("Primary Metric: **Macro F1 Score** (higher is better)\n\n")
        f.write("Efficiency Metric: $\\text{Efficiency} = \\frac{F_1^2}{\\log_{10}(\\text{time}_{ms}) \\times \\log_{10}(\\text{params})}$\n\n")
        f.write("---\n\n")
        
        # Table header (extended format)
        f.write("| Rank | Participant | Macro-F1 | Efficiency | Params | Time (ms) | Cliff Acc | Last Updated |\n")
        f.write("|------|-------------|----------|------------|--------|-----------|-----------|-----------------|\n")
        
        # Table rows
        for i, entry in enumerate(entries, 1):
            # Add medal emojis for top 3
            if i == 1:
                rank_str = "🥇 1"
            elif i == 2:
                rank_str = "🥈 2"
            elif i == 3:
                rank_str = "🥉 3"
            else:
                rank_str = str(i)
            
            # Mark baseline entries
            participant = entry['participant']
            if 'baseline' in participant.lower():
                participant = f"*{participant}*"
            
            # Format efficiency metrics
            eff_str = f"{entry['efficiency']:.4f}" if entry.get('efficiency') else '-'
            params_str = format_params(entry.get('params'))
            time_str = f"{entry['time_ms']:.1f}" if entry.get('time_ms') else '-'
            cliff_str = f"{entry['cliff_accuracy']:.4f}" if entry.get('cliff_accuracy') is not None else '-'
            
            f.write(f"| {rank_str} | {participant} | {entry['score']:.4f} | {eff_str} | {params_str} | {time_str} | {cliff_str} | {entry['date']} |\n")
        
        # Footer
        f.write("\n---\n\n")
        f.write("### Legend\n\n")
        f.write("- **Macro-F1**: Primary ranking metric (harmonic mean of class-wise F1 scores)\n")
        f.write("- **Efficiency**: Higher is better - rewards both accuracy and computational efficiency\n")
        f.write("- **Params**: Total number of trainable parameters\n")
        f.write("- **Time (ms)**: Average inference time per batch\n")
        f.write("- **Cliff Acc**: MMP-OOD Pairwise Cliff Accuracy — fraction of activity-cliff pairs correctly ranked\n\n")
        f.write("*Italic entries are baseline models provided by organizers.*\n\n")
        f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*\n")


def compute_submission_score(submission_file: str, truth_file: str) -> float:
    """
    Compute the macro F1 score for a submission.
    
    Args:
        submission_file: Path to submission CSV
        truth_file: Path to ground truth CSV
        
    Returns:
        float: Macro F1 score
    """
    submission_df = pd.read_csv(submission_file)
    truth_df = pd.read_csv(truth_file)
    
    # Merge on ID
    merged = truth_df.merge(submission_df, on='id', suffixes=('_true', '_pred'))
    
    y_true = merged['target_true'].values
    y_pred = merged['target_pred'].values
    
    return f1_score(y_true, y_pred, average='macro')


def compute_efficiency_score(f1: float, time_ms: float, params: int) -> float:
    """
    Compute efficiency score.
    
    Efficiency = F1² / (log₁₀(time_ms) × log₁₀(params))
    
    Args:
        f1: Macro F1 score
        time_ms: Inference time in milliseconds
        params: Number of model parameters
        
    Returns:
        Efficiency score (higher is better)
    """
    if f1 <= 0 or time_ms <= 0 or params <= 0:
        return 0.0
    
    time_ms = max(time_ms, 0.1)
    params = max(params, 100)
    
    log_time = math.log10(time_ms)
    log_params = math.log10(params)
    
    denominator = log_time * log_params
    if denominator <= 0:
        denominator = max(log_params, 1.0)
    
    return (f1 ** 2) / denominator


def main():
    parser = argparse.ArgumentParser(
        description='Update leaderboard with submission score and efficiency metrics'
    )
    parser.add_argument('submission_file', type=str, help='Path to submission CSV')
    parser.add_argument('--efficiency', type=float, default=None, help='Pre-computed efficiency score')
    parser.add_argument('--params', type=int, default=None, help='Model parameter count')
    parser.add_argument('--time', type=float, default=None, help='Inference time in ms')
    parser.add_argument('--cliff-acc', type=float, default=None, help='MMP-OOD cliff accuracy')
    parser.add_argument('--participant', type=str, default=None, help='Override participant name')
    
    args = parser.parse_args()
    
    submission_file = args.submission_file
    
    # Get participant name from args, environment, or filename
    participant = args.participant or os.environ.get('ACTOR')
    if not participant:
        # Extract from filename (e.g., "alice.csv" -> "alice")
        participant = os.path.basename(submission_file).replace('.csv', '')
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    leaderboard_path = os.path.join(script_dir, 'leaderboard.md')
    truth_file = os.path.join(script_dir, 'data', 'test_labels.csv')
    
    print(f"Updating leaderboard for participant: {participant}")
    print(f"Submission file: {submission_file}")
    
    # Check if ground truth exists
    if not os.path.exists(truth_file):
        print(f"Error: Ground truth file not found: {truth_file}")
        sys.exit(1)
    
    # Compute score
    try:
        score = compute_submission_score(submission_file, truth_file)
        print(f"Computed score: {score:.4f}")
    except Exception as e:
        print(f"Error computing score: {e}")
        sys.exit(1)
    
    # Compute efficiency if params and time provided
    efficiency = args.efficiency
    params = args.params
    time_ms = args.time
    cliff_acc = args.cliff_acc
    
    if efficiency is None and params and time_ms:
        efficiency = compute_efficiency_score(score, time_ms, params)
        print(f"Computed efficiency: {efficiency:.4f}")
    
    # Load current leaderboard
    entries = load_leaderboard(leaderboard_path)
    print(f"Current leaderboard has {len(entries)} entries")
    
    # Check if participant already exists
    existing_idx = None
    for i, entry in enumerate(entries):
        if entry['participant'].lower().strip('*') == participant.lower():
            existing_idx = i
            break
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    if existing_idx is not None:
        # Update existing entry if new score is better
        old_score = entries[existing_idx]['score']
        if score > old_score:
            print(f"Updating score: {old_score:.4f} -> {score:.4f}")
            entries[existing_idx]['score'] = score
            entries[existing_idx]['date'] = current_date
            # Update efficiency metrics
            if efficiency is not None:
                entries[existing_idx]['efficiency'] = efficiency
            if params is not None:
                entries[existing_idx]['params'] = params
            if time_ms is not None:
                entries[existing_idx]['time_ms'] = time_ms
            if cliff_acc is not None:
                entries[existing_idx]['cliff_accuracy'] = cliff_acc
        else:
            print(f"Keeping existing score: {old_score:.4f} (new score: {score:.4f})")
            # Still update efficiency if not set and now provided
            if entries[existing_idx].get('efficiency') is None and efficiency is not None:
                entries[existing_idx]['efficiency'] = efficiency
                entries[existing_idx]['params'] = params
                entries[existing_idx]['time_ms'] = time_ms
            if entries[existing_idx].get('cliff_accuracy') is None and cliff_acc is not None:
                entries[existing_idx]['cliff_accuracy'] = cliff_acc
    else:
        # Add new entry
        print(f"Adding new entry for {participant}")
        entries.append({
            'participant': participant,
            'score': score,
            'efficiency': efficiency,
            'params': params,
            'time_ms': time_ms,
            'cliff_accuracy': cliff_acc,
            'date': current_date
        })
    
    # Save updated leaderboard
    save_leaderboard(leaderboard_path, entries)
    print(f"Leaderboard updated successfully!")
    
    # Print current top 5
    entries.sort(key=lambda x: x['score'], reverse=True)
    print("\nTop 5 on leaderboard:")
    for i, entry in enumerate(entries[:5], 1):
        eff_str = f", eff={entry['efficiency']:.3f}" if entry.get('efficiency') else ""
        print(f"  {i}. {entry['participant']}: {entry['score']:.4f}{eff_str}")


if __name__ == "__main__":
    main()
