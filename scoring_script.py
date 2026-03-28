"""
Scoring Script for GNN Molecular Graph Classification Challenge
================================================================

This script evaluates a participant's submission against the ground truth
test labels and computes the macro F1 score along with efficiency metrics.

Usage:
    python scoring_script.py <submission_file> [--metadata <metadata.yaml>]
    
Example:
    python scoring_script.py submissions/participant_submission.csv
    python scoring_script.py submissions/submission.csv --metadata submissions/metadata.yaml
    
The submission file should have two columns:
    - id: The molecule ID (matching those in test.csv)
    - target: The predicted label (0 or 1)

Optional metadata file (YAML) can include:
    - inference_time_ms: Average inference time per batch
    - total_params: Total model parameters
    - model_name: Name of the model architecture

Metrics Computed:
    - Macro F1 Score (primary metric)
    - Accuracy, Precision, Recall
    - Efficiency Score: F1² / (log₁₀(time_ms) × log₁₀(params))

Note: The ground truth labels are stored securely and not publicly available.
This script is used by GitHub Actions for automated evaluation.
"""

import sys
import os
import math
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Try to import yaml for metadata parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def validate_submission(submission_df, truth_df):
    """
    Validate that the submission file has the correct format.
    
    Args:
        submission_df: DataFrame with participant predictions
        truth_df: DataFrame with ground truth labels
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check required columns
    if 'id' not in submission_df.columns:
        return False, "Submission is missing 'id' column"
    
    if 'target' not in submission_df.columns:
        return False, "Submission is missing 'target' column"
    
    # Check number of rows
    if len(submission_df) != len(truth_df):
        return False, f"Submission has {len(submission_df)} rows, expected {len(truth_df)}"
    
    # Check that all IDs are present
    submission_ids = set(submission_df['id'].tolist())
    truth_ids = set(truth_df['id'].tolist())
    
    missing_ids = truth_ids - submission_ids
    if missing_ids:
        return False, f"Submission is missing {len(missing_ids)} IDs: {list(missing_ids)[:5]}..."
    
    extra_ids = submission_ids - truth_ids
    if extra_ids:
        return False, f"Submission has {len(extra_ids)} extra IDs: {list(extra_ids)[:5]}..."
    
    # Check that target values are valid (0 or 1)
    invalid_targets = submission_df[~submission_df['target'].isin([0, 1])]
    if len(invalid_targets) > 0:
        return False, f"Found {len(invalid_targets)} invalid target values (should be 0 or 1)"
    
    return True, ""


def compute_score(submission_df, truth_df):
    """
    Compute the macro F1 score for the submission.
    
    Args:
        submission_df: DataFrame with participant predictions
        truth_df: DataFrame with ground truth labels
        
    Returns:
        dict: Dictionary with various metrics
    """
    # Merge on ID to ensure alignment
    merged = truth_df.merge(submission_df, on='id', suffixes=('_true', '_pred'))
    
    y_true = merged['target_true'].values
    y_pred = merged['target_pred'].values
    
    # Compute metrics
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_class_0': f1_score(y_true, y_pred, pos_label=0, average='binary'),
        'f1_class_1': f1_score(y_true, y_pred, pos_label=1, average='binary'),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def load_metadata(metadata_path: str) -> Optional[Dict[str, Any]]:
    """
    Load submission metadata from YAML or JSON file.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Dictionary with metadata or None if file doesn't exist
    """
    if not os.path.exists(metadata_path):
        return None
    
    ext = Path(metadata_path).suffix.lower()
    
    with open(metadata_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                print("Warning: PyYAML not installed. Cannot parse YAML metadata.")
                return None
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            print(f"Warning: Unknown metadata format: {ext}")
            return None


def compute_efficiency_score(
    f1_score: float,
    inference_time_ms: float,
    total_params: int
) -> float:
    """
    Compute the efficiency score.
    
    Efficiency = F1² / (log₁₀(time_ms) × log₁₀(params))
    
    This metric balances prediction quality with computational cost:
    - Higher F1 → better efficiency
    - Lower inference time → better efficiency
    - Fewer parameters → better efficiency
    
    The logarithmic scaling on time and params ensures:
    - 10x speedup gives same benefit regardless of base speed
    - Model size differences are fairly weighted
    
    Args:
        f1_score: Macro F1 score
        inference_time_ms: Average inference time in milliseconds
        total_params: Total number of model parameters
        
    Returns:
        Efficiency score (higher is better)
    """
    # Handle edge cases
    if f1_score <= 0:
        return 0.0
    
    # Ensure positive values for log
    time_ms = max(inference_time_ms, 0.1)  # Minimum 0.1ms
    params = max(total_params, 100)  # Minimum 100 params
    
    # Compute log terms
    log_time = math.log10(time_ms)
    log_params = math.log10(params)
    
    # Handle edge case where log product is very small or zero
    denominator = log_time * log_params
    if denominator <= 0:
        # Use only params if time is < 1ms
        denominator = max(log_params, 1.0)
    
    efficiency = (f1_score ** 2) / denominator
    
    return round(efficiency, 6)


def main():
    parser = argparse.ArgumentParser(
        description='GNN Molecular Graph Classification Challenge - Scoring Script'
    )
    parser.add_argument(
        'submission_file',
        type=str,
        help='Path to submission CSV file'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='Path to metadata YAML/JSON file with efficiency metrics'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to output JSON file with all metrics'
    )
    parser.add_argument(
        '--pairs-csv',
        type=str,
        default=None,
        help='Path to MMP-OOD pairs CSV for pairwise cliff accuracy evaluation'
    )
    
    args = parser.parse_args()
    
    submission_file = args.submission_file
    
    # Check if submission file exists
    if not os.path.exists(submission_file):
        print(f"Error: Submission file not found: {submission_file}")
        sys.exit(1)
    
    # Load metadata if provided
    metadata = None
    if args.metadata:
        metadata = load_metadata(args.metadata)
        if metadata:
            print(f"Loaded metadata from: {args.metadata}")
    
    # Auto-detect metadata file if not provided
    if metadata is None:
        # Look for metadata file with same name as submission
        base_name = Path(submission_file).stem
        for ext in ['.yaml', '.yml', '.json']:
            meta_path = Path(submission_file).parent / f"{base_name}_metadata{ext}"
            if meta_path.exists():
                metadata = load_metadata(str(meta_path))
                if metadata:
                    print(f"Auto-detected metadata from: {meta_path}")
                    break
    
    # Load ground truth labels
    # In production, this file is stored securely and populated by GitHub Actions
    truth_file = os.path.join(os.path.dirname(__file__), 'data', 'test_labels.csv')
    
    if not os.path.exists(truth_file):
        print(f"Error: Ground truth file not found: {truth_file}")
        print("Note: This file is only available in the evaluation environment.")
        sys.exit(1)
    
    print("="*60)
    print("GNN Molecular Graph Classification Challenge - Scoring")
    print("="*60)
    
    # Load files
    print(f"\nLoading submission: {submission_file}")
    submission_df = pd.read_csv(submission_file)
    
    print(f"Loading ground truth: {truth_file}")
    truth_df = pd.read_csv(truth_file)
    
    # Validate submission
    print("\nValidating submission format...")
    is_valid, error_msg = validate_submission(submission_df, truth_df)
    
    if not is_valid:
        print(f"❌ Validation failed: {error_msg}")
        sys.exit(1)
    
    print("✅ Submission format is valid")
    
    # Compute score
    print("\nComputing metrics...")
    metrics = compute_score(submission_df, truth_df)
    
    # Compute efficiency score if metadata is available
    efficiency_score = None
    inference_time_ms = None
    total_params = None
    
    if metadata:
        inference_time_ms = metadata.get('inference_time_ms') or metadata.get('efficiency_metrics', {}).get('inference_time_ms')
        total_params = metadata.get('total_params') or metadata.get('efficiency_metrics', {}).get('total_params')
        
        if inference_time_ms and total_params:
            efficiency_score = compute_efficiency_score(
                metrics['macro_f1'],
                inference_time_ms,
                total_params
            )
            metrics['efficiency_score'] = efficiency_score
            metrics['inference_time_ms'] = inference_time_ms
            metrics['total_params'] = total_params
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n🎯 Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"\nAdditional metrics:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  - Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  - F1 (class 0): {metrics['f1_class_0']:.4f}")
    print(f"  - F1 (class 1): {metrics['f1_class_1']:.4f}")
    
    # Display efficiency metrics if available
    if efficiency_score is not None:
        print(f"\n⚡ Efficiency Metrics:")
        print(f"  - Inference Time: {inference_time_ms:.2f} ms")
        print(f"  - Parameters: {total_params:,}")
        print(f"  - Efficiency Score: {efficiency_score:.4f}")
        print(f"    (Formula: F1² / (log₁₀(time) × log₁₀(params)))")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  Predicted:    0      1")
    print(f"  Actual 0:  {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"  Actual 1:  {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    # --- MMP-OOD Pairwise Cliff Accuracy (optional) ---
    cliff_accuracy = None
    if args.pairs_csv:
        try:
            from evaluation.mmp_ood import load_pairs_csv, compute_cliff_accuracy_hard
            pairs = load_pairs_csv(args.pairs_csv)
            # Build prediction dict from submission
            merged = truth_df.merge(submission_df, on='id', suffixes=('_true', '_pred'))
            pred_dict = dict(zip(merged['id'].values, merged['target_pred'].values))
            cliff_acc, per_pair = compute_cliff_accuracy_hard(pairs, pred_dict)
            cliff_accuracy = cliff_acc
            metrics['cliff_accuracy'] = cliff_accuracy
            metrics['cliff_pairs_evaluated'] = len(per_pair)
            metrics['cliff_pairs_correct'] = sum(per_pair)
            
            print(f"\n🧬 MMP-OOD Pairwise Cliff Accuracy:")
            print(f"  - Cliff Accuracy: {cliff_accuracy:.4f}")
            print(f"  - Pairs evaluated: {len(per_pair)}")
            print(f"  - Pairs correct:   {sum(per_pair)}")
        except ImportError:
            print("\n⚠️  Could not load MMP-OOD module (rdkit not installed?)")
        except Exception as e:
            print(f"\n⚠️  MMP-OOD evaluation failed: {e}")
    
    print("\n" + "="*60)
    
    # Output the main score for GitHub Actions to capture
    # This line is parsed by update_leaderboard.py
    print(f"SCORE:{metrics['macro_f1']:.6f}")
    
    # Output efficiency metrics if available
    if efficiency_score is not None:
        print(f"EFFICIENCY:{efficiency_score:.6f}")
        print(f"PARAMS:{total_params}")
        print(f"TIME_MS:{inference_time_ms:.2f}")
    
    # Output cliff accuracy if computed
    if cliff_accuracy is not None:
        print(f"CLIFF_ACC:{cliff_accuracy:.6f}")
    
    # Write output JSON if requested
    if args.output_json:
        output_data = {
            'macro_f1': metrics['macro_f1'],
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_class_0': metrics['f1_class_0'],
            'f1_class_1': metrics['f1_class_1'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        if efficiency_score is not None:
            output_data['efficiency_score'] = efficiency_score
            output_data['inference_time_ms'] = inference_time_ms
            output_data['total_params'] = total_params
        if cliff_accuracy is not None:
            output_data['cliff_accuracy'] = cliff_accuracy
            output_data['cliff_pairs_evaluated'] = metrics.get('cliff_pairs_evaluated', 0)
            output_data['cliff_pairs_correct'] = metrics.get('cliff_pairs_correct', 0)
        
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nMetrics saved to: {args.output_json}")
    
    return metrics['macro_f1']


if __name__ == "__main__":
    score = main()
