"""
Train Advanced Baseline Models for GNN Competition
==================================================

This script trains D-MPNN and Spectral GNN models and generates submissions.

Usage:
    python train_advanced.py [--model {dmpnn,spectral,all}]
"""

import os
import sys
import random
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix for PyTorch 2.6+ compatibility with torch_geometric
import torch.serialization
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except (ImportError, AttributeError):
    pass

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import f1_score

# Import our advanced models
from dmpnn import DMPNNModel
from spectral_gnn import SpectralGNN

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device, is_spectral=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        if is_spectral:
            # Spectral GNN returns regularization loss
            out, reg_loss = model(batch, return_reg_loss=True)
            cls_loss = criterion(out, batch.y.squeeze().long())
            loss = cls_loss + model.laplacian_weight * reg_loss
            total_reg_loss += reg_loss.item() * batch.num_graphs
        else:
            out = model(batch)
            loss = criterion(out, batch.y.squeeze().long())
            cls_loss = loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_cls_loss += cls_loss.item() * batch.num_graphs
    
    n = len(loader.dataset)
    return total_loss / n, total_cls_loss / n, total_reg_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    y_true, y_pred = [], []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1).cpu().numpy()
        y_true.extend(batch.y.cpu().numpy().flatten())
        y_pred.extend(pred)
    
    return f1_score(y_true, y_pred, average='macro'), y_true, y_pred


@torch.no_grad()
def predict(model, loader, device):
    """Generate predictions."""
    model.eval()
    predictions = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1).cpu().numpy()
        predictions.extend(pred)
    
    return predictions


@torch.no_grad()
def measure_inference_time(model, loader, device, num_runs=3):
    """Measure average inference time per batch."""
    model.eval()
    
    # Warmup
    for batch in loader:
        batch = batch.to(device)
        _ = model(batch)
        break
    
    times = []
    for _ in range(num_runs):
        batch_times = []
        for batch in loader:
            batch = batch.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(batch)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            batch_times.append((end - start) * 1000)  # ms
        
        times.append(np.mean(batch_times))
    
    return np.mean(times)


def train_model(model_name, model, train_loader, valid_loader, test_loader, 
                device, num_epochs=100, lr=0.001, weight_decay=1e-4,
                class_weights=None, is_spectral=False):
    """Train a model and return best validation F1."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {count_parameters(model):,}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    best_state = None
    patience = 15
    no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, cls_loss, reg_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, is_spectral
        )
        
        # Validate
        val_f1, _, _ = evaluate(model, valid_loader, device)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            reg_str = f" | Reg: {reg_loss:.4f}" if is_spectral else ""
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f}{reg_str} | Val F1: {val_f1:.4f} | Best: {best_val_f1:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Final evaluation
    val_f1, _, _ = evaluate(model, valid_loader, device)
    print(f"\nBest Validation F1: {best_val_f1:.4f}")
    
    return model, best_val_f1


def main():
    parser = argparse.ArgumentParser(description='Train advanced baseline models')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['dmpnn', 'spectral', 'all'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--mmp-ood', action='store_true',
                        help='Use MMP-OOD split instead of default OGB scaffold split')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading OGB MolBACE dataset...")
    dataset = PygGraphPropPredDataset(name='ogbg-molbace', root='../data/ogb')
    
    # Get train/valid/test splits
    if args.mmp_ood:
        mmp_split_dir = os.path.join('..', 'data', 'mmp_split')
        if not os.path.exists(mmp_split_dir):
            print("Error: MMP-OOD split not found. Run 'python scripts/generate_mmp_split.py' first.")
            sys.exit(1)
        print("Using MMP-OOD split (activity-cliff-based)")
        train_idx = torch.tensor(pd.read_csv(os.path.join(mmp_split_dir, 'train.csv'))['index'].values)
        valid_idx = torch.tensor(pd.read_csv(os.path.join(mmp_split_dir, 'valid.csv'))['index'].values)
        test_idx = torch.tensor(pd.read_csv(os.path.join(mmp_split_dir, 'test.csv'))['index'].values)
    else:
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    split_name = 'MMP-OOD' if args.mmp_ood else 'scaffold'
    print(f"Dataset: {len(dataset)} molecules")
    print(f"Split ({split_name}): Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
    print(f"Node features: {dataset.num_node_features}")
    
    # Create loaders
    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[valid_idx], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False)
    
    # Class weights
    train_labels = [dataset[i].y.item() for i in train_idx]
    pos_ratio = sum(train_labels) / len(train_labels)
    class_weights = torch.tensor([pos_ratio, 1 - pos_ratio], dtype=torch.float)
    print(f"Class distribution: {pos_ratio:.2%} positive")
    
    # Results storage
    results = {}
    
    # Train D-MPNN
    if args.model in ['dmpnn', 'all']:
        dmpnn = DMPNNModel(
            in_channels=dataset.num_node_features,
            edge_channels=3,
            hidden_channels=args.hidden,
            out_channels=2,
            num_layers=3,
            dropout=0.2,
            pooling='mean'
        )
        
        dmpnn, best_f1 = train_model(
            "D-MPNN",
            dmpnn,
            train_loader, valid_loader, test_loader,
            device,
            num_epochs=args.epochs,
            lr=args.lr,
            class_weights=class_weights,
            is_spectral=False
        )
        
        # Generate predictions and measure performance
        predictions = predict(dmpnn, test_loader, device)
        inference_time = measure_inference_time(dmpnn, test_loader, device)
        num_params = count_parameters(dmpnn)
        
        # Save submission - use test indices as IDs to match label file
        test_indices = test_idx.tolist()
        submission_df = pd.DataFrame({
            'id': test_indices,
            'target': predictions
        })
        submission_path = '../submissions/dmpnn_submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"\n✅ D-MPNN submission saved to: {submission_path}")
        
        # Save metadata
        metadata = {
            'model_name': 'D-MPNN',
            'architecture': 'Directional Message Passing Neural Network',
            'parameters': num_params,
            'inference_time_ms': round(inference_time, 2),
            'validation_f1': round(best_f1, 4),
            'hidden_dim': args.hidden,
            'num_layers': 3,
            'framework': 'PyTorch Geometric'
        }
        with open('../submissions/dmpnn_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   Metadata saved to: ../submissions/dmpnn_metadata.json")
        
        results['dmpnn'] = {
            'f1': best_f1,
            'params': num_params,
            'time_ms': inference_time
        }
    
    # Train Spectral GNN
    if args.model in ['spectral', 'all']:
        spectral = SpectralGNN(
            in_channels=dataset.num_node_features,
            hidden_channels=args.hidden,
            out_channels=2,
            num_layers=3,
            K=3,  # Chebyshev order
            dropout=0.2,
            laplacian_weight=0.01,
            use_positional_encoding=False
        )
        
        spectral, best_f1 = train_model(
            "Spectral GNN",
            spectral,
            train_loader, valid_loader, test_loader,
            device,
            num_epochs=args.epochs,
            lr=args.lr,
            class_weights=class_weights,
            is_spectral=True
        )
        
        # Generate predictions and measure performance
        predictions = predict(spectral, test_loader, device)
        inference_time = measure_inference_time(spectral, test_loader, device)
        num_params = count_parameters(spectral)
        
        # Save submission - use test indices as IDs to match label file
        test_indices = test_idx.tolist()
        submission_df = pd.DataFrame({
            'id': test_indices,
            'target': predictions
        })
        submission_path = '../submissions/spectral_submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"\n✅ Spectral GNN submission saved to: {submission_path}")
        
        # Save metadata
        metadata = {
            'model_name': 'Spectral-GNN',
            'architecture': 'Spectral GNN with Chebyshev Convolutions',
            'parameters': num_params,
            'inference_time_ms': round(inference_time, 2),
            'validation_f1': round(best_f1, 4),
            'hidden_dim': args.hidden,
            'num_layers': 3,
            'chebyshev_order': 3,
            'laplacian_weight': 0.01,
            'framework': 'PyTorch Geometric'
        }
        with open('../submissions/spectral_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   Metadata saved to: ../submissions/spectral_metadata.json")
        
        results['spectral'] = {
            'f1': best_f1,
            'params': num_params,
            'time_ms': inference_time
        }
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for name, res in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Validation F1: {res['f1']:.4f}")
        print(f"  Parameters:    {res['params']:,}")
        print(f"  Inference:     {res['time_ms']:.2f} ms/batch")
    
    print("\n✅ All submissions generated in ../submissions/")
    print("   Run scoring_script.py to evaluate and update leaderboard")


if __name__ == "__main__":
    main()
