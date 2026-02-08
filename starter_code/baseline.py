"""
Baseline Models for GNN Molecular Graph Classification Challenge
=================================================================

This script demonstrates how to train GNN models on the OGB MolBACE dataset 
for predicting BACE-1 enzyme inhibition.

Included models:
- GraphSAGE: Inductive representation learning using sampling and aggregation
- GCN: Graph Convolutional Network with spectral convolutions
- GIN: Graph Isomorphism Network with powerful expressiveness

Usage:
    python baseline.py [--model {graphsage,gcn,gin}]

Output:
    - Prints validation F1 score during training
    - Generates sample_submission.csv in the submissions/ folder
"""

import os
import sys
import random
import argparse
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
from torch_geometric.nn import SAGEConv, GCNConv, GINConv, global_mean_pool, global_add_pool
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import f1_score

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE-based model for graph classification.
    
    Reference:
        Hamilton, W., Ying, Z., & Leskovec, J. (2017).
        "Inductive Representation Learning on Large Graphs"
        https://arxiv.org/abs/1706.02216
    
    Architecture:
    - 3 GraphSAGE convolution layers with mean aggregation
    - Global mean pooling to get graph-level representation
    - Linear classifier for binary classification
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        
        # GraphSAGE convolution layers
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Classifier
        self.lin = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Convert node features to float
        x = x.float()
        
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GraphSAGE layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Classifier
        x = self.lin(x)
        
        return x


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) for graph classification.
    
    Reference:
        Kipf, T. N., & Welling, M. (2017).
        "Semi-Supervised Classification with Graph Convolutional Networks"
        https://arxiv.org/abs/1609.02907
    
    Architecture:
    - 3 GCN convolution layers
    - Global mean pooling to get graph-level representation
    - Linear classifier for binary classification
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCNModel, self).__init__()
        
        # GCN convolution layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Classifier
        self.lin = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Convert node features to float
        x = x.float()
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Classifier
        x = self.lin(x)
        
        return x


class GINModel(nn.Module):
    """
    Graph Isomorphism Network (GIN) for graph classification.
    
    Reference:
        Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019).
        "How Powerful are Graph Neural Networks?"
        https://arxiv.org/abs/1810.00826
    
    GIN is proven to be as powerful as the Weisfeiler-Lehman graph isomorphism test,
    making it one of the most expressive GNN architectures.
    
    Architecture:
    - 3 GIN convolution layers with MLP update functions
    - Global sum pooling (recommended for GIN)
    - Linear classifier for binary classification
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GINModel, self).__init__()
        
        # GIN uses MLP for the update function
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            train_eps=True
        )
        
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            train_eps=True
        )
        
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            train_eps=True
        )
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Classifier
        self.lin = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Convert node features to float
        x = x.float()
        
        # First GIN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GIN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GIN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global sum pooling (recommended for GIN)
        x = global_add_pool(x, batch)
        
        # Classifier
        x = self.lin(x)
        
        return x


def train(model, loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        # OGB labels have shape [batch_size, 1], squeeze to [batch_size]
        loss = criterion(out, data.y.squeeze().long())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model and return predictions and true labels."""
    model.eval()
    y_true = []
    y_pred = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1).cpu().numpy()
        
        y_true.extend(data.y.cpu().numpy().flatten().tolist())
        y_pred.extend(pred.tolist())
    
    return y_true, y_pred


@torch.no_grad()
def predict(model, loader, device):
    """Generate predictions for the test set."""
    model.eval()
    predictions = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1).cpu().numpy()
        predictions.extend(pred.tolist())
    
    return predictions


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GNN baseline models for molecular graph classification')
    parser.add_argument('--model', type=str, default='graphsage', 
                        choices=['graphsage', 'gcn', 'gin'],
                        help='Model architecture to use (default: graphsage)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--mmp-ood', action='store_true',
                        help='Use MMP-OOD split instead of default OGB scaffold split')
    args = parser.parse_args()
    
    # Configuration
    BATCH_SIZE = args.batch_size
    HIDDEN_CHANNELS = args.hidden
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = 1e-4
    DROPOUT = args.dropout
    MODEL_NAME = args.model
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME.upper()}")
    
    # Load OGB MolBACE dataset
    print("\nLoading OGB MolBACE dataset...")
    dataset = PygGraphPropPredDataset(name='ogbg-molbace', root='../data/ogb')
    
    # Get train/valid/test splits
    if args.mmp_ood:
        # Use MMP-OOD split (activity-cliff-based)
        mmp_split_dir = os.path.join('..', 'data', 'mmp_split')
        if not os.path.exists(mmp_split_dir):
            print("Error: MMP-OOD split not found. Run 'python scripts/generate_mmp_split.py' first.")
            sys.exit(1)
        print("Using MMP-OOD split (activity-cliff-based)")
        _mmp_train = pd.read_csv(os.path.join(mmp_split_dir, 'train.csv'))['index'].values
        _mmp_valid = pd.read_csv(os.path.join(mmp_split_dir, 'valid.csv'))['index'].values
        _mmp_test = pd.read_csv(os.path.join(mmp_split_dir, 'test.csv'))['index'].values
        train_idx = torch.tensor(_mmp_train)
        valid_idx = torch.tensor(_mmp_valid)
        test_idx = torch.tensor(_mmp_test)
    else:
        # Default OGB scaffold split
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
    
    print(f"Dataset size: {len(dataset)}")
    split_name = 'MMP-OOD' if args.mmp_ood else 'scaffold'
    print(f"Split ({split_name}): Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
    print(f"Number of node features: {dataset.num_node_features}")
    
    # Create data loaders
    train_loader = DataLoader(dataset[train_idx], batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset[valid_idx], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset[test_idx], batch_size=BATCH_SIZE, shuffle=False)
    
    # Check class distribution
    train_labels = [dataset[i].y.item() for i in train_idx]
    pos_ratio = sum(train_labels) / len(train_labels)
    print(f"\nClass distribution in training set: {pos_ratio:.2%} positive")
    
    # Calculate class weights for imbalanced data
    neg_weight = pos_ratio
    pos_weight = 1 - pos_ratio
    class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float).to(device)
    
    # Initialize model based on selection
    model_classes = {
        'graphsage': GraphSAGEModel,
        'gcn': GCNModel,
        'gin': GINModel
    }
    
    ModelClass = model_classes[MODEL_NAME]
    model = ModelClass(
        in_channels=dataset.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=2,  # Binary classification
        dropout=DROPOUT
    ).to(device)
    
    print(f"\nModel architecture ({MODEL_NAME.upper()}):")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop
    print("\n" + "="*60)
    print(f"Starting training with {MODEL_NAME.upper()}...")
    print("="*60)
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        # Evaluate on validation set
        y_true, y_pred = evaluate(model, valid_loader, device)
        val_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Best F1: {best_val_f1:.4f}")
    
    print("\n" + "="*60)
    print(f"Training complete! Best validation F1 ({MODEL_NAME.upper()}): {best_val_f1:.4f}")
    print("="*60)
    
    # Load best model for predictions
    model.load_state_dict(best_model_state)
    
    # Generate predictions on test set
    print("\nGenerating predictions on test set...")
    test_predictions = predict(model, test_loader, device)
    
    # Create submission file
    # Map predictions to test IDs
    test_ids = test_idx.tolist()
    submission_df = pd.DataFrame({
        'id': test_ids,
        'target': test_predictions
    })
    
    # Save submission
    submission_path = f'../submissions/{MODEL_NAME}_submission.csv'
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    print(f"Number of predictions: {len(submission_df)}")
    print(f"Prediction distribution: {submission_df['target'].value_counts().to_dict()}")
    
    # Also save validation predictions for local testing
    val_true, val_pred = evaluate(model, valid_loader, device)
    val_f1_final = f1_score(val_true, val_pred, average='macro')
    print(f"\nFinal validation F1 score ({MODEL_NAME.upper()}): {val_f1_final:.4f}")
    
    return best_val_f1


def run_all_baselines():
    """Run all baseline models and compare their performance."""
    print("="*60)
    print("Running All Baseline Models")
    print("="*60)
    
    results = {}
    
    for model_name in ['graphsage', 'gcn', 'gin']:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}...")
        print("="*60)
        
        # Reset seeds for fair comparison
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        # Override sys.argv to set model
        original_argv = sys.argv.copy()
        sys.argv = ['baseline.py', '--model', model_name]
        
        try:
            f1 = main()
            results[model_name] = f1
        finally:
            sys.argv = original_argv
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'Validation F1':>15}")
    print("-"*30)
    for model_name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name.upper():<15} {f1:>15.4f}")
    
    return results


if __name__ == "__main__":
    # Check if running all baselines
    if '--all' in sys.argv:
        sys.argv.remove('--all')
        run_all_baselines()
    else:
        main()
