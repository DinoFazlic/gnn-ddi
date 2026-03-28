# Graph Matrices — Adjacency (A) and Node Features (X)

This directory contains the **explicit graph representation** for every
molecule in the OGB MolBACE dataset, exported as compressed NumPy archives.

## Files

| File | Contents |
|------|----------|
| `train_graphs.npz` | 1,210 training molecules |
| `valid_graphs.npz` | 151 validation molecules |
| `test_graphs.npz`  | 152 test molecules (labels excluded) |

## Format

Each `.npz` file contains the following arrays for molecule index `i`:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `adj_{i}` | `(n, n)` | `uint8` | Adjacency matrix A ∈ {0,1}^{n×n} (symmetric, undirected) |
| `x_{i}` | `(n, 9)` | `int64` | Node (atom) feature matrix X ∈ ℝ^{n×9} |
| `y_{i}` | `(1,)` | `int64` | Binary label (train/valid only; omitted for test) |
| `indices` | `(N,)` | `int64` | Sorted array of all molecule indices in this split |

where `n` = number of atoms in molecule `i`, and `N` = number of molecules
in the split.

## Node Features (9 dimensions)

| Dim | Feature | Description |
|-----|---------|-------------|
| 0 | Atomic number | Type of atom (e.g., 6 = carbon) |
| 1 | Chirality tag | Stereo configuration |
| 2 | Degree | Number of bonds |
| 3 | Formal charge | Net ionic charge |
| 4 | Num Hs | Number of hydrogen atoms |
| 5 | Num radical electrons | Unpaired electrons |
| 6 | Hybridization | sp, sp2, sp3, etc. |
| 7 | Is aromatic | Part of aromatic ring (0/1) |
| 8 | Is in ring | Part of any ring (0/1) |

## Loading Example

```python
import numpy as np

# Load training graphs
data = np.load('data/graphs/train_graphs.npz', allow_pickle=False)

# List all molecule indices
indices = data['indices']
print(f"Training molecules: {len(indices)}")

# Load adjacency matrix and features for molecule 2
A = data['adj_2']   # shape (n, n), binary adjacency
X = data['x_2']     # shape (n, 9), node features
y = data['y_2']     # label: 0 or 1

print(f"Molecule 2: {A.shape[0]} atoms, label = {y[0]}")
print(f"Adjacency matrix shape: {A.shape}")
print(f"Feature matrix shape:   {X.shape}")
```

## Relation to OGB API

These matrices are equivalent to what you get from the OGB API:

```python
from ogb.graphproppred import PygGraphPropPredDataset

dataset = PygGraphPropPredDataset(name='ogbg-molbace')
graph = dataset[2]

# graph.edge_index → sparse representation of A
# graph.x          → same as X
# graph.y          → same as y
```

The `.npz` files provide these as explicit dense matrices for convenience
and compliance with the mandatory graph specification requirement.
