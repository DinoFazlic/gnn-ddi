#!/usr/bin/env python3
"""
Export adjacency matrices (A) and node feature matrices (X) for all molecules
in the OGB MolBACE dataset.

This script satisfies the competition requirement that A and X are
explicitly provided.  Each split (train / valid / test) is saved as a
compressed NumPy archive (.npz) in ``data/graphs/``.

Usage:
    python scripts/export_graph_matrices.py

Outputs:
    data/graphs/train_graphs.npz
    data/graphs/valid_graphs.npz
    data/graphs/test_graphs.npz
    data/graphs/README_graphs.md

Each .npz file contains, for every molecule index i:
    adj_{i}  — adjacency matrix  A ∈ {0,1}^{n×n}  (dense, symmetric)
    x_{i}    — node feature matrix  X ∈ ℝ^{n×9}
    y_{i}    — label (train/valid only; test labels are hidden)

where n = number of atoms in molecule i.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.serialization
except ImportError:
    print("❌ torch is required.  Install with:  pip install torch")
    sys.exit(1)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_GRAPHS_DIR = _REPO_ROOT / "data" / "graphs"
_PROCESSED_PT = _REPO_ROOT / "data" / "ogb" / "ogbg_molbace" / "processed" / "geometric_data_processed.pt"


def _load_dataset():
    """Load the pre-processed PyG dataset directly from .pt file.

    This avoids OGB/rdkit/numpy version conflicts by loading the
    already-processed tensors.
    """
    if not _PROCESSED_PT.exists():
        print(f"❌ Processed data not found: {_PROCESSED_PT}")
        print("   Run the baseline first to download the dataset, or use:")
        print("   python -c \"from ogb.graphproppred import PygGraphPropPredDataset; "
              "PygGraphPropPredDataset(name='ogbg-molbace')\"")
        sys.exit(1)

    print(f"Loading pre-processed data from {_PROCESSED_PT} ...")
    data_obj, slices = torch.load(str(_PROCESSED_PT), weights_only=False)
    return data_obj, slices


def _get_graph(data_obj, slices, idx: int):
    """Extract a single graph from the batched PyG data object."""
    # Node features X
    x_start = int(slices["x"][idx])
    x_end = int(slices["x"][idx + 1])
    x = data_obj.x[x_start:x_end].numpy()

    num_nodes = x.shape[0]

    # Edge index
    ei_start = int(slices["edge_index"][idx])
    ei_end = int(slices["edge_index"][idx + 1])
    edge_index = data_obj.edge_index[:, ei_start:ei_end].numpy()

    # Build dense adjacency
    adj = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1

    # Label
    y_start = int(slices["y"][idx])
    y_end = int(slices["y"][idx + 1])
    y = data_obj.y[y_start:y_end].numpy().flatten()

    return adj, x, y


def _load_split_indices():
    """Load train/valid/test split indices from data/*.csv files."""
    splits = {}
    for name in ("train", "valid", "test"):
        path = _REPO_ROOT / "data" / (name + ".csv")
        if not path.exists():
            path = _REPO_ROOT / "data" / "public" / (name + ".csv")
        indices = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("id"):
                    indices.append(int(line))
        splits[name] = indices
    return splits


def export_split(data_obj, slices, indices, split_name: str, include_labels: bool = True) -> Path:
    """Export molecules from one split to a .npz archive."""
    arrays: dict[str, np.ndarray] = {}

    for idx in indices:
        idx = int(idx)
        adj, x, y = _get_graph(data_obj, slices, idx)

        # Adjacency matrix A ∈ {0,1}^{n×n}
        arrays[f"adj_{idx}"] = adj

        # Node feature matrix X ∈ ℝ^{n×9}
        arrays[f"x_{idx}"] = x

        # Label y (skip for test to keep labels hidden)
        if include_labels:
            arrays[f"y_{idx}"] = y

    # Store the list of molecule indices
    arrays["indices"] = np.array(sorted(int(i) for i in indices), dtype=np.int64)

    out_path = _GRAPHS_DIR / f"{split_name}_graphs.npz"
    np.savez_compressed(out_path, **arrays)
    print(f"✅ {split_name}: {len(indices)} molecules → {out_path}  "
          f"({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def write_readme() -> None:
    """Write a README explaining the graph matrix format."""
    readme = _GRAPHS_DIR / "README_graphs.md"
    readme.write_text("""\
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
""")
    print(f"✅ README written: {readme}")


def main() -> None:
    _GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    data_obj, slices = _load_dataset()
    splits = _load_split_indices()

    export_split(data_obj, slices, splits["train"], "train", include_labels=True)
    export_split(data_obj, slices, splits["valid"], "valid", include_labels=True)
    export_split(data_obj, slices, splits["test"],  "test",  include_labels=False)

    write_readme()
    print()
    print("Done! Graph matrices exported to data/graphs/")
    print("Commit these files to the repository.")


if __name__ == "__main__":
    main()
