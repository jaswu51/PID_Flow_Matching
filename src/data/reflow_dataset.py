"""
Reflow Dataset

Loads pre-generated (noise, image) pairs for Stage 2 Reflow training.
The pairs are stored as two .pt tensor files: x0_noise.pt and x1_generated.pt.
"""

import os
import torch
from torch.utils.data import Dataset


class ReflowDataset(Dataset):
    """
    Dataset for Reflow (Stage 2) training.

    Loads paired (x_0, x_1_hat) tensors from disk.

    Args:
        pairs_dir: Directory containing x0_noise.pt and x1_generated.pt
    """

    def __init__(self, pairs_dir: str):
        x0_path = os.path.join(pairs_dir, "x0_noise.pt")
        x1_path = os.path.join(pairs_dir, "x1_generated.pt")

        if not os.path.exists(x0_path) or not os.path.exists(x1_path):
            raise FileNotFoundError(
                f"Reflow pairs not found in {pairs_dir}. "
                f"Run generate_reflow_pairs.py first."
            )

        print(f"Loading reflow pairs from {pairs_dir}...")
        self.x0 = torch.load(x0_path, weights_only=True)
        self.x1 = torch.load(x1_path, weights_only=True)

        assert self.x0.shape == self.x1.shape, (
            f"Shape mismatch: x0={self.x0.shape}, x1={self.x1.shape}"
        )
        print(f"Loaded {len(self)} reflow pairs, shape: {self.x0.shape[1:]}")

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]
