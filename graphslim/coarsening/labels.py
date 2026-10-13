"""Class-balanced synthetic labels shared by structure-free coarsening."""

from collections import Counter
import numpy as np
import torch


def synthetic_labels(labels_train, reduction_rate):
    """Keep the historical rarest-class-first order and one node per class."""
    counts = Counter(labels_train.tolist())
    labels = [
        label
        for label, count in sorted(counts.items(), key=lambda item: item[1])
        for _ in range(max(int(count * reduction_rate), 1))
    ]
    return np.asarray(labels, dtype=np.int64)


def label_tensor(labels, reference):
    """NumPy boolean masks must not become byte masks during Tensor indexing."""
    return torch.as_tensor(labels, dtype=torch.long, device=reference.device)
