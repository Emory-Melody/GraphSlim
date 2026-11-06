# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
"""Class-balanced synthetic labels shared by structure-free coarsening."""

from collections import Counter
import numpy as np
import torch


def synthetic_labels(labels_train, reduction_rate):
    """Allocate synthetic class labels in rarest-class-first order with at least one node per class.

    :param labels_train: Training node labels used for classwise budget allocation.
    :param reduction_rate: Fraction of labeled training nodes used to set the synthetic node budget.
    :returns: NumPy int64 vector of synthetic labels.
    """
    counts = Counter(labels_train.tolist())
    labels = [
        label
        for label, count in sorted(counts.items(), key=lambda item: item[1])
        for _ in range(max(int(count * reduction_rate), 1))
    ]
    return np.asarray(labels, dtype=np.int64)


def label_tensor(labels, reference):
    """Convert labels to an integer tensor on the reference tensor device.

    :param labels: Node labels in the same node ordering as the corresponding features.
    :param reference: Tensor supplying the destination device.
    :returns: Long tensor suitable for class-based indexing.
    """
    return torch.as_tensor(labels, dtype=torch.long, device=reference.device)
