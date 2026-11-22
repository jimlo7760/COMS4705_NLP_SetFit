"""SetFit pair sampler (proposal protocol).

For every anchor in the selected training subset:
  * 4 distinct same-label partners, never the anchor itself      -> z = 1
  * 4 distinct different-label partners: draw another class uniformly, then a
    sentence uniformly within it; resample if the partner was already drawn -> z = 0
The 8 * n pairs are shuffled once here; the trainer reshuffles every epoch.
Reversed pairs (A->B and B->A) may both occur. Only the selected subset is used.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def build_setfit_pairs(labels, rng: np.random.Generator, n_pos: int = C.N_POS, n_neg: int = C.N_NEG):
    """Return (anchor_idx, partner_idx, z) as numpy arrays of length n * (n_pos + n_neg)."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("need at least two classes")
    members = {c: np.flatnonzero(labels == c) for c in classes}
    others = {c: classes[classes != c] for c in classes}

    anchors, partners, z = [], [], []
    for i, c in enumerate(labels):
        same = members[c][members[c] != i]
        if len(same) < n_pos:
            raise ValueError(f"class {c} has {len(same)} other examples < n_pos={n_pos}")
        pos = rng.choice(same, size=n_pos, replace=False)

        neg: list = []
        while len(neg) < n_neg:
            c2 = others[c][rng.integers(len(others[c]))]
            j = int(members[c2][rng.integers(len(members[c2]))])
            if j not in neg:
                neg.append(j)

        anchors.extend([i] * (n_pos + n_neg))
        partners.extend(pos.tolist() + neg)
        z.extend([1.0] * n_pos + [0.0] * n_neg)

    order = rng.permutation(len(z))
    return (np.asarray(anchors)[order], np.asarray(partners)[order],
            np.asarray(z, dtype=np.float32)[order])
