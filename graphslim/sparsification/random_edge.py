# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
from graphslim.sparsification.networkit_sparsifiers import NetworkitSparsifier


class RandomEdge(NetworkitSparsifier):
    """Uniform random edge sparsification."""

    networkit_class = "RandomEdgeSparsifier"
