# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
from graphslim.sparsification.networkit_sparsifiers import NetworkitSparsifier


class GSpar(NetworkitSparsifier):
    """G-Spar using NetworKit Jaccard similarity scores."""

    networkit_class = "JaccardSimilaritySparsifier"
