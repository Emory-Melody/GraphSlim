# SPDX-License-Identifier: MIT
# See LICENSE and graphslim/THIRD_PARTY_NOTICES.md for retained upstream terms.
from graphslim.sparsification.networkit_sparsifiers import NetworkitSparsifier


class Scan(NetworkitSparsifier):
    """Edge sparsification by SCAN structural similarity."""

    networkit_class = "SCANSparsifier"
