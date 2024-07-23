# import matplotlib.pylab as plt
import numbers

# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
from pygsp import filters, reduction, graphs
from pygsp.utils import resistance_distance
from sortedcontainers import SortedList

# from graphslim.coarsening.maxWeightMatching import *


"""Weighted maximum matching in general graphs.

The algorithm is taken from "Efficient Algorithms for Finding Maximum
Matching in Graphs" by Zvi Galil, ACM Computing Surveys, 1986.
It is based on the "blossom" method for finding augmenting paths and
the "primal-dual" method for finding a matching of maximum weight, both
due to Jack Edmonds.
Some ideas came from "Implementation of algorithms for maximum matching
on non-bipartite graphs" by H.J. Gabow, Standford Ph.D. thesis, 1973.

A C program for maximum weight matching by Ed Rothberg was used extensively
to validate this new code.
"""

DEBUG = None
CHECK_DELTA = False
CHECK_OPTIMUM = False


def maxWeightMatching(edges, maxcardinality=False):
    """Compute a maximum-weighted matching in the general undirected
    weighted graph given by "edges".  If "maxcardinality" is true,
    only maximum-cardinality matchings are considered as solutions.

    Edges is a sequence of tuples (i, j, wt) describing an undirected
    edge between vertex i and vertex j with weight wt.  There is at most
    one edge between any two vertices; no vertex has an edge to itself.
    Vertices are identified by consecutive, non-negative integers.

    Return a list "mate", such that mate[i] == j if vertex i is
    matched to vertex j, and mate[i] == -1 if vertex i is not matched.

    This function takes time O(n ** 3)."""

    #
    # Vertices are numbered 0 .. (nvertex-1).
    # Non-trivial blossoms are numbered nvertex .. (2*nvertex-1)
    #
    # Edges are numbered 0 .. (nedge-1).
    # Edge endpoints are numbered 0 .. (2*nedge-1), such that endpoints
    # (2*k) and (2*k+1) both belong to edge k.
    #
    # Many terms used in the comments (sub-blossom, T-vertex) come from
    # the paper by Galil; read the paper before reading this code.
    #

    # # Python 2/3 compatibility.
    # if sys_version < '3':
    #     integer_types = (int, long)
    # else:
    #     integer_types = (int,)

    # Deal swiftly with empty graphs.
    if not edges:
        return []

    # Count vertices.
    nedge = len(edges)
    nvertex = 0
    for (i, j, w) in edges:
        assert i >= 0 and j >= 0 and i != j
        if i >= nvertex:
            nvertex = i + 1
        if j >= nvertex:
            nvertex = j + 1

    # Find the maximum edge weight.
    maxweight = max(0, max([wt for (i, j, wt) in edges]))

    # If p is an edge endpoint,
    # endpoint[p] is the vertex to which endpoint p is attached.
    # Not modified by the algorithm.
    endpoint = [edges[p // 2][p % 2] for p in range(2 * nedge)]

    # If v is a vertex,
    # neighbend[v] is the list of remote endpoints of the edges attached to v.
    # Not modified by the algorithm.
    neighbend = [[] for i in range(nvertex)]
    for k in range(len(edges)):
        (i, j, w) = edges[k]
        neighbend[i].append(2 * k + 1)
        neighbend[j].append(2 * k)

    # If v is a vertex,
    # mate[v] is the remote endpoint of its matched edge, or -1 if it is single
    # (i.e. endpoint[mate[v]] is v's partner vertex).
    # Initially all vertices are single; updated during augmentation.
    mate = nvertex * [-1]

    # If b is a top-level blossom,
    # label[b] is 0 if b is unlabeled (free);
    #             1 if b is an S-vertex/blossom;
    #             2 if b is a T-vertex/blossom.
    # The label of a vertex is found by looking at the label of its
    # top-level containing blossom.
    # If v is a vertex inside a T-blossom,
    # label[v] is 2 iff v is reachable from an S-vertex outside the blossom.
    # Labels are assigned during a stage and reset after each augmentation.
    label = (2 * nvertex) * [0]

    # If b is a labeled top-level blossom,
    # labelend[b] is the remote endpoint of the edge through which b obtained
    # its label, or -1 if b's base vertex is single.
    # If v is a vertex inside a T-blossom and label[v] == 2,
    # labelend[v] is the remote endpoint of the edge through which v is
    # reachable from outside the blossom.
    labelend = (2 * nvertex) * [-1]

    # If v is a vertex,
    # inblossom[v] is the top-level blossom to which v belongs.
    # If v is a top-level vertex, v is itself a blossom (a trivial blossom)
    # and inblossom[v] == v.
    # Initially all vertices are top-level trivial blossoms.
    inblossom = list(range(nvertex))

    # If b is a sub-blossom,
    # blossomparent[b] is its immediate parent (sub-)blossom.
    # If b is a top-level blossom, blossomparent[b] is -1.
    blossomparent = (2 * nvertex) * [-1]

    # If b is a non-trivial (sub-)blossom,
    # blossomchilds[b] is an ordered list of its sub-blossoms, starting with
    # the base and going round the blossom.
    blossomchilds = (2 * nvertex) * [None]

    # If b is a (sub-)blossom,
    # blossombase[b] is its base VERTEX (i.e. recursive sub-blossom).
    blossombase = list(range(nvertex)) + nvertex * [-1]

    # If b is a non-trivial (sub-)blossom,
    # blossomendps[b] is a list of endpoints on its connecting edges,
    # such that blossomendps[b][i] is the local endpoint of blossomchilds[b][i]
    # on the edge that connects it to blossomchilds[b][wrap(i+1)].
    blossomendps = (2 * nvertex) * [None]

    # If v is a free vertex (or an unreached vertex inside a T-blossom),
    # bestedge[v] is the edge to an S-vertex with least slack,
    # or -1 if there is no such edge.
    # If b is a (possibly trivial) top-level S-blossom,
    # bestedge[b] is the least-slack edge to a different S-blossom,
    # or -1 if there is no such edge.
    # This is used for efficient computation of delta2 and delta3.
    bestedge = (2 * nvertex) * [-1]

    # If b is a non-trivial top-level S-blossom,
    # blossombestedges[b] is a list of least-slack edges to neighbouring
    # S-blossoms, or None if no such list has been computed yet.
    # This is used for efficient computation of delta3.
    blossombestedges = (2 * nvertex) * [None]

    # List of currently unused blossom numbers.
    unusedblossoms = list(range(nvertex, 2 * nvertex))

    # If v is a vertex,
    # dualvar[v] = 2 * u(v) where u(v) is the v's variable in the dual
    # optimization problem (multiplication by two ensures integer values
    # throughout the algorithm if all edge weights are integers).
    # If b is a non-trivial blossom,
    # dualvar[b] = z(b) where z(b) is b's variable in the dual optimization
    # problem.
    dualvar = nvertex * [maxweight] + nvertex * [0]

    # If allowedge[k] is true, edge k has zero slack in the optimization
    # problem; if allowedge[k] is false, the edge's slack may or may not
    # be zero.
    allowedge = nedge * [False]

    # Queue of newly discovered S-vertices.
    queue = []

    # Return 2 * slack of edge k (does not work inside blossoms).
    def slack(k):
        (i, j, wt) = edges[k]
        return dualvar[i] + dualvar[j] - 2 * wt

    # Generate the leaf vertices of a blossom.
    def blossomLeaves(b):
        if b < nvertex:
            yield b
        else:
            for t in blossomchilds[b]:
                if t < nvertex:
                    yield t
                else:
                    for v in blossomLeaves(t):
                        yield v

    # Assign label t to the top-level blossom containing vertex w
    # and record the fact that w was reached through the edge with
    # remote endpoint p.
    def assignLabel(w, t, p):
        if DEBUG: DEBUG('assignLabel(%d,%d,%d)' % (w, t, p))
        b = inblossom[w]
        assert label[w] == 0 and label[b] == 0
        label[w] = label[b] = t
        labelend[w] = labelend[b] = p
        bestedge[w] = bestedge[b] = -1
        if t == 1:
            # b became an S-vertex/blossom; add it(s vertices) to the queue.
            queue.extend(blossomLeaves(b))
            if DEBUG: DEBUG('PUSH ' + str(list(blossomLeaves(b))))
        elif t == 2:
            # b became a T-vertex/blossom; assign label S to its mate.
            # (If b is a non-trivial blossom, its base is the only vertex
            # with an external mate.)
            base = blossombase[b]
            assert mate[base] >= 0
            assignLabel(endpoint[mate[base]], 1, mate[base] ^ 1)

    # Trace back from vertices v and w to discover either a new blossom
    # or an augmenting path. Return the base vertex of the new blossom or -1.
    def scanBlossom(v, w):
        if DEBUG: DEBUG('scanBlossom(%d,%d)' % (v, w))
        # Trace back from v and w, placing breadcrumbs as we go.
        path = []
        base = -1
        while v != -1 or w != -1:
            # Look for a breadcrumb in v's blossom or put a new breadcrumb.
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            assert label[b] == 1
            path.append(b)
            label[b] = 5
            # Trace one step back.
            assert labelend[b] == mate[blossombase[b]]
            if labelend[b] == -1:
                # The base of blossom b is single; stop tracing this path.
                v = -1
            else:
                v = endpoint[labelend[b]]
                b = inblossom[v]
                assert label[b] == 2
                # b is a T-blossom; trace one more step back.
                assert labelend[b] >= 0
                v = endpoint[labelend[b]]
            # Swap v and w so that we alternate between both paths.
            if w != -1:
                v, w = w, v
        # Remove breadcrumbs.
        for b in path:
            label[b] = 1
        # Return base vertex, if we found one.
        return base

    # Construct a new blossom with given base, containing edge k which
    # connects a pair of S vertices. Label the new blossom as S; set its dual
    # variable to zero; relabel its T-vertices to S and add them to the queue.
    def addBlossom(base, k):
        (v, w, wt) = edges[k]
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        # Create blossom.
        b = unusedblossoms.pop()
        if DEBUG: DEBUG('addBlossom(%d,%d) (v=%d w=%d) -> %d' % (base, k, v, w, b))
        blossombase[b] = base
        blossomparent[b] = -1
        blossomparent[bb] = b
        # Make list of sub-blossoms and their interconnecting edge endpoints.
        blossomchilds[b] = path = []
        blossomendps[b] = endps = []
        # Trace back from v to base.
        while bv != bb:
            # Add bv to the new blossom.
            blossomparent[bv] = b
            path.append(bv)
            endps.append(labelend[bv])
            assert (label[bv] == 2 or
                    (label[bv] == 1 and labelend[bv] == mate[blossombase[bv]]))
            # Trace one step back.
            assert labelend[bv] >= 0
            v = endpoint[labelend[bv]]
            bv = inblossom[v]
        # Reverse lists, add endpoint that connects the pair of S vertices.
        path.append(bb)
        path.reverse()
        endps.reverse()
        endps.append(2 * k)
        # Trace back from w to base.
        while bw != bb:
            # Add bw to the new blossom.
            blossomparent[bw] = b
            path.append(bw)
            endps.append(labelend[bw] ^ 1)
            assert (label[bw] == 2 or
                    (label[bw] == 1 and labelend[bw] == mate[blossombase[bw]]))
            # Trace one step back.
            assert labelend[bw] >= 0
            w = endpoint[labelend[bw]]
            bw = inblossom[w]
        # Set label to S.
        assert label[bb] == 1
        label[b] = 1
        labelend[b] = labelend[bb]
        # Set dual variable to zero.
        dualvar[b] = 0
        # Relabel vertices.
        for v in blossomLeaves(b):
            if label[inblossom[v]] == 2:
                # This T-vertex now turns into an S-vertex because it becomes
                # part of an S-blossom; add it to the queue.
                queue.append(v)
            inblossom[v] = b
        # Compute blossombestedges[b].
        bestedgeto = (2 * nvertex) * [-1]
        for bv in path:
            if blossombestedges[bv] is None:
                # This subblossom does not have a list of least-slack edges;
                # get the information from the vertices.
                nblists = [[p // 2 for p in neighbend[v]]
                           for v in blossomLeaves(bv)]
            else:
                # Walk this subblossom's least-slack edges.
                nblists = [blossombestedges[bv]]
            for nblist in nblists:
                for k in nblist:
                    (i, j, wt) = edges[k]
                    if inblossom[j] == b:
                        i, j = j, i
                    bj = inblossom[j]
                    if (bj != b and label[bj] == 1 and
                            (bestedgeto[bj] == -1 or
                             slack(k) < slack(bestedgeto[bj]))):
                        bestedgeto[bj] = k
            # Forget about least-slack edges of the subblossom.
            blossombestedges[bv] = None
            bestedge[bv] = -1
        blossombestedges[b] = [k for k in bestedgeto if k != -1]
        # Select bestedge[b].
        bestedge[b] = -1
        for k in blossombestedges[b]:
            if bestedge[b] == -1 or slack(k) < slack(bestedge[b]):
                bestedge[b] = k
        if DEBUG: DEBUG('blossomchilds[%d]=' % b + repr(blossomchilds[b]))

    # Expand the given top-level blossom.
    def expandBlossom(b, endstage):
        if DEBUG: DEBUG('expandBlossom(%d,%d) %s' % (b, endstage, repr(blossomchilds[b])))
        # Convert sub-blossoms into top-level blossoms.
        for s in blossomchilds[b]:
            blossomparent[s] = -1
            if s < nvertex:
                inblossom[s] = s
            elif endstage and dualvar[s] == 0:
                # Recursively expand this sub-blossom.
                expandBlossom(s, endstage)
            else:
                for v in blossomLeaves(s):
                    inblossom[v] = s
        # If we expand a T-blossom during a stage, its sub-blossoms must be
        # relabeled.
        if (not endstage) and label[b] == 2:
            # Start at the sub-blossom through which the expanding
            # blossom obtained its label, and relabel sub-blossoms untili
            # we reach the base.
            # Figure out through which sub-blossom the expanding blossom
            # obtained its label initially.
            assert labelend[b] >= 0
            entrychild = inblossom[endpoint[labelend[b] ^ 1]]
            # Decide in which direction we will go round the blossom.
            j = blossomchilds[b].index(entrychild)
            if j & 1:
                # Start index is odd; go forward and wrap.
                j -= len(blossomchilds[b])
                jstep = 1
                endptrick = 0
            else:
                # Start index is even; go backward.
                jstep = -1
                endptrick = 1
            # Move along the blossom until we get to the base.
            p = labelend[b]
            while j != 0:
                # Relabel the T-sub-blossom.
                label[endpoint[p ^ 1]] = 0
                label[endpoint[blossomendps[b][j - endptrick] ^ endptrick ^ 1]] = 0
                assignLabel(endpoint[p ^ 1], 2, p)
                # Step to the next S-sub-blossom and note its forward endpoint.
                allowedge[blossomendps[b][j - endptrick] // 2] = True
                j += jstep
                p = blossomendps[b][j - endptrick] ^ endptrick
                # Step to the next T-sub-blossom.
                allowedge[p // 2] = True
                j += jstep
            # Relabel the base T-sub-blossom WITHOUT stepping through to
            # its mate (so don't call assignLabel).
            bv = blossomchilds[b][j]
            label[endpoint[p ^ 1]] = label[bv] = 2
            labelend[endpoint[p ^ 1]] = labelend[bv] = p
            bestedge[bv] = -1
            # Continue along the blossom until we get back to entrychild.
            j += jstep
            while blossomchilds[b][j] != entrychild:
                # Examine the vertices of the sub-blossom to see whether
                # it is reachable from a neighbouring S-vertex outside the
                # expanding blossom.
                bv = blossomchilds[b][j]
                if label[bv] == 1:
                    # This sub-blossom just got label S through one of its
                    # neighbours; leave it.
                    j += jstep
                    continue
                for v in blossomLeaves(bv):
                    if label[v] != 0:
                        break
                # If the sub-blossom contains a reachable vertex, assign
                # label T to the sub-blossom.
                if label[v] != 0:
                    assert label[v] == 2
                    assert inblossom[v] == bv
                    label[v] = 0
                    label[endpoint[mate[blossombase[bv]]]] = 0
                    assignLabel(v, 2, labelend[v])
                j += jstep
        # Recycle the blossom number.
        label[b] = labelend[b] = -1
        blossomchilds[b] = blossomendps[b] = None
        blossombase[b] = -1
        blossombestedges[b] = None
        bestedge[b] = -1
        unusedblossoms.append(b)

    # Swap matched/unmatched edges over an alternating path through blossom b
    # between vertex v and the base vertex. Keep blossom bookkeeping consistent.
    def augmentBlossom(b, v):
        if DEBUG: DEBUG('augmentBlossom(%d,%d)' % (b, v))
        # Bubble up through the blossom tree from vertex v to an immediate
        # sub-blossom of b.
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        # Recursively deal with the first sub-blossom.
        if t >= nvertex:
            augmentBlossom(t, v)
        # Decide in which direction we will go round the blossom.
        i = j = blossomchilds[b].index(t)
        if i & 1:
            # Start index is odd; go forward and wrap.
            j -= len(blossomchilds[b])
            jstep = 1
            endptrick = 0
        else:
            # Start index is even; go backward.
            jstep = -1
            endptrick = 1
        # Move along the blossom until we get to the base.
        while j != 0:
            # Step to the next sub-blossom and augment it recursively.
            j += jstep
            t = blossomchilds[b][j]
            p = blossomendps[b][j - endptrick] ^ endptrick
            if t >= nvertex:
                augmentBlossom(t, endpoint[p])
            # Step to the next sub-blossom and augment it recursively.
            j += jstep
            t = blossomchilds[b][j]
            if t >= nvertex:
                augmentBlossom(t, endpoint[p ^ 1])
            # Match the edge connecting those sub-blossoms.
            mate[endpoint[p]] = p ^ 1
            mate[endpoint[p ^ 1]] = p
            if DEBUG: DEBUG('PAIR %d %d (k=%d)' % (endpoint[p], endpoint[p ^ 1], p // 2))
        # Rotate the list of sub-blossoms to put the new base at the front.
        blossomchilds[b] = blossomchilds[b][i:] + blossomchilds[b][:i]
        blossomendps[b] = blossomendps[b][i:] + blossomendps[b][:i]
        blossombase[b] = blossombase[blossomchilds[b][0]]
        assert blossombase[b] == v

    # Swap matched/unmatched edges over an alternating path between two
    # single vertices. The augmenting path runs through edge k, which
    # connects a pair of S vertices.
    def augmentMatching(k):
        (v, w, wt) = edges[k]
        if DEBUG: DEBUG('augmentMatching(%d) (v=%d w=%d)' % (k, v, w))
        if DEBUG: DEBUG('PAIR %d %d (k=%d)' % (v, w, k))
        for (s, p) in ((v, 2 * k + 1), (w, 2 * k)):
            # Match vertex s to remote endpoint p. Then trace back from s
            # until we find a single vertex, swapping matched and unmatched
            # edges as we go.
            while 1:
                bs = inblossom[s]
                assert label[bs] == 1
                assert labelend[bs] == mate[blossombase[bs]]
                # Augment through the S-blossom from s to base.
                if bs >= nvertex:
                    augmentBlossom(bs, s)
                # Update mate[s]
                mate[s] = p
                # Trace one step back.
                if labelend[bs] == -1:
                    # Reached single vertex; stop.
                    break
                t = endpoint[labelend[bs]]
                bt = inblossom[t]
                assert label[bt] == 2
                # Trace one step back.
                assert labelend[bt] >= 0
                s = endpoint[labelend[bt]]
                j = endpoint[labelend[bt] ^ 1]
                # Augment through the T-blossom from j to base.
                assert blossombase[bt] == t
                if bt >= nvertex:
                    augmentBlossom(bt, j)
                # Update mate[j]
                mate[j] = labelend[bt]
                # Keep the opposite endpoint;
                # it will be assigned to mate[s] in the next step.
                p = labelend[bt] ^ 1
                if DEBUG: DEBUG('PAIR %d %d (k=%d)' % (s, t, p // 2))

    # Verify that the optimum solution has been reached.
    def verifyOptimum():
        if maxcardinality:
            # Vertices may have negative dual;
            # find a constant non-negative number to add to all vertex duals.
            vdualoffset = max(0, -min(dualvar[:nvertex]))
        else:
            vdualoffset = 0
        # 0. all dual variables are non-negative
        assert min(dualvar[:nvertex]) + vdualoffset >= 0
        assert min(dualvar[nvertex:]) >= 0
        # 0. all edges have non-negative slack and
        # 1. all matched edges have zero slack;
        for k in range(nedge):
            (i, j, wt) = edges[k]
            s = dualvar[i] + dualvar[j] - 2 * wt
            iblossoms = [i]
            jblossoms = [j]
            while blossomparent[iblossoms[-1]] != -1:
                iblossoms.append(blossomparent[iblossoms[-1]])
            while blossomparent[jblossoms[-1]] != -1:
                jblossoms.append(blossomparent[jblossoms[-1]])
            iblossoms.reverse()
            jblossoms.reverse()
            for (bi, bj) in zip(iblossoms, jblossoms):
                if bi != bj:
                    break
                s += 2 * dualvar[bi]
            assert s >= 0
            if mate[i] // 2 == k or mate[j] // 2 == k:
                assert mate[i] // 2 == k and mate[j] // 2 == k
                assert s == 0
        # 2. all single vertices have zero dual value;
        for v in range(nvertex):
            assert mate[v] >= 0 or dualvar[v] + vdualoffset == 0
        # 3. all blossoms with positive dual value are full.
        for b in range(nvertex, 2 * nvertex):
            if blossombase[b] >= 0 and dualvar[b] > 0:
                assert len(blossomendps[b]) % 2 == 1
                for p in blossomendps[b][1::2]:
                    assert mate[endpoint[p]] == p ^ 1
                    assert mate[endpoint[p ^ 1]] == p
        # Ok.

    # Check optimized delta2 against a trivial computation.
    def checkDelta2():
        for v in range(nvertex):
            if label[inblossom[v]] == 0:
                bd = None
                bk = -1
                for p in neighbend[v]:
                    k = p // 2
                    w = endpoint[p]
                    if label[inblossom[w]] == 1:
                        d = slack(k)
                        if bk == -1 or d < bd:
                            bk = k
                            bd = d
                if DEBUG and (bestedge[v] != -1 or bk != -1) and (bestedge[v] == -1 or bd != slack(bestedge[v])):
                    DEBUG('v=' + str(v) + ' bk=' + str(bk) + ' bd=' + str(bd) + ' bestedge=' + str(
                        bestedge[v]) + ' slack=' + str(slack(bestedge[v])))
                assert (bk == -1 and bestedge[v] == -1) or (bestedge[v] != -1 and bd == slack(bestedge[v]))

    # Check optimized delta3 against a trivial computation.
    def checkDelta3():
        bk = -1
        bd = None
        tbk = -1
        tbd = None
        for b in range(2 * nvertex):
            if blossomparent[b] == -1 and label[b] == 1:
                for v in blossomLeaves(b):
                    for p in neighbend[v]:
                        k = p // 2
                        w = endpoint[p]
                        if inblossom[w] != b and label[inblossom[w]] == 1:
                            d = slack(k)
                            if bk == -1 or d < bd:
                                bk = k
                                bd = d
                if bestedge[b] != -1:
                    (i, j, wt) = edges[bestedge[b]]
                    assert inblossom[i] == b or inblossom[j] == b
                    assert inblossom[i] != b or inblossom[j] != b
                    assert label[inblossom[i]] == 1 and label[inblossom[j]] == 1
                    if tbk == -1 or slack(bestedge[b]) < tbd:
                        tbk = bestedge[b]
                        tbd = slack(bestedge[b])
        if DEBUG and bd != tbd:
            DEBUG('bk=%d tbk=%d bd=%s tbd=%s' % (bk, tbk, repr(bd), repr(tbd)))
        assert bd == tbd

    # Main loop: continue until no further improvement is possible.
    for t in range(nvertex):

        # Each iteration of this loop is a "stage".
        # A stage finds an augmenting path and uses that to improve
        # the matching.
        if DEBUG: DEBUG('STAGE %d' % t)

        # Remove labels from top-level blossoms/vertices.
        label[:] = (2 * nvertex) * [0]

        # Forget all about least-slack edges.
        bestedge[:] = (2 * nvertex) * [-1]
        blossombestedges[nvertex:] = nvertex * [None]

        # Loss of labeling means that we can not be sure that currently
        # allowable edges remain allowable througout this stage.
        allowedge[:] = nedge * [False]

        # Make queue empty.
        queue[:] = []

        # Label single blossoms/vertices with S and put them in the queue.
        for v in range(nvertex):
            if mate[v] == -1 and label[inblossom[v]] == 0:
                assignLabel(v, 1, -1)

        # Loop until we succeed in augmenting the matching.
        augmented = 0
        while 1:

            # Each iteration of this loop is a "substage".
            # A substage tries to find an augmenting path;
            # if found, the path is used to improve the matching and
            # the stage ends. If there is no augmenting path, the
            # primal-dual method is used to pump some slack out of
            # the dual variables.
            if DEBUG: DEBUG('SUBSTAGE')

            # Continue labeling until all vertices which are reachable
            # through an alternating path have got a label.
            while queue and not augmented:

                # Take an S vertex from the queue.
                v = queue.pop()
                if DEBUG: DEBUG('POP v=%d' % v)
                assert label[inblossom[v]] == 1

                # Scan its neighbours:
                for p in neighbend[v]:
                    k = p // 2
                    w = endpoint[p]
                    # w is a neighbour to v
                    if inblossom[v] == inblossom[w]:
                        # this edge is internal to a blossom; ignore it
                        continue
                    if not allowedge[k]:
                        kslack = slack(k)
                        if kslack <= 0:
                            # edge k has zero slack => it is allowable
                            allowedge[k] = True
                    if allowedge[k]:
                        if label[inblossom[w]] == 0:
                            # (C1) w is a free vertex;
                            # label w with T and label its mate with S (R12).
                            assignLabel(w, 2, p ^ 1)
                        elif label[inblossom[w]] == 1:
                            # (C2) w is an S-vertex (not in the same blossom);
                            # follow back-links to discover either an
                            # augmenting path or a new blossom.
                            base = scanBlossom(v, w)
                            if base >= 0:
                                # Found a new blossom; add it to the blossom
                                # bookkeeping and turn it into an S-blossom.
                                addBlossom(base, k)
                            else:
                                # Found an augmenting path; augment the
                                # matching and end this stage.
                                augmentMatching(k)
                                augmented = 1
                                break
                        elif label[w] == 0:
                            # w is inside a T-blossom, but w itself has not
                            # yet been reached from outside the blossom;
                            # mark it as reached (we need this to relabel
                            # during T-blossom expansion).
                            assert label[inblossom[w]] == 2
                            label[w] = 2
                            labelend[w] = p ^ 1
                    elif label[inblossom[w]] == 1:
                        # keep track of the least-slack non-allowable edge to
                        # a different S-blossom.
                        b = inblossom[v]
                        if bestedge[b] == -1 or kslack < slack(bestedge[b]):
                            bestedge[b] = k
                    elif label[w] == 0:
                        # w is a free vertex (or an unreached vertex inside
                        # a T-blossom) but we can not reach it yet;
                        # keep track of the least-slack edge that reaches w.
                        if bestedge[w] == -1 or kslack < slack(bestedge[w]):
                            bestedge[w] = k

            if augmented:
                break

            # There is no augmenting path under these constraints;
            # compute delta and reduce slack in the optimization problem.
            # (Note that our vertex dual variables, edge slacks and delta's
            # are pre-multiplied by two.)
            deltatype = -1
            delta = deltaedge = deltablossom = None

            # Verify data structures for delta2/delta3 computation.
            if CHECK_DELTA:
                checkDelta2()
                checkDelta3()

            # Compute delta1: the minumum value of any vertex dual.
            if not maxcardinality:
                deltatype = 1
                delta = min(dualvar[:nvertex])

            # Compute delta2: the minimum slack on any edge between
            # an S-vertex and a free vertex.
            for v in range(nvertex):
                if label[inblossom[v]] == 0 and bestedge[v] != -1:
                    d = slack(bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v]

            # Compute delta3: half the minimum slack on any edge between
            # a pair of S-blossoms.
            for b in range(2 * nvertex):
                if (blossomparent[b] == -1 and label[b] == 1 and
                        bestedge[b] != -1):
                    kslack = slack(bestedge[b])
                    if isinstance(kslack, numbers.Integral):
                        assert (kslack % 2) == 0
                        d = kslack // 2
                    else:
                        d = kslack / 2
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b]

            # Compute delta4: minimum z variable of any T-blossom.
            for b in range(nvertex, 2 * nvertex):
                if (blossombase[b] >= 0 and blossomparent[b] == -1 and
                        label[b] == 2 and
                        (deltatype == -1 or dualvar[b] < delta)):
                    delta = dualvar[b]
                    deltatype = 4
                    deltablossom = b

            if deltatype == -1:
                # No further improvement possible; max-cardinality optimum
                # reached. Do a final delta update to make the optimum
                # verifyable.
                assert maxcardinality
                deltatype = 1
                delta = max(0, min(dualvar[:nvertex]))

            # Update dual variables according to delta.
            for v in range(nvertex):
                if label[inblossom[v]] == 1:
                    # S-vertex: 2*u = 2*u - 2*delta
                    dualvar[v] -= delta
                elif label[inblossom[v]] == 2:
                    # T-vertex: 2*u = 2*u + 2*delta
                    dualvar[v] += delta
            for b in range(nvertex, 2 * nvertex):
                if blossombase[b] >= 0 and blossomparent[b] == -1:
                    if label[b] == 1:
                        # top-level S-blossom: z = z + 2*delta
                        dualvar[b] += delta
                    elif label[b] == 2:
                        # top-level T-blossom: z = z - 2*delta
                        dualvar[b] -= delta

            # Take action at the point where minimum delta occurred.
            if DEBUG: DEBUG('delta%d=%f' % (deltatype, delta))
            if deltatype == 1:
                # No further improvement possible; optimum reached.
                break
            elif deltatype == 2:
                # Use the least-slack edge to continue the search.
                allowedge[deltaedge] = True
                (i, j, wt) = edges[deltaedge]
                if label[inblossom[i]] == 0:
                    i, j = j, i
                assert label[inblossom[i]] == 1
                queue.append(i)
            elif deltatype == 3:
                # Use the least-slack edge to continue the search.
                allowedge[deltaedge] = True
                (i, j, wt) = edges[deltaedge]
                assert label[inblossom[i]] == 1
                queue.append(i)
            elif deltatype == 4:
                # Expand the least-z blossom.
                expandBlossom(deltablossom, False)

            # End of a this substage.

        # Stop when no more augmenting path can be found.
        if not augmented:
            break

        # End of a stage; expand all S-blossoms which have dualvar = 0.
        for b in range(nvertex, 2 * nvertex):
            if (blossomparent[b] == -1 and blossombase[b] >= 0 and
                    label[b] == 1 and dualvar[b] == 0):
                expandBlossom(b, True)

    # Verify that we reached the optimum solution.
    if CHECK_OPTIMUM:
        verifyOptimum()

    # Transform mate[] such that mate[v] is the vertex to which v is paired.
    for v in range(nvertex):
        if mate[v] >= 0:
            mate[v] = endpoint[mate[v]]
    for v in range(nvertex):
        assert mate[v] == -1 or mate[mate[v]] == v

    return mate


################################################################################
# General coarsening utility functions
################################################################################


def coarsen_vector(x, C):
    return (C.power(2)).dot(x)


def lift_vector(x, C):
    # Pinv = C.T; Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return Pinv.dot(x)


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))


def lift_matrix(W, C):
    P = C.power(2)
    return (P.T).dot(W.dot(P))


def get_coarsening_matrix(G, partitioning):
    """
    This function should be called in order to build the coarsening matrix C.

    Parameters
    ----------
    G : the graph to be coarsened
    partitioning : a list of subgraphs to be contracted

    Returns
    -------
    C : the new coarsening matrix

    Example
    -------
    C = contract(gsp.graphs.sensor(20),[0,1]) ??
    """

    # C = np.eye(G.N)
    C = sp.sparse.eye(G.N, format="lil")

    rows_to_delete = []
    for subgraph in partitioning:
        nc = len(subgraph)

        # add v_j's to v_i's row
        C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  # np.ones((1,nc))/np.sqrt(nc)

        rows_to_delete.extend(subgraph[1:])

    # delete vertices
    # C = np.delete(C,rows_to_delete,0)

    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csc_matrix(C)

    # check that this is a projection matrix
    # assert sp.sparse.linalg.norm( ((C.T).dot(C))**2 - ((C.T).dot(C)) , ord='fro') < 1e-5

    return C


def coarsening_quality(G, C, kmax=30, Uk=None, lk=None):
    """
    Measures how good is a coarsening.

    Parameters
    ----------
    G : pygsp Graph
    C : np.array(n,N)
        The coarsening matrix
    kmax : int
        Until which eigenvalue we are interested in.

    Returns
    -------
    metric : dictionary
        Contains all relevant metrics for coarsening quality:
        * error_eigenvalue : np.array(kmax)
        * error_subspace : np.array(kmax)
        * error_sintheta : np.array(kmax)
        * angle_matrix : np.array(kmax)
        * rss constants : np.array(kmax)
        as well as some general properties of Gc:
        * r : int
            reduction ratio
        * m : int
            number of edges
    """
    N = G.N
    I = np.eye(N)

    if (Uk is not None) and (lk is not None) and (len(lk) >= kmax):
        U, l = Uk, lk
    elif hasattr(G, "U"):
        U, l = G.U, G.e
    else:
        l, U = sp.sparse.linalg.eigsh(G.L, k=kmax, which="SM", tol=1e-3)

    l[0] = 1
    linv = l ** (-0.5)
    linv[0] = 0
    # l[0] = 0 # avoids divide by 0

    # below here, everything is C specific
    n = C.shape[0]
    Pi = C.T @ C
    S = get_S(G).T
    Lc = C.dot((G.L).dot(C.T))
    Lp = Pi @ G.L @ Pi

    if kmax > n / 2:
        [Uc, lc] = eig(Lc.toarray())
    else:
        lc, Uc = sp.sparse.linalg.eigsh(Lc, k=kmax, which="SM", tol=1e-3)

    if not sp.sparse.issparse(Lc):
        print("warning: Lc should be sparse.")

    metrics = {"r": 1 - n / N, "m": int((Lc.nnz - n) / 2)}

    kmax = np.clip(kmax, 1, n)

    # eigenvalue relative error
    metrics["error_eigenvalue"] = np.abs(l[:kmax] - lc[:kmax]) / l[:kmax]
    metrics["error_eigenvalue"][0] = 0

    # angles between eigenspaces
    metrics["angle_matrix"] = U.T @ C.T @ Uc

    # rss constants
    #    metrics['rss'] = np.diag(U.T @ Lp @ U)/l - 1
    #    metrics['rss'][0] = 0

    # other errors
    kmax = np.clip(kmax, 2, n)

    error_subspace = np.zeros(kmax)
    error_subspace_bound = np.zeros(kmax)
    error_sintheta = np.zeros(kmax)

    M = S @ Pi @ U @ np.diag(linv)
    #    M_bound = S @ (I - Pi) @ U @ np.diag(linv)

    for kIdx in range(1, kmax):
        error_subspace[kIdx] = np.abs(np.linalg.norm(M[:, : kIdx + 1], ord=2) - 1)
        #        error_subspace_bound[kIdx] = np.linalg.norm( M_bound[:,:kIdx + 1], ord=2)
        error_sintheta[kIdx] = (
                np.linalg.norm(metrics["angle_matrix"][0: kIdx + 1, kIdx + 1:], ord="fro")
                ** 2
        )

    metrics["error_subspace"] = error_subspace
    # metrics['error_subspace_bound'] = error_subspace_bound
    metrics["error_sintheta"] = error_sintheta

    return metrics


# def plot_coarsening(
#         Gall, Call, size=3, edge_width=0.8, node_size=20, alpha=0.55, title=""
# ):
#     """
#     Plot a (hierarchical) coarsening
#
#     Parameters
#     ----------
#     G_all : list of pygsp Graphs
#     Call  : list of np.arrays
#
#     Returns
#     -------
#     fig : matplotlib figure
#     """
#     # colors signify the size of a coarsened subgraph ('k' is 1, 'g' is 2, 'b' is 3, and so on)
#     colors = ["k", "g", "b", "r", "y"]
#
#     n_levels = len(Gall) - 1
#     if n_levels == 0:
#         return None
#     fig = plt.figure(figsize=(n_levels * size * 3, size * 2))
#
#     for level in range(n_levels):
#
#         G = Gall[level]
#         edges = np.array(G.get_edge_list()[0:2])
#
#         Gc = Gall[level + 1]
#         #         Lc = C.dot(G.L.dot(C.T))
#         #         Wc = sp.sparse.diags(Lc.diagonal(), 0) - Lc;
#         #         Wc = (Wc + Wc.T) / 2
#         #         Gc = gsp.graphs.Graph(Wc, coords=(C.power(2)).dot(G.coords))
#         edges_c = np.array(Gc.get_edge_list()[0:2])
#         C = Call[level]
#         C = C.toarray()
#
#         if G.coords.shape[1] == 2:
#             ax = fig.add_subplot(1, n_levels + 1, level + 1)
#             ax.axis("off")
#             ax.set_title(f"{title} | level = {level}, N = {G.N}")
#
#             [x, y] = G.coords.T
#             for eIdx in range(0, edges.shape[1]):
#                 ax.plot(
#                     x[edges[:, eIdx]],
#                     y[edges[:, eIdx]],
#                     color="k",
#                     alpha=alpha,
#                     lineWidth=edge_width,
#                 )
#             for i in range(Gc.N):
#                 subgraph = np.arange(G.N)[C[i, :] > 0]
#                 ax.scatter(
#                     x[subgraph],
#                     y[subgraph],
#                     c=colors[np.clip(len(subgraph) - 1, 0, 4)],
#                     s=node_size * len(subgraph),
#                     alpha=alpha,
#                 )
#
#         elif G.coords.shape[1] == 3:
#             ax = fig.add_subplot(1, n_levels + 1, level + 1, projection="3d")
#             ax.axis("off")
#
#             [x, y, z] = G.coords.T
#             for eIdx in range(0, edges.shape[1]):
#                 ax.plot(
#                     x[edges[:, eIdx]],
#                     y[edges[:, eIdx]],
#                     zs=z[edges[:, eIdx]],
#                     color="k",
#                     alpha=alpha,
#                     lineWidth=edge_width,
#                 )
#             for i in range(Gc.N):
#                 subgraph = np.arange(G.N)[C[i, :] > 0]
#                 ax.scatter(
#                     x[subgraph],
#                     y[subgraph],
#                     z[subgraph],
#                     c=colors[np.clip(len(subgraph) - 1, 0, 4)],
#                     s=node_size * len(subgraph),
#                     alpha=alpha,
#                 )
#
#     # the final graph
#     Gc = Gall[-1]
#     edges_c = np.array(Gc.get_edge_list()[0:2])
#
#     if G.coords.shape[1] == 2:
#         ax = fig.add_subplot(1, n_levels + 1, n_levels + 1)
#         ax.axis("off")
#         [x, y] = Gc.coords.T
#         ax.scatter(x, y, c="k", s=node_size, alpha=alpha)
#         for eIdx in range(0, edges_c.shape[1]):
#             ax.plot(
#                 x[edges_c[:, eIdx]],
#                 y[edges_c[:, eIdx]],
#                 color="k",
#                 alpha=alpha,
#                 lineWidth=edge_width,
#             )
#
#     elif G.coords.shape[1] == 3:
#         ax = fig.add_subplot(1, n_levels + 1, n_levels + 1, projection="3d")
#         ax.axis("off")
#         [x, y, z] = Gc.coords.T
#         ax.scatter(x, y, z, c="k", s=node_size, alpha=alpha)
#         for eIdx in range(0, edges_c.shape[1]):
#             ax.plot(
#                 x[edges_c[:, eIdx]],
#                 y[edges_c[:, eIdx]],
#                 z[edges_c[:, eIdx]],
#                 color="k",
#                 alpha=alpha,
#                 lineWidth=edge_width,
#             )
#
#     ax.set_title(f"{title} | level = {n_levels}, n = {Gc.N}")
#     fig.tight_layout()
#     return fig


################################################################################
# Variation-based contraction algorithms
################################################################################


def contract_variation_edges(G, A=None, K=10, r=0.5, algorithm="greedy"):
    """
    Sequential contraction with local variation and edge-based families.
    This is a specialized implementation for the edge-based family, that works
    slightly faster than the contract_variation() function, which works for
    any family.

    See contract_variation() for documentation.
    """
    N, deg, M = G.N, G.dw, G.Ne
    ones = np.ones(2)
    Pibot = np.eye(2) - np.outer(ones, ones) / 2

    # cost function for the edge
    def subgraph_cost(G, A, edge):
        edge, w = edge[:2].astype(np.int_), edge[2]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    # cost function for the edge
    def subgraph_cost_old(G, A, edge):
        w = G.W[edge[0], edge[1]]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    # edges = np.array(G.get_edge_list()[0:2])
    edges = np.array(G.get_edge_list())
    weights = np.array([subgraph_cost(G, A, edges[:, e]) for e in range(M)])
    # weights = np.zeros(M)
    # for e in range(M):
    #    weights[e] = subgraph_cost_old(G, A, edges[:,e])

    if algorithm == "optimal":
        # identify the minimum weight matching
        coarsening_list = matching_optimal(G, weights=weights, r=r)

    elif algorithm == "greedy":
        # find a heavy weight matching
        coarsening_list = matching_greedy(G, weights=-weights, r=r)

    return coarsening_list


def contract_variation_linear(G, A=None, K=10, r=0.5, mode="neighborhood"):
    """
    Sequential contraction with local variation and general families.
    This is an implemmentation that improves running speed,
    at the expense of being more greedy (and thus having slightly larger error).

    See contract_variation() for documentation.
    """

    N, deg, W_lil = G.N, G.dw, G.W.tolil()

    # The following is correct only for a single level of coarsening.
    # Normally, A should be passed as an argument.
    if A is None:
        lk, Uk = sp.sparse.linalg.eigsh(
            G.L, k=K, which="SM", tol=1e-3
        )  # this is not optimized!
        lk[0] = 1
        lsinv = lk ** (-0.5)
        lsinv[0] = 0
        lk[0] = 0
        D_lsinv = np.diag(lsinv)
        A = Uk @ np.diag(lsinv)

    # cost function for the subgraph induced by nodes array
    def subgraph_cost(nodes):
        nc = len(nodes)
        ones = np.ones(nc)
        W = W_lil[nodes, :][:, nodes]  # .tocsc()
        L = np.diag(2 * deg[nodes] - W.dot(ones)) - W
        B = (np.eye(nc) - np.outer(ones, ones) / nc) @ A[nodes, :]
        return np.linalg.norm(B.T @ L @ B) / (nc - 1)

    class CandidateSet:
        def __init__(self, candidate_list):
            self.set = candidate_list
            self.cost = subgraph_cost(candidate_list)

        def __lt__(self, other):
            return self.cost < other.cost

    family = []
    W_bool = G.A + sp.sparse.eye(G.N, dtype=np.bool_, format="csr")
    if "neighborhood" in mode:
        for i in range(N):
            # i_set = G.A[i,:].indices # graph_utils.get_neighbors(G, i)
            # i_set = np.append(i_set, i)
            i_set = W_bool[i, :].indices
            family.append(CandidateSet(i_set))

    if "cliques" in mode:

        Gnx = nx.from_scipy_sparse_array(G.W)
        for clique in nx.find_cliques(Gnx):
            family.append(CandidateSet(np.array(clique)))

    else:
        if "edges" in mode:
            edges = np.array(G.get_edge_list()[0:2])
            for e in range(0, edges.shape[1]):
                family.append(CandidateSet(edges[:, e]))
        if "triangles" in mode:
            triangles = set([])
            edges = np.array(G.get_edge_list()[0:2])
            for e in range(0, edges.shape[1]):
                [u, v] = edges[:, e]
                for w in range(G.N):
                    if G.W[u, w] > 0 and G.W[v, w] > 0:
                        triangles.add(frozenset([u, v, w]))
            triangles = list(map(lambda x: np.array(list(x)), triangles))
            for triangle in triangles:
                family.append(CandidateSet(triangle))

    family = SortedList(family)
    marked = np.zeros(G.N, dtype=np.bool_)

    # ----------------------------------------------------------------------------
    # Construct a (minimum weight) independent set.
    # ----------------------------------------------------------------------------
    coarsening_list = []
    # n, n_target = N, (1-r)*N
    n_reduce = np.floor(r * N)  # how many nodes do we need to reduce/eliminate?

    while len(family) > 0:

        i_cset = family.pop(index=0)
        i_set = i_cset.set

        # check if marked
        i_marked = marked[i_set]

        if not any(i_marked):

            n_gain = len(i_set) - 1
            if n_gain > n_reduce:
                continue  # this helps avoid over-reducing

            # all vertices are unmarked: add i_set to the coarsening list
            marked[i_set] = True
            coarsening_list.append(i_set)
            # n -= len(i_set) - 1
            n_reduce -= n_gain

            # if n <= n_target: break
            if n_reduce <= 0:
                break

        # may be worth to keep this set
        else:
            i_set = i_set[~i_marked]
            if len(i_set) > 1:
                # todo1: check whether to add to coarsening_list before adding to family
                # todo2: currently this will also select contraction sets that are disconnected
                # should we eliminate those?
                i_cset.set = i_set
                i_cset.cost = subgraph_cost(i_set)
                family.add(i_cset)

    return coarsening_list


################################################################################
# Edge-based contraction algorithms
################################################################################


def get_proximity_measure(G, name, K=10):
    N = G.N
    W = G.W  # np.array(G.W.toarray(), dtype=np.float32)
    deg = G.dw  # np.sum(W, axis=0)
    edges = np.array(G.get_edge_list()[0:2])
    weights = np.array(G.get_edge_list()[2])
    M = edges.shape[1]

    num_vectors = K  # int(1*K*np.log(K))
    if "lanczos" in name:
        l_lan, X_lan = sp.sparse.linalg.eigsh(G.L, k=K, which="SM", tol=1e-2)
    elif "cheby" in name:
        X_cheby = generate_test_vectors(
            G, num_vectors=num_vectors, method="Chebychev", lambda_cut=G.e[K + 1]
        )
    elif "JC" in name:
        X_jc = generate_test_vectors(
            G, num_vectors=num_vectors, method="JC", iterations=20
        )
    elif "GS" in name:
        X_gs = generate_test_vectors(
            G, num_vectors=num_vectors, method="GS", iterations=1
        )
    if "expected" in name:
        X = X_lan
        assert not np.isnan(X).any()
        assert X.shape[0] == N
        K = X.shape[1]

    proximity = np.zeros(M, dtype=np.float32)

    # heuristic for mutligrid
    if name == "heavy_edge":
        wmax = np.array(np.max(G.W, 0).todense())[0] + 1e-5
        for e in range(0, M):
            proximity[e] = weights[e] / max(
                wmax[edges[:, e]]
            )  # select edges with large proximity
        return proximity

    # heuristic for mutligrid
    elif name == "algebraic_JC":
        proximity += np.Inf
        for e in range(0, M):
            i, j = edges[:, e]
            for kIdx in range(num_vectors):
                xk = X_jc[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(np.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

        return proximity

    # heuristic for mutligrid
    elif name == "affinity_GS":
        c = np.zeros((N, N))
        for e in range(0, M):
            i, j = edges[:, e]
            c[i, j] = (X_gs[i, :] @ X_gs[j, :].T) ** 2 / (
                    (X_gs[i, :] @ X_gs[i, :].T) ** 2 * (X_gs[j, :] @ X_gs[j, :].T) ** 2
            )  # select

        c += c.T
        c -= np.diag(np.diag(c))
        for e in range(0, M):
            i, j = edges[:, e]
            proximity[e] = c[i, j] / (max(c[i, :]) * max(c[j, :]))

        return proximity

    for e in range(0, M):
        i, j = edges[:, e]

        if name == "heavy_edge_degree":
            proximity[e] = (
                    deg[i] + deg[j] + 2 * G.W[i, j]
            )  # select edges with large proximity

        # loose as little information as possible (custom)
        elif "min_expected_loss" in name:
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [proximity[e], (xk[i] - xk[j]) ** 2]
                )  # select edges with small proximity
        # loose as little gradient information as possible (custom)
        elif name == "min_expected_gradient_loss":
            for kIdx in range(1, K):
                xk = X[:, kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2 * (deg[i] + deg[j] + 2 * G.W[i, j]),
                    ]
                )  # select edges with small proximity

        # relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss":
            for kIdx in range(1, K):
                xk = G.U[:, kIdx]
                lk = G.e[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4)
                        / lk,
                    ]
                )  # select edges with small proximity

        # fast relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_lanczos":
            for kIdx in range(1, K):
                xk = X_lan[:, kIdx]
                lk = l_lan[kIdx]
                proximity[e] = sum(
                    [
                        proximity[e],
                        (xk[i] - xk[j]) ** 2
                        * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0.5 * (lk + lk))
                        / lk,
                    ]
                )  # select edges with small proximity

        # approximate relaxation ensuring that K first eigenspaces are aligned (custom)
        elif name == "rss_cheby":
            for kIdx in range(num_vectors):
                xk = X_cheby[:, kIdx]
                lk = xk.T @ G.L @ xk
                proximity[e] = sum(
                    [
                        proximity[e],
                        (
                                (xk[i] - xk[j]) ** 2
                                * ((deg[i] + deg[j] + 2 * G.W[i, j]) / 4 - 0 * lk)
                                / lk
                        ),
                    ]
                )  # select edges with small proximity

        # heuristic for mutligrid (algebraic multigrid)
        elif name == "algebraic_GS":
            proximity[e] = np.Inf
            for kIdx in range(num_vectors):
                xk = X_gs[:, kIdx]
                proximity[e] = min(
                    proximity[e], 1 / max(np.abs(xk[i] - xk[j]) ** 2, 1e-6)
                )  # select edges with large proximity

    if ("rss" in name) or ("expected" in name):
        proximity = -proximity

    return proximity


def generate_test_vectors(
        G, num_vectors=10, method="Gauss-Seidel", iterations=5, lambda_cut=0.1
):
    L = G.L
    N = G.N
    X = np.random.randn(N, num_vectors) / np.sqrt(N)

    if method == "GS" or method == "Gauss-Seidel":

        L_upper = sp.sparse.triu(L, 1, format="csc")
        L_lower_diag = sp.sparse.triu(L, 0, format="csc").T

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                x = -sp.sparse.linalg.spsolve_triangular(L_lower_diag, L_upper @ x)
            X[:, j] = x
        return X

    if method == "JC" or method == "Jacobi":

        deg = G.dw.astype(np.float_)
        D = sp.sparse.diags(deg, 0)
        deginv = deg ** (-1)
        deginv[deginv == np.Inf] = 0
        Dinv = sp.sparse.diags(deginv, 0)
        M = Dinv.dot(D - L)

        for j in range(num_vectors):
            x = X[:, j]
            for t in range(iterations):
                x = 0.5 * x + 0.5 * M.dot(x)
            X[:, j] = x
        return X

    elif method == "Chebychev":

        f = filters.Filter(G, lambda x: ((x <= lambda_cut) * 1).astype(np.float32))
        return f.filter(X, method="chebyshev", order=50)


def matching_optimal(G, weights, r=0.4):
    """
    Generates a matching optimally with the objective of minimizing the total
    weight of all edges in the matching.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    ratio : float
        The desired dimensionality reduction (ratio = 1 - n/N)

    Notes:
    * The complexity of this is O(N^3)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """
    N = G.N

    # the edge set
    edges = G.get_edge_list()
    edges = np.array(edges[0:2])
    M = edges.shape[1]

    max_weight = 1 * np.max(weights)

    # prepare the input for the minimum weight matching problem
    edge_list = []
    for edgeIdx in range(M):
        [i, j] = edges[:, edgeIdx]
        if i == j:
            continue
        edge_list.append((i, j, max_weight - weights[edgeIdx]))

    assert min(weights) >= 0

    # solve it
    tmp = np.array(maxWeightMatching(edge_list))

    # format output
    m = tmp.shape[0]
    matching = np.zeros((m, 2), dtype=int)
    matching[:, 0] = range(m)
    matching[:, 1] = tmp

    # remove null edges and duplicates
    idx = np.where(tmp != -1)[0]
    matching = matching[idx, :]
    idx = np.where(matching[:, 0] > matching[:, 1])[0]
    matching = matching[idx, :]

    assert matching.shape[0] >= 1

    # if the returned matching is larger than what is requested, select the min weight subset of it
    matched_weights = np.zeros(matching.shape[0])
    for mIdx in range(matching.shape[0]):
        i = matching[mIdx, 0]
        j = matching[mIdx, 1]
        eIdx = [
            e
            for e, t in enumerate(edges[:, :].T)
            if ((t == [i, j]).all() or (t == [j, i]).all())
        ]
        matched_weights[mIdx] = weights[eIdx]

    keep = min(int(np.ceil(r * N)), matching.shape[0])
    if keep < matching.shape[0]:
        idx = np.argpartition(matched_weights, keep)
        idx = idx[0:keep]
        matching = matching[idx, :]

    return matching


def matching_greedy(G, weights, r=0.4):
    """
    Generates a matching greedily by selecting at each iteration the edge
    with the largest weight and then removing all adjacent edges from the
    candidate set.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    r : float
        The desired dimensionality reduction (r = 1 - n/N)

    Notes:
    * The complexity of this is O(M)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """

    N = G.N

    # the edge set
    edges = np.array(G.get_edge_list()[0:2])
    M = edges.shape[1]

    idx = np.argsort(-weights)
    # idx = np.argsort(weights)[::-1]
    edges = edges[:, idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []

    # which vertices have been selected
    marked = np.zeros(N, dtype=np.bool_)

    n, n_target = N, (1 - r) * N
    while len(candidate_edges) > 0:

        # pop a candidate edge
        [i, j] = candidate_edges.pop(0)

        # check if marked
        if any(marked[[i, j]]):
            continue

        marked[[i, j]] = True
        n -= 1

        # add it to the matching
        matching.append(np.array([i, j]))

        # termination condition
        if n <= n_target:
            break

    return np.array(matching)


##############################################################################
# Sparsification and Kron reduction
# Most of the code has been adapted from the PyGSP implementation.
##############################################################################
def kron_coarsening(G, r=0.5, m=None):
    if not hasattr(G, "coords"):
        G.set_coordinates(np.random.rand(G.N, 2))  # needed by kron

    n_target = np.floor((1 - r) * G.N)
    levels = int(np.ceil(np.log2(G.N / n_target)))

    try:
        Gs = my_graph_multiresolution(
            G,
            levels,
            r=r,
            sparsify=False,
            sparsify_eps=None,
            reduction_method="kron",
            reg_eps=0.01,
        )
        Gk = Gs[-1]

        # sparsify to a target number of edges m
        if m is not None:
            M = Gk.Ne  # int(Gk.W.nnz/2)
            epsilon = min(10 / np.sqrt(G.N), 0.3)  # 1 - m/M
            Gc = graph_sparsify(Gk, epsilon, maxiter=10)
            Gc.mr = Gk.mr
        else:
            Gc = Gk

        return Gc, Gs[0]

    except:
        return None, None


def kron_quality(G, Gc, kmax=30, Uk=None, lk=None):
    N, n = G.N, Gc.N
    keep_inds = Gc.mr["idx"]

    metrics = {"r": 1 - n / N, "m": int(Gc.W.nnz / 2), "failed": False}
    kmax = np.clip(kmax, 1, n)

    if (Uk is not None) and (lk is not None) and (len(lk) >= kmax):
        U, l = Uk, lk
    elif hasattr(G, "U"):
        U, l = G.U, G.e
    else:
        l, U = sp.sparse.linalg.eigsh(G.L, k=kmax, which="SM", tol=1e-3)

    l[0] = 1
    linv = l ** (-0.5)
    linv[0] = 0
    # avoids divide by 0

    C = np.eye(N)
    C = C[keep_inds, :]
    L = G.L.toarray()

    try:
        Phi = np.linalg.pinv(L + 0.01 * np.eye(N))
        Cinv = (Phi @ C.T) @ np.linalg.pinv(C @ Phi @ C.T)

        if kmax > n / 2:
            [Uc, lc] = eig(Gc.L.toarray())
        else:
            lc, Uc = sp.sparse.linalg.eigsh(Gc.L, k=kmax, which="SM", tol=1e-3)

        # eigenvalue relative error
        metrics["error_eigenvalue"] = np.abs(l[:kmax] - lc[:kmax]) / l[:kmax]
        metrics["error_eigenvalue"][0] = 0

        # TODO : angles between eigenspaces
        # metrics['angle_matrix'] = U.T @ C.T @ Uc

        # other errors
        kmax = np.clip(kmax, 2, n)

        error_subspace = np.zeros(kmax)
        error_sintheta = np.zeros(kmax)

        # M = (Lsqrtm - sp.linalg.sqrtm(Cinv @ Gc.L.dot(C))) @ U @ np.diag(linv) # is this correct?
        M = U - sp.linalg.sqrtm(Cinv @ Gc.L.dot(C)) @ U @ np.diag(
            linv
        )  # is this correct?
        for kIdx in range(0, kmax):
            error_subspace[kIdx] = np.abs(np.linalg.norm(M[:, : kIdx + 1], ord=2) - 1)

        metrics["error_subspace"] = error_subspace
        metrics["error_sintheta"] = error_sintheta

    except:
        metrics["failed"] = True

    return metrics


def kron_interpolate(G, Gc, x):
    return np.squeeze(reduction.interpolate(G, x, Gc.mr["idx"]))


def my_graph_multiresolution(
        G,
        levels,
        r=0.5,
        sparsify=True,
        sparsify_eps=None,
        downsampling_method="largest_eigenvector",
        reduction_method="kron",
        compute_full_eigen=False,
        reg_eps=0.005,
):
    r"""Compute a pyramid of graphs (by Kron reduction).

    'graph_multiresolution(G,levels)' computes a multiresolution of
    graph by repeatedly downsampling and performing graph reduction. The
    default downsampling method is the largest eigenvector method based on
    the polarity of the components of the eigenvector associated with the
    largest graph Laplacian eigenvalue. The default graph reduction method
    is Kron reduction followed by a graph sparsification step.
    *param* is a structure of optional parameters.

    Parameters
    ----------
    G : Graph structure
        The graph to reduce.
    levels : int
        Number of level of decomposition
    lambd : float
        Stability parameter. It adds self loop to the graph to give the
        algorithm some stability (default = 0.025). [UNUSED?!]
    sparsify : bool
        To perform a spectral sparsification step immediately after
        the graph reduction (default is True).
    sparsify_eps : float
        Parameter epsilon used in the spectral sparsification
        (default is min(10/sqrt(G.N),.3)).
    downsampling_method: string
        The graph downsampling method (default is 'largest_eigenvector').
    reduction_method : string
        The graph reduction method (default is 'kron')
    compute_full_eigen : bool
        To also compute the graph Laplacian eigenvalues and eigenvectors
        for every graph in the multiresolution sequence (default is False).
    reg_eps : float
        The regularized graph Laplacian is :math:`\bar{L}=L+\epsilon I`.
        A smaller epsilon may lead to better regularization, but will also
        require a higher order Chebyshev approximation. (default is 0.005)

    Returns
    -------
    Gs : list
        A list of graph layers.

    Examples
    --------
    >>> from pygsp import reduction
    >>> levels = 5
    >>> G = graphs.Sensor(N=512)
    >>> G.compute_fourier_basis()
    >>> Gs = reduction.graph_multiresolution(G, levels, sparsify=False)
    >>> for idx in range(levels):
    ...     Gs[idx].plotting['plot_name'] = 'Reduction level: {}'.format(idx)
    ...     Gs[idx].plot()

    """
    if sparsify_eps is None:
        sparsify_eps = min(10.0 / np.sqrt(G.N), 0.3)

    if compute_full_eigen:
        G.compute_fourier_basis()
    else:
        G.estimate_lmax()

    Gs = [G]
    Gs[0].mr = {"idx": np.arange(G.N), "orig_idx": np.arange(G.N)}

    n_target = int(np.floor(G.N * (1 - r)))

    for i in range(levels):
        if downsampling_method == "largest_eigenvector":
            if hasattr(Gs[i], "_U"):
                V = Gs[i].U[:, -1]
            else:
                V = sp.sparse.linalg.eigs(Gs[i].L, 1)[1][:, 0]

            V *= np.sign(V[0])
            n = max(int(Gs[i].N / 2), n_target)

            ind = np.argsort(V)  # np.nonzero(V >= 0)[0]
            ind = np.flip(ind, 0)
            ind = ind[:n]

        else:
            raise NotImplementedError("Unknown graph downsampling method.")

        if reduction_method == "kron":
            Gs.append(reduction.kron_reduction(Gs[i], ind))

        else:
            raise NotImplementedError("Unknown graph reduction method.")

        if sparsify and Gs[i + 1].N > 2:
            Gs[i + 1] = reduction.graph_sparsify(
                Gs[i + 1], min(max(sparsify_eps, 2.0 / np.sqrt(Gs[i + 1].N)), 1.0)
            )

        if Gs[i + 1].is_directed():
            W = (Gs[i + 1].W + Gs[i + 1].W.T) / 2
            Gs[i + 1] = graphs.Graph(W, coords=Gs[i + 1].coords)

        if compute_full_eigen:
            Gs[i + 1].compute_fourier_basis()
        else:
            Gs[i + 1].estimate_lmax()

        Gs[i + 1].mr = {"idx": ind, "orig_idx": Gs[i].mr["orig_idx"][ind], "level": i}

        L_reg = Gs[i].L + reg_eps * sp.sparse.eye(Gs[i].N)
        Gs[i].mr["K_reg"] = reduction.kron_reduction(L_reg, ind)
        Gs[i].mr["green_kernel"] = filters.Filter(Gs[i], lambda x: 1.0 / (reg_eps + x))

    return Gs


def graph_sparsify(M, epsilon, maxiter=10):
    # Test the input parameters
    if isinstance(M, graphs.Graph):
        if not M.lap_type == "combinatorial":
            raise NotImplementedError
        L = M.L
    else:
        L = M

    N = np.shape(L)[0]

    if not 1.0 / np.sqrt(N) <= epsilon < 1:
        raise ValueError("GRAPH_SPARSIFY: Epsilon out of required range")

    # Not sparse
    resistance_distances = resistance_distance(L).toarray()
    # Get the Weight matrix
    if isinstance(M, graphs.Graph):
        W = M.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0

    W = sp.sparse.coo_matrix(W)
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = sp.sparse.find(sp.sparse.tril(W))

    # Calculate the new weights.
    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re + 1e-4
    Pe = Pe / np.sum(Pe)

    for i in range(maxiter):
        # Rudelson, 1996 Random Vectors in the Isotropic Position
        # (too hard to figure out actual C0)
        C0 = 1 / 30.0
        # Rudelson and Vershynin, 2007, Thm. 3.1
        C = 4 * C0
        q = round(N * np.log(N) * 9 * C ** 2 / (epsilon ** 2))

        results = sp.stats.rv_discrete(values=(np.arange(np.shape(Pe)[0]), Pe)).rvs(
            size=int(q)
        )
        spin_counts = sp.stats.itemfreq(results).astype(int)
        per_spin_weights = weights / (q * Pe)

        counts = np.zeros(np.shape(weights)[0])
        counts[spin_counts[:, 0]] = spin_counts[:, 1]
        new_weights = counts * per_spin_weights

        sparserW = sp.sparse.csc_matrix(
            (new_weights, (start_nodes, end_nodes)), shape=(N, N)
        )
        sparserW = sparserW + sparserW.T
        sparserL = sp.sparse.diags(sparserW.diagonal(), 0) - sparserW

    #        if graphs.Graph(W=sparserW).is_connected():
    #            break
    #        elif i == maxiter - 1:
    #            print('Despite attempts to reduce epsilon, sparsified graph is disconnected')
    #        else:
    #            epsilon -= (epsilon - 1/np.sqrt(N)) / 2.

    if isinstance(M, graphs.Graph):
        sparserW = sp.sparse.diags(sparserL.diagonal(), 0) - sparserL
        if not M.is_directed():
            sparserW = (sparserW + sparserW.T) / 2.0

        Mnew = graphs.Graph(W=sparserW)
        # M.copy_graph_attributes(Mnew)
    else:
        Mnew = sp.sparse.lil_matrix(sparserL)

    return Mnew


def get_S(G):
    """
    Construct the N x |E| gradient matrix S
    """
    # the edge set
    edges = G.get_edge_list()
    weights = np.array(edges[2])
    edges = np.array(edges[0:2])
    M = edges.shape[1]

    # Construct the N x |E| gradient matrix S
    S = np.zeros((G.N, M))
    for e in np.arange(M):
        S[edges[0, e], e] = np.sqrt(weights[e])
        S[edges[1, e], e] = -np.sqrt(weights[e])

    return S


def eig(A, order='ascend'):
    # eigenvalue decomposition
    [l, X] = np.linalg.eigh(A)

    # reordering indices
    idx = l.argsort()
    if order == 'descend':
        idx = idx[::-1]

    # reordering
    l = np.real(l[idx])
    X = X[:, idx]
    return (X, np.real(l))


def zero_diag(A):
    import scipy as sp

    if sp.sparse.issparse(A):
        return A - sp.sparse.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=(A.shape[0], A.shape[1]))
    else:
        D = A.diagonal()
        return A - np.diag(D)
##############################################################################
