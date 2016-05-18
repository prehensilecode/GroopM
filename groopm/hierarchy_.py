#!/usr/bin/env python
###############################################################################
#                                                                             #
#    hierarchy.py                                                             #
#                                                                             #
#    Working with hierarchical clusterings                                    #
#                                                                             #
#    Copyright (C) Tim Lamberton                                              #
#                                                                             #
###############################################################################
#                                                                             #
#          .d8888b.                                    888b     d888          #
#         d88P  Y88b                                   8888b   d8888          #
#         888    888                                   88888b.d88888          #
#         888        888d888 .d88b.   .d88b.  88888b.  888Y88888P888          #
#         888  88888 888P"  d88""88b d88""88b 888 "88b 888 Y888P 888          #
#         888    888 888    888  888 888  888 888  888 888  Y8P  888          #
#         Y88b  d88P 888    Y88..88P Y88..88P 888 d88P 888   "   888          #
#          "Y8888P88 888     "Y88P"   "Y88P"  88888P"  888       888          #
#                                             888                             #
#                                             888                             #
#                                             888                             #
#                                                                             #
###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

__author__ = "Tim Lamberton"
__copyright__ = "Copyright 2016"
__credits__ = ["Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################

import numpy as np
import scipy.cluster.hierarchy as sp_hierarchy
import scipy.spatial.distance as sp_distance
from Queue import PriorityQueue

# local imports
import distance
from classification import ClassificationManager, ClassificationConsensusFinder

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################        
class ClassificationCoherenceClusterTool:
    """Partition a hierarchical clustering using the taxonomic classification
    distances of marker gene hits to identify clusters that maximise a measure
    of taxonomic coherence. 
    """
    def __init__(self, markers):
        self._mapping = markers
        
    def cluster_classification(self, Z, t, greedy):
        Z = np.asarray(Z)
        n = Z.shape[0] + 1
        
        mcf = ClassificationConsensusFinder(self._mapping, t)
        mll = ClassificationLeavesLister(Z, self._mapping)
        #mct = HierarchyCliqueFinder(Z, t, self._mapping)
        mnodes = mll.nodes
        #Size difference between of 1st- and 2nd-maximal cliques of `Q(k)`
        mcc = np.array([mcf.disagreement(i) for i in (mll.leaves_list(k) for k in mnodes)])
        #mcc = np.array([2*len(mcf.maxClique(i)) - len(i) for i in (mll.leaves_list(k) for k in mnodes)])
        
        cc = np.zeros(2*n - 1, dtype=mcc.dtype)
        cc[mnodes] = np.where(mcc < 0, 0, mcc)
        
        
        # The root nodes of the flat clusters begin as nodes with maximum
        # coefficient.
        rootinds = maxcoeff_roots(Z, cc)
        rootancestors = ancestors(Z, rootinds)
        
        if greedy:
            # Greedily extend clusters until a node with an actively lower
            # coefficient is reached. Requires an additional pass over
            # hierarchy.
            rootinds = np.intersect1d(mnodes, rootancestors)
            rootancestors = ancestors(Z, rootinds, inclusive=True)
            
        # Partition by finding the sets of leaves of the forest created by
        # removing ancestor nodes of forest root nodes.
        T = cluster_remove(Z, rootancestors)
        return T

    
def cluster_remove(Z, remove):
    """Form flat clusters from hierarchical clustering defined by linkage matrix
    `Z` .
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    remove : ndarray
        1-D array of node indices to "remove" from the cluster hierarchy before
        forming flat clusters from the remaining forest.
        
    Returns
    -------
    T : ndarray
        1-D array. `T[i]` is the flat cluster number to which original
        observation `i` belongs.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    
    monocrit = np.zeros(2*n-1, dtype=int)
    monocrit[remove] = 1
    # work around scipy 0.14 bug
    Zz = Z.copy()
    Zz[:, 2] = monocrit[n:]
    T = sp_hierarchy.fcluster(Zz, 0, criterion="distance")
    #T = sp_hierarchy.fcluster(Z, 0, criterion="monocrit", monocrit=monocrit[n:]) # should work in scipy 0.17
    return T

       
def maxcoeff_roots(Z, coeffs):
    """Returns nodes with highest coefficient of any parent node, and at least as 
    high as any descendent.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    coeffs : ndarray
        1-D array. `coeffs[i]` for `i<n` is the coefficient for the i-th
        singleton node, and for `i>=n` is the coefficient for the cluster
        encoded by the `(i-n)`-th row in `Z`.
        
    Returns
    -------
    nondescendents : ndarray
        1-D array of node indices `i` where `coeffs[i] == coeffs[Q[i]].max` and 
        `coeffs[i] > coeffs[Q(j)].max()` for all parent nodes `j`, i.e. with `i`
        in `Q(j)`, where `Q(j)` is the set of all node indices below and
        including node j.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    
    # Algorithm traverses the cluster hierarchy twice times.
    
    # The first time, we find nodes where the node coefficient is
    # non-negative and equal to the maximum of all non-negative
    # descendents (including itself). These are nodes where the coefficient
    # maximum is non-decreasing along all leaf-to-root paths. 
    maxcc = maxcoeffs(Z, coeffs)
    maxinds = np.flatnonzero(maxcc == coeffs)
    
    # The second time, we descend from the root until a node identified in
    # the first pass or a leaf node is encountered. These are nodes where
    # the coefficient is greatest along any root-to-leaf path. 
    maxinds = filter_descendents(Z, maxinds)
    
    return maxinds

    
def ancestors(Z, indices, inclusive=False):
    """Compute ancestor node indices.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    indices : ndarray
        1-D array of node indices.
    inclusive : boolean, optional
        If `True`, indices are counted as their own ancestors.
        
    Returns
    -------
    ancestors : ndarray
        1-D array of node indices of the union of the sets of ancestors of input nodes. 
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    isancestor = np.zeros(2*n-1, dtype=bool)
    isancestor_or_index = isancestor.copy()
    isancestor_or_index[indices] = True
    for i in range(n-1):
        isancestor[i+n] = isancestor[i+n] or isancestor_or_index[Z[i,:2].astype(int)].any()
        isancestor_or_index[i+n] = isancestor_or_index[i+n] or isancestor[i+n]
        
    if inclusive:
        return np.flatnonzero(isancestor_or_index)
    else:
        return np.flatnonzero(isancestor)
        
    
def filter_descendents(Z, indices):
    """Find nodes that are not descendents of other nodes.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    indices : ndarray
        1-D array of node indices.
        
    Returns
    -------
    nondescendents : ndarray
        1-D array of node indices `j` where `j` is either `i` or not in `Q(i)`
        for all `i` in `indices`, where `Q(i)` is the set of nodes below and
        including node i.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    indices = set(indices)
    outarr = []
    stack = [2*n - 2]
    while True:
        if len(stack) == 0:
            break
        i = stack.pop()
        if i in indices:
            outarr.append(i)
        elif i >= n:
            stack.extend(Z[i-n,:2].astype(int))
        
    return np.sort(outarr)
        

def maxcoeffs(Z, coeffs):
    """Compute the maximum coefficient of any descendent for nodes in
    hierarchical clustering.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    coeffs : ndarray
        1-D array of coefficients for each cluster node. `coeffs[i]` for `i<n`
        is the coefficient for the i-th leaf node, and for `i>=n` is the
        coefficient for the cluster encoded by the `(i-n)`-th row in `Z`.
        
    Returns
    -------
    maxcoeffs : ndarray
        `maxcoeffs[i]` is the maximum coefficient value of any cluster below and
        including the node i. More specifically
        `maxcoeffs[i] == coeff[Q(i)].max()` where `Q(i)` is the set of all nodes
        below and including node i. 
    """
    
    Z = np.asarray(Z)  
    n = Z.shape[0] + 1
    coeffs = np.asarray(coeffs)
    if coeffs.shape[0] != 2*n - 1:
        raise ValueError("Number of coefficients must equal the number of"
                         "clusters encoded by linkage matrix")
                       
    outarr = coeffs.copy()
    for i in range(n-1):
        outarr[n+i] = np.maximum(outarr[n+i], outarr[Z[i,:2].astype(int)].max())
    
    return outarr
    
    
def height(Z):
    """Generate a condensed matrix of common ancestor node heights.
    """
    Z = np.copy(Z)
    Z[:, 2] = np.arange(Z.shape[0])
    return sp_hierarchy.cophenet(Z).astype(int)
    
    
def leaves(Z, k):
    """Compute leaf nodes of a cluster"""
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    outarr = []
    stack = [k]
    while len(stack) > 0:
        i = stack.pop()
        if i < n:
            outarr.append(i)
        else:
            stack.extend(Z[i-n, :2].astype(int))
            
    return np.sort(outarr)

    
class ClassificationLeavesLister:
    """Find descendents for nodes in hierarchical clustering.
    
    Parameters
    ----------
    Z: ndarray
        Linkage matrix encoding hierarchical clustering.
    markers: Markers instance
        See ProfileManager class documentation
    """
    def __init__(self, Z, markers):
        Z = np.asarray(Z)
        n = Z.shape[0] + 1
        
        height_map = flat_nodes(Z) # map indices in Z to equal height ancestor
        H = height_map[height(Z)]
        
        indices = np.asarray(markers.rowIndices)
        #idx = distance.ccoords(indices, np.arange(n), n)
        #self._mA = np.where(idx==-1, indices[:, None]*(idx==-1), H[idx]+n)

        mA = np.empty((len(indices), n), dtype=H.dtype)
        for (i, ix) in enumerate(indices):
            for j in range(n):
                if ix == j:
                    mA[i, j] = ix
                else:
                    mA[i, j] = H[distance.condensed_index(n, ix, j)]+n
                    
        self._mA = mA
        
        """Nodes of Z that correspond to embedded hierarchy nodes."""
        self.nodes = np.unique(self._mA[:, indices])
        
    def leaves_list(self, node):
        """Computes the original observations composing a node."""
        return np.flatnonzero(np.any(self._mA==node, axis=1))

        
def flat_nodes(Z):
    """Return node indices of the furtherest ancestor of each node (including itself) of equal height
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    
    node_ids = np.arange(n-1)
    for i in range(n-2, -1, -1):
        children = Z[i, :2].astype(int)
        for c in children:
            if c >= n and Z[i, 2] == Z[c - n, 2]:
                node_ids[c - n] = node_ids[i]
            
    return node_ids
        
        
def linkage_from_reachability(o, d):
    """Hierarchical clustering from reachability ordering and distances"""
    o = np.asarray(o)
    d = np.asarray(d)
    n = len(o)
    Z = np.empty((n - 1, 4), dtype=d.dtype)
    
    ordered_dists = d[o]
    sorting_indices = np.hstack((ordered_dists[1:].argsort()+1, 0))
    indices_dict = dict([(2*n-2, (0, n))])
    
    for i in range(n-2, -1, -1):
        (low, high) = indices_dict.pop(n+i)
        split = sorting_indices[i]
        if split == low + 1:
            left_node = o[low]
        else:
            left_node = np.flatnonzero(np.logical_and(low <= sorting_indices[:i], sorting_indices[:i] < split))[-1]+n
            indices_dict[left_node] = (low, split)
            
        if split == high - 1:
            right_node = o[split]
        else:
            right_node = np.flatnonzero(np.logical_and(split <= sorting_indices[:i], sorting_indices[:i] < high))[-1]+n
            indices_dict[right_node] = (split, high)
        
        if left_node < right_node:
            Z[i, :2] = np.array([left_node, right_node])
        else:
            Z[i, :2] = np.array([right_node, left_node])
        Z[i, 2] = ordered_dists[split]
        Z[i, 3] = high - low
        
    return Z
        
        
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
