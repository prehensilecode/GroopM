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

# local imports
import distance
from Classification import ClassificationManager

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
        
    def cluster_classification(self, Z, t, greedy=False):
        Z = np.asarray(Z)
        n = Z.shape[0] + 1
        H = height(Z)
        
        indices = self._mapping.rowIndices
        classifier = ClassificationManager(self._mapping)
        (rows, cols) = np.ix_(indices, indices)
        idx = distance.squareform_coords(rows, cols, n)
        mA = np.where(rows==cols, np.diag(indices), H[idx]+n)
        mC = np.logical_not(sp_distance.squareform(classifier.makeDisconnectivity(t)))
        (mcc, mnodes) = connectivity_coeffs(mA, mC)
        
        cc = np.zeros(2*n - 1, dtype=np.dtype(mcc))
        cc[mnodes] = mcc
        
        # Algorithm traverses the cluster hierarchy three times.
        
        # The first time, we find nodes where the node coefficient is
        # non-negative and equal to the maximum of all non-negative
        # descendents. These are nodes where the coefficient maximum is
        # non-decreasing along all leaf-to-root paths. 
        maxcc = maxcoeffs(Z, np.where(cc < 0, 0, cc))
        maxinds = np.flatnonzero(maxcc == cc[n:])
        
        # The second time, we descend from the root until a node identified in
        # the first pass or a leaf node is encountered. These are nodes where
        # the coefficient is greatest along any root-to-leaf path. 
        maxinds = filter_descendents(Z, maxinds)
        
        # The root nodes of the flat clusters begin as nodes with maximum
        # coefficient.
        rootinds = maxinds
        rootancestors = ancestors(Z, rootinds)
        
        if greedy:
            # Greedily extend clusters until a node with an actively lower
            # coefficient is reached. Requires an additional pass over
            # hierarchy.
            rootinds = np.intersect1d(mnodes-n, rootancestors)
            rootancestors = ancestors(Z, rootinds, inclusive=True)
            
        # Partition by finding the sets of leaves of the forest created by
        # removing ancestor nodes of forest root nodes.
        remove = np.zeros(n-1, dtype=bool)
        remove[rootancestors] = True
        T = cluster_remove(Z, remove)
        return T

        
def connectivity_coeffs(A, C):
    """Find connectivity coefficient for nodes in hierarchical clustering. The
    coefficient is defined for a node `i` as the difference of the number of
    descendents `P(i)` such that `C[j,k] == True` for all pairs `j`,`k` from
    `Q(i)`, minus the number of descendent `N(i)` such that `C[j,k] == False`
    for any `k` from `P(i)` where `j` from `N(i)`.
    
    Parameters
    ----------
    A : ndarray, optional
        Common ancestor node matrix for pairs of observations.
    C : ndarray
        Connectivity matrix for same pairs of observations.
        
    Returns
    -------
    coeffs : ndarray
        `coeffs[i]` is the connectivity coefficient for the `nodes[i]`th non-
        singleton cluster.
    nodes : ndarray
        Sorted cluster node indices at which `coeffs` was computed.
    """
    
    A = np.asarray(A)
    C = np.asarray(C, dtype=bool)
    if A.shape != C.shape:
        raise ValueError("Condensed ancestor height and connectivity matrices must have the same shape.")
        
    nodes = np.unique(A)
    coeffs = np.zeros(len(nodes), dtype=int)
    for (i, k) in enumerate(nodes):
        
        qv = np.flatnonzero((A == k).any(axis=1)) # descendents of node i
        pv = qv[greedy_clique_by_elimination(C[np.ix_(qv, qv)])]
        nv = qv[C[qv, pv].min(axis=1) < 0]
        
        coeff = len(pv) - len(nv)
        coeffs[i] = coeff
        
    return (coeffs, nodes)
    

def greedy_clique_by_elimination(C):
    """Find clique from connectivity matrix by repeatedly removing least connected
    nodes. Efficient and should generally be accurate enough for our purposes.
    
    Parameters
    ----------
    C : (N, N) ndarray
        Connectivity matrix for graph with `N` nodes.
        
    Returns
    -------
    q : ndarray
        1-D arrray of node indices of clique.
    """
    C = np.asarray(C, dtype=bool)
    n = C.shape[0]
    if C.shape[1] != n:
        raise ValueError("Connectivity matrix must be square.")
    keep = np.ones(n, dtype=bool)
    while True:
        nkeep = np.count_nonzero(keep)
        counts = np.sum(C[np.ix_(keep, keep)], axis=1)
        which_min = counts.argmin()
        if counts[which_min] == nkeep:
            break
        keep[which_min] = False
        
    return np.flatnonzero(keep)

    
def cluster_remove(Z, remove):
    """Form flat clusters from hierarchical clustering defined by linkage matrix
    `Z` .
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    remove : ndarray
        1D-array of booleans for each non-singleton clustering. `remove[i]` is
        `True` if leaf descendents of node `i` are not all contained within a
        single flat cluster. `remove` forms a monotonic array.
        
    Returns
    -------
    T : ndarray
        1D-array. `T[i]` is the flat cluster number to which original
        observation `i` belongs.
    """
    
    sp_hierarchy.fcluster(Z, 0, criterion="monocrit", monocrit=remove)

            
        
def ancestors(Z, indices, inclusive=False):
    """Compute ancestor node indices.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    indices : ndarray
        1-D array of non-singleton node indices
    inclusive : boolean, optional
        If `True` indices are counted as their own ancestors.
        
    Returns
    -------
    ancestors : ndarray
        1-D array of non-singleton node indices of the union of the sets of
        ancestors of input nodes. 
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    isancestor = np.zeros(2*n-1, dtype=bool)
    isancestor_or_index = isancestor.copy()
    isancestor_or_index[indices] = True
    for i in range(n-1):
        isancestor[i+n] = isancestor_or_index[Z[i,:2].astype(int)].any()
        isancestor_or_index[i+n] = isancestor[i+n]
        
    if inclusive:
        return np.flatnonzero(isancestor_or_index[n:])
    else:
        return np.flatnonzero(isancestor[n:])
        
    
def filter_descendents(Z, indices):
    """Find nodes that are not descendents of other nodes.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    indices : ndarray
        1-D array of cluster node indices.
        
    Returns
    -------
    nondescendents : ndarray
        1-D array containing node indices `j` such that `j` is either `i` or
        not in `Q(i)` for all `i` in `indices`, where `Q(i)` is the set of node
        indices below and including node `i`.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    indices = set(indices)
    
    outarr = []
    stack = [2*n - 1]
    while True:
        if len(stack) == 0:
            break
        i = stack.pop()
        if i < n:
            continue
        j = i-n
        if j in indices:
            outarr.append(j)
            continue
        stack.extend(Z[j,:2].astype(int))
        
    return np.array(outarr)
        

def maxcoeffs(Z, coeffs):
    """Compute the maximum coefficient of any descendent for nodes in
    hierarchical clustering.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    coeffs : ndarray
        1-D array of coefficients for each cluster node. `coeffs[i]` for `i<n`
        is the coefficient for the `i`th leaf node, and for `i>=n` is the
        coefficient for the cluster encoded by the `i-n`th row in `Z`.
        
    Returns
    -------
    maxcoeffs : ndarray
        `maxcoeffs[i]` is the maximum coefficient value of any cluster below and
        including the node with index `i`. More specifically
        `maxcoeffs[i] == coeff[Q(i)].max()` where `Q(i)` is the set of
        all nodes below and including node `i`. 
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
    
    return outarr[:n]
    
    
def height(Z):
    """Generate a condensed matrix of common ancestor node heights.
    """
    Z = np.copy(Z)
    Z[:, 2] = np.arange(Z.shape[0])
    return sp_hierachy.cophenet(Z)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
