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
from classification import ClassificationManager

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
        
        mct = HierarchyCliqueFinder(Z, t, self._mapping)
        mnodes = mct.nodes
        #Size of maximal clique of `Q(k)`, minus number of non-clique elements of `Q(k)`
        mcc = np.array([2*len(mct.maxClique(i)) - len(i) for i in (mct.indices(k) for k in mnodes)])
        
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
    # descendents. These are nodes where the coefficient maximum is
    # non-decreasing along all leaf-to-root paths. 
    maxcc = maxcoeffs(Z, coeffs)
    maxinds = np.flatnonzero(maxcc == coeffs)
    
    # The second time, we descend from the root until a node identified in
    # the first pass or a leaf node is encountered. These are nodes where
    # the coefficient is greatest along any root-to-leaf path. 
    maxinds = filter_descendents(Z, maxinds)
    
    print "maxcc=", maxcc[maxinds]
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

    
class HierarchyCliqueFinder:
    """Find fully connected subsets of descendents for nodes in hierarchical clustering.
    
    Parameters
    ----------
    Z: ndarray
        Linkage matrix encoding hierarchical clustering.
    markers: Markers instance
        See ProfileManager class documentation
    """
    def __init__(self, Z, t, markers):
        Z = np.asarray(Z)
        n = Z.shape[0] + 1
        H = height(Z)
        
        indices = np.asarray(markers.rowIndices)
        idx = distance.ccoords(indices, np.arange(n), n)
        self._mA = np.where(idx==-1, indices[:, None]*(idx==-1), H[idx]+n)
        cm = ClassificationManager(markers)
        self._mC = cm.makeConnectivity(t)
        """Nodes of Z that correspond to embedded hierarchy nodes."""
        self.nodes = np.unique(self._mA[:, indices])
        
    def indices(self, node):
        """Computes the original observations composing a node."""
        return np.flatnonzero(np.any(self._mA==node, axis=1))
        
    def maxClique(self, indices):
        """Compute a maximal set `P(i)` of indices j such that `C[j,k] == True`
        for all pairs `j`,`k` from `Q(i)"""
        if len(indices) == 0:
            return np.array([], dtype=np.intp)
        return greedy_clique_by_elimination(self._mC[np.ix_(indices, indices)])
        

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
        if nkeep==0:
            break
        counts = np.sum(C[np.ix_(keep, keep)], axis=1)
        which_min = counts.argmin()
        if counts[which_min] == nkeep:
            break
        keep[keep] = np.arange(nkeep)!=which_min
        
    return np.flatnonzero(keep)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
