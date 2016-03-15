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


np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def fcluster(Z, remove):
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


def ismaxcoeff_ancestor(Z, coeffs, H=None):
    """Compute whether nodes in hierarchical clustering encoded as linkage matrix
    `Z` have a coefficient greater than any ancestor node.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    coeffs : ndarray
        1D-array of coefficients for each non-singleton clustering. `coeffs[i]`
        is the coefficient for the cluster encoded by row `Z[i,:]`.
    H : ndarrray, optional
        Condensed common ancestor height matrix for pairs of original
        observations. Computed from Z if missing.
        
    Returns
    -------
    ismaxcoeff : ndarray
        1D-array of booleans. `ismaxcoeff[i]` indicates if non-singleton node `i`
        has `coeff[i] > coeff[A(i)].max()` where `A(i)` is the set of ancestors
        of `i`. 
    """
    
    Z = np.asarray(Z)
    coeffs = np.asarray(coeffs)
    if Z.shape[0] != coeffs.shape[0]:
        raise ValueError("Number of coefficients must be sams as number rows of"
                         "linkage matrix")
                         
    n = Z.shape[0] + 1
    
    # Algorithm traverses the cluster hierarchy twice.
    
    # The first time, we find nodes where the node coefficient is non-negative 
    # and equal to the maximum of all non-negative descendents. These are nodes
    # where the coefficient maximum is non-decreasing along all leaf-to-root
    # paths. 
    Zcoeff = Z.copy()
    Zcoeff[:, 2] = np.where(coeffs > 0, coeffs, 0) # linkage distances must be positive
    ismaxcoeff_of_descendents = coeffs==sp_hierarchy.maxdists(Zcoeff)
    
    
    # The second time, we descend from the root until a node identified in the first
    # pass is encountered
    ismaxcoeff_ancestor = np.zeros(n-1, dtype=bool)
    stack = [2*n-1]
    while len(stack) > 0:
        i = stack.pop()
        if i < n or maxcoeff_of_descendents[i-n]:
            continue
        
        ismaxcoeff_ancestor[i-n] = True
        stack.extend(Z[i-n, :1])
        
    return ismaxcoeff_ancestor


def connectivity_coeffs(H, C):
    """Find connectivity coefficient for nodes in hierarchical clustering. The
    coefficient is defined for a node `i` as the difference of the number of
    descendents `P(i)` such that `C[j,k] == True` for all pairs `j`,`k` from
    `Q(i)`, minus the number of descendent `N(i)` such that `C[j,k] == False`
    for any `k` from `P(i)` where `j` from `N(i)`.
    
    Parameters
    ----------
    H : ndarray, optional
        Condensed common ancestor height matrix for pairs of original
        observations.
    C : ndarray
        Condensed connectivity matrix for pairs of original observations.
        
    Returns
    -------
    coeffs : ndarray
        `coeffs[i]` is the connectivity coefficient for the `i`th 
        non-singleton cluster.
    """
    
    H = np.asarray(H)
    C = np.asarray(C, dtype=bool)
    if H.size != C.size:
        raise ValueError("Condensed ancestor height and connectivity matrices must be the same size.")
        
    H = sp_distance.squareform(H+1) #we want height 0 to correspond to singleton clusters
    C = sp_distance.squareform(C)
    
    n = H.shape[0]
    
    coeffs = np.zeros(n-1)
    for i in range(n-1):
        
        Q = np.flatnonzero((H == i+1).any(axis=1)) # descendents of node i
        P = Q[greedy_clique_by_elimination(C[np.ix_(Q, Q)])]
        N = Q[C[Q, P].min(axis=1) < 0]
        
        coeff = len(P) - len(N)
        coeffs[k] = coeff
        
    return coeffs
    

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
        1D-arrray of node indices of clique.
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
