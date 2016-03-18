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
