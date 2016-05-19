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
from classification import ClassificationConsensusFinder

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
############################################################################### 
def fcluster_classification(Z, markers, level, return_coeffs=False, return_nodes=False):
    """Run the process to determine flat clusters"""
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    
    (coeffs, num_markers) = consensus_coefficients(Z, markers, level)
    ids = flatten_nodes(Z)
    coeffs[n:] = coeffs[ids+n]
    return fcluster_coeffs(Z, coeffs, num_markers, return_coeffs=return_coeffs, return_nodes=return_nodes)


def consensus_coefficients(Z, markers, level):
    """Compute measure of taxonomic coherence for hierarchical clustering.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    markers : Mapping object
        See ProfileManager.py.
    level : int
        Taxonomic level at which to define clusters.
        
    Returns
    -------
    coeffs : ndarray
        `coeffs[i]` for `i<n` is defines the taxonomic measure value for
        the `i`th singleton node, and for `i>=n` is the value for the cluster
        encoded by the `(i-n)`-th row in `Z`.
    num_markers : ndarray
        `num_markers[i]` is the number of taxonomic assignments made to contigs
        in cluster `i`, where `i<n` corresponds to singleton clusters.
    """
    cfinder = ClassificationConsensusFinder(markers, level)
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    
    markers_dict = dict(markers.iterindices())
    num_markers = np.zeros(2*n-1, dtype=int)
    for (i, indices) in markers_dict.iteritems():
        num_markers[i] = len(indices)
    coeffs = np.zeros(2*n-1, dtype=int)
        
    # Bottom-up traversal
    for i in range(n-1):
        left_child = int(Z[i, 0])
        left_num_markers = num_markers[left_child]
        right_child = int(Z[i, 1])
        right_num_markers = num_markers[right_child]
        current_node = n+i
        current_num_markers = left_num_markers + right_num_markers
        num_markers[current_node] = current_num_markers
        
        # update leaf cache
        try:
            current_markers = markers_dict[left_child]
            del markers_dict[left_child]
        except:
            current_markers = []
        try:
            current_markers += markers_dict[right_child]
            del markers_dict[right_child]
        except:
            pass
        markers_dict[current_node] = current_markers
        
        # We only need to compute a new coefficient for new sets of markers, i.e. if
        # both left and right child clusters have markers.
        if left_num_markers == 0:
            coeffs[current_node] = coeffs[right_child]
        elif right_num_markers == 0:
            coeffs[current_node] = coeffs[left_child]
        else:
            coeffs[current_node] = cfinder.disagreement(current_markers)

    return (coeffs, num_markers)
            

def fcluster_coeffs(Z, coeffs, num_markers, return_coeffs=False, return_nodes=False):
    """Partition a hierarchical clustering by identifying clusters that
    maximise a measure of taxonomic coherence.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    coeffs : ndarray
        `coeffs[i]` for `i<n` is defines the taxonomic measure value for
        the `i`th singleton node, and for `i>=n` is the value for the cluster
        encoded by the `(i-n)`-th row in `Z`.
    num_markers : ndarray
        `num_markers[i]` is the number of taxonomic assignments made to contigs
        in cluster `i`, where `i<n` corresponds to singleton clusters.
        
    Returns
    -------
    T : ndarray
        1-D array. `T[i]` is the flat cluster number to which original
        observation `i` belongs.
    """
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    num_markers = np.asarray(num_markers)
    max_coeffs = np.copy(coeffs)
    
    if return_coeffs:
        leaf_max_coeffs = np.zeros(n, dtype=int)
    leaf_max_nodes = np.arange(n)
    leaves_dict = dict([(i, [i]) for i in range(n)])
    
    # Bottom-up traversal
    for i in range(n-1):
        left_child = int(Z[i, 0])
        left_num_markers = num_markers[left_child]
        left_max_coeff = max_coeffs[left_child]
        right_child = int(Z[i, 1])
        right_num_markers = num_markers[right_child]
        right_max_coeff = max_coeffs[right_child]
        current_node = n+i
        current_coeff = max_coeffs[current_node]
        current_max_coeff = np.max([current_coeff, left_max_coeff, right_max_coeff])
        max_coeffs[current_node] = current_max_coeff
        
        # update leaf cache
        current_leaves = leaves_dict[left_child] + leaves_dict[right_child]
        del leaves_dict[left_child]
        del leaves_dict[right_child]
        leaves_dict[current_node] = current_leaves
        
        # Merge if cluster is at least as coherent taxonomically any descendent
        # cluster.
        if current_coeff == current_max_coeff:
            if return_coeffs:
                leaf_max_coeffs[current_leaves] = current_max_coeff
            leaf_max_nodes[current_leaves] = n+i
    
    (nodes, bids) = np.unique(leaf_max_nodes, return_inverse=True)
    bids += 1 # start bin ids from 1
    
    if not (return_coeffs or return_nodes):
        return bids 
        
    out = (bids,)
    if return_coeffs:
        out += (leaf_max_coeffs,)
    if return_nodes:
        out += (nodes,)
    return out
    
     
def flatten_nodes(Z):
    """Collapse nested clusters of equal height by mapping descendent nodes to their earliest
    equal height ancestor
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
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
