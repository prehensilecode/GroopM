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

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
############################################################################### 
def fcluster_coeffs(Z, leaf_data, coeff_fn, return_coeffs=False, return_nodes=False):
    """Find flat clusters by maximising cluster coefficient scores for nodes a
    hierarchical clustering.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    leaf_data : dict
        Dictionary with leaf node ids as keys with values corresponding to
        lists of values to be passed to coeff_fn
    coeff_fn : function
        Function called at each node in `Z` with the concatenation of values 
        from `leaf_data` for all descendent leaves of the node, and computes
        the node coefficient.
    return_coeffs : bool
        If True, also return array of cluster coefficients.
    return_nodes : bool
        If True, also return array of flat cluster root nodes.
        
    Returns
    -------
    T : ndarray
        1-D array. `T[i]` is the flat cluster number to which original
        observation `i` belongs.
    leaf_max_coeffs : ndarray
        1-D array. `leaf_max_coeffs[i]` is the cluster coefficient for the flat
        cluster of original observation `i`. Only provided if `return_coeffs` is True.
    nodes : ndarray
        1-D array. `nodes[i]` is the cluster index corresponding to the flat cluster 
        of the `i`th original observation. Only provided if `return_nodes` is True.
        
    """
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    
    coeffs = coeffs_linkage(Z, leaf_data, coeff_fn)
    flat_ids = flatten_nodes(Z)
    coeffs[n:] = coeffs[flat_ids+n] # Map coefficient scores to descendents of equal height
    merge = maxcoeffs(Z, coeffs)[n:] == coeffs[n:]
    merge = merge[flat_ids] # Map merge value to descendents of equal height
    
    
    if not (return_nodes or return_coeffs):
        return fcluster_merge(Z, merge)
        
    (T, M) = fcluster_merge(Z,
                            merge,
                            return_nodes=True)
        
    out = (T,)
    if return_coeffs:
        out += (coeffs[M],)
    if return_nodes:
        out += (M,)
    return out
    
    
def coeffs_linkage(Z, leaf_data, coeff_fn):
    """Compute coefficients for hierarchical clustering.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    leaf_data : dict
        Dictionary with leaf node ids as keys with values corresponding to
        lists of values to be passed to coeff_fn
    coeff_fn : function
        Function called at each node in `Z` with the concatenation of values 
        from `leaf_data` for all descendent leaves of the node, and computes
        the node coefficient.
        
    Returns
    -------
    coeffs : ndarray
        `coeffs[i]` for `i<n` is defines the coefficient for the `i`th
        singleton node, and for `i>=n` is the coefficient for the cluster
        encoded by the `(i-n)`-th row in `Z`.
    """
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    
    node_data = dict()
    coeffs = np.zeros(2*n-1, dtype=int)
    
    # Compute leaf clusters
    for (i, indices) in leaf_data.iteritems():
        node_data[i] = indices
        coeffs[i] = coeff_fn(indices)
        
    # Bottom-up traversal
    for i in range(n-1):
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        current_node = n+i
        
        # update leaf cache
        try:
            left_data = node_data[left_child]
            del node_data[left_child]
        except:
            left_data = []
        try:
            right_data = node_data[right_child]
            del node_data[right_child]
        except:
            right_data = []
            
        current_data = left_data + right_data
        if current_data != []:
            node_data[current_node] = current_data
        
        # We only need to compute a new coefficient for new sets of data points, i.e. if
        # both left and right child clusters have data points.
        if left_data == []:
            coeffs[current_node] = coeffs[right_child]
        elif right_data == []:
            coeffs[current_node] = coeffs[left_child]
        else:
            coeffs[current_node] = coeff_fn(current_data)
            
    return coeffs
            

def maxcoeffs(Z, coeffs):
    """Compute the maximum coefficient of cluster nodes and their descendents.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    coeffs : ndarray
        `coeffs[i]` for `i<n` is defines the taxonomic measure value for
        the `i`th singleton node, and for `i>=n` is the value for the cluster
        encoded by the `(i-n)`-th row in `Z`.
        
    Returns
    -------
    maxcoeffs : ndarray
        `maxcoeffs[i]` is the maximum coefficient of any cluster below and 
        including cluster `i`.
    """
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    max_coeffs = np.copy(coeffs)
    
    # Bottom-up traversal
    for i in range(n-1):
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        current_node = n+i
        current_coeff = max_coeffs[current_node]
        current_max_coeff = np.max([current_coeff, max_coeffs[left_child], max_coeffs[right_child]])
        max_coeffs[current_node] = current_max_coeff
    
    return max_coeffs
    
    
def fcluster_merge(Z, merge, return_nodes=False):
    """Partition a hierarchical clustering by flattening clusters.
    
    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    merge : ndarray
        Boolean array. `merge[i]` indicates whether the cluster represented by
        `Z[i, :]` should be flattened.
    return_nodes : bool
        If True, also return array of flat cluster root nodes.
        
    Returns
    -------
    T : ndarray
        1-D array. `T[i]` is the flat cluster number to which original
        observation `i` belongs.
    nodes : ndarray
        1-D array. `nodes[i]` is the cluster index corresponding to the flat
        cluster of the `i`th original obseration. Only provided if
        `return_nodes` is True.
    """
    Z = np.asarray(Z)
    n = Z.shape[0]+1
    
    # Compute leaf clusters
    leaf_max_nodes = np.arange(n)
    leaves_dict = dict([(i, [i]) for i in range(n)])
    
    # Bottom-up traversal
    for i in range(n-1):
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        current_node = n+i
        
        # update leaf cache
        current_leaves = leaves_dict[left_child] + leaves_dict[right_child]
        del leaves_dict[left_child]
        del leaves_dict[right_child]
        leaves_dict[current_node] = current_leaves
        
        # Merge if cluster is at least as coherent taxonomically any descendent
        # cluster.
        if merge[i]:
            leaf_max_nodes[current_leaves] = n+i
    
    (_, bids) = np.unique(leaf_max_nodes, return_inverse=True)
    bids += 1 # start bin ids from 1
    
    if not return_nodes:
        return bids 
        
    out = (bids,)
    if return_nodes:
        out += (leaf_max_nodes,)
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
