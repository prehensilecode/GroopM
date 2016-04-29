#!/usr/bin/env python
###############################################################################
#                                                                             #
#    distance.py                                                              #
#                                                                             #
#    Working with distance metrics for features                               #
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
import scipy.spatial.distance as sp_distance
from Queue import PriorityQueue

# local imports
from utils import multi_apply_along_axis

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
def mediod(Y):
    """Get member index that minimises the sum distance to other members

    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix containing distances for pairs of
        observations. See scipy's `squareform` function for details.
        
    Returns
    -------
    index : int
        Mediod observation index.
    """
    # for each member, sum of distances to other members
    index = sp_distance.squareform(Y).sum(axis=1).argmin()

    return index

    
def argrank(array, weights=None, axis=0):
    """Return the positions of elements of a when sorted along the specified axis"""
    if weights is None:
        return np.apply_along_axis(_rank_with_ties, axis, array)
    else:
        array = np.asarray(array)
        weights = np.asarray(weights)
        indexer = [None]*len(array.shape)
        indexer[axis] = slice(None)
        return multi_apply_along_axis(lambda (a, w): _rank_with_ties(a, w), axis, np.broadcast_arrays(array, weights[indexer]))

        
def density_distance(Y, weights, minWt):
    """Compute pairwise density distance, defined as the max of the pairwise
    distance between two points and the minimum core distance of the two
    points.

    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix containing distances for pairs of
        observations. See scipy's `squareform` function for details.
    weights : ndarray
        Condensed matrix containing pairwise weights.
    minWt : int or float or ndarray
        Total cumulative neighbour weight used to compute density distance for individual points.
        
    Returns
    -------
    density_distance : ndarray
        Condensed distance matrix of pairwise density distances.
    """
    n = sp_distance.num_obs_y(Y)
    core_dists = core_distance(Y, weights, minWt)
    
    inds = np.triu_indices(n, k=1)
    dd = np.maximum(np.minimum(core_dists[inds[0]], core_dists[inds[1]]), Y)
    return dd
        
        
def core_distance(Y, weights, minWt):
    """Compute core distance for data points, defined as the distance to the furtherest
    neighbour where the cumulative weight of closer points is less than minWt.

    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix containing distances for pairs of
        observations. See scipy's `squareform` function for details.
    weights : ndarray
        Condensed matrix containing pairwise weights.
    minWt : int or float or ndarray
        Total cumulative neighbour weight used to compute density distance for individual points.
        
    Returns
    -------
    core_distance : ndarray
        Core distances for data points.
    """
    n = sp_distance.num_obs_y(Y)
    dm = sp_distance.squareform(Y)
    wm = sp_distance.squareform(weights)
    sorting_indices = dm.argsort(axis=1)
    core_dist = np.empty(n, dtype=dm.dtype)
    for i in range(n):
        minPts = int(np.sum(wm[i, sorting_indices[i]].cumsum() <= minWt[i]) - 1)
        core_dist[i] = dm[i, sorting_indices[i, minPts]]
    
    return core_dist
        
        
def core_distance_(Y, minPts):
    """Compute pairwise density distance, defined as the max of the pairwise
    distance between two points and the minimum distance of the minPts
    neighbours of the two points.

    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix containing distances for pairs of
        observations. See scipy's `squareform` function for details.
    minPts : int
        Number of neighbours used to compute density distance.
        
    Returns
    -------
    core_distance : ndarray
        Core distances for observations.
    """
    dm = sp_distance.squareform(Y)
    dm.sort(axis=1)
    return dm[:, minPts]

    
def reachability_order(Y):
    """Traverse collection of nodes by choosing the closest unvisited node to
    a visited node at each step to produce a reachability plot.
    
    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix
        
    Returns
    -------
    o : ndarray
        1-D array of indices of leaf nodes in traversal order.
    """
    n = sp_distance.num_obs_y(Y)
    dm = sp_distance.squareform(Y)
    o = np.empty(n, dtype=np.intp)
    to_visit = np.ones(n, dtype=bool)
    closest = 0
    o[0] = 0
    to_visit[0] = False
    d = dm[0].copy()
    for i in range(1, n):
        closest = np.flatnonzero(to_visit)[d[to_visit].argmin()]
        o[i] = closest
        to_visit[closest] = False
        d[to_visit] = np.minimum(d[to_visit], dm[closest, to_visit])
    return (o, d)
    
    
def reachability_order_(Y):
    """Traverse collection of nodes by choosing the closest unvisited node to
    a visited node at each step to produce a reachability plot.
    
    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix
        
    Returns
    -------
    o : ndarray
        1-D array of indices of leaf nodes in traversal order.
    """
    n = sp_distance.num_obs_y(Y)
    dm = sp_distance.squareform(Y)
    o = np.empty(n, dtype=np.intp)
    d = np.empty(n, dtype=np.intp)
    closest = 0
    o[0] = 0
    d[0] = 0
    to_visit = np.arange(1, n)
    dists = dm[0, 1:]
    for i in range(1, n):
        closest_index = dists.argmin()
        closest_node = to_visit[closest_index]
        o[i] = closest_node
        d[i] = dists[closest_index]
        keep = to_visit != closest_node
        to_visit = to_visit[keep]
        dists = np.minimum(dists[keep], dm[closest_node, to_visit])
    return o
    
    
def pcoords(idx, n):
    idx = np.asarray(idx).ravel()
    (iu, ju) = np.triu_indices(idx.size, k=1)
    return condensed_index_(n, idx[iu], idx[ju])
  
  
def ccoords(idxA, idxB, n):
    idxA = np.asarray(idxA).ravel()
    idxB = np.asarray(idxB).ravel()
    (rows, cols) = np.ix_(idxA, idxB)
    return condensed_index_(n, rows, cols)
    
    
def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    Based on scipy Cython function:
    https://github.com/scipy/scipy/blob/v0.17.0/scipy/cluster/_hierarchy.pyx
    """
    if i < j:
        return n * i - (i * (i + 1) // 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) // 2) + (i - j - 1)
    
  
def condensed_index_(n, i, j):
    """Returns indices in a condensed matrix corresponding to input
    coordinates, or -1 for diagonal elements.
    
    From the numpy documentation for `squareform` the element of the condensed
    matrix `v` containing the distance between points `i` and `j` is
    `{n choose 2} - {n-i choose 2} + (j-i-1)`.
    
    Equivalently, the starting element for the `i`th row is the number of
    elements in the `i` higher rows minus the number of non-upper diagonal
    elements of the top left `i+1`-by-`i+1` matrix. The element index can be
    computed using:
        ``index of first element of row i = n*i - i*(i+1)/2``
    
    E.g. at row 3 in the matrix below, there are
    ``3*n (x's and +'s) - 6 x's = 3*n-6 +'s``.
    
    x + + + + ...
    x x + + + ...
    x x x + + ...
    - - - - i ...
    
    The corresponding element for the `j`th column is found by incrementing by
    the distance from the centre diagonal (the `i+1`th column) to the `j`th
    column. The element index can be computed using:
        ``index of column j = index of first element of row i + j - i - 1``
    
    E.g. at row 2 and column 5 in the matrix below, the difference between the
    element `j` and `i` is ``5 (column of `j`) - (2+1) (column of `i`) = 2``.
    
    x + + + + + ...
    x x + + + + ...
    - - - i + j ...
    """
    
    # must have row < col
    ii = np.where(j < i, j, i)
    jj = np.where(i > j, i, j)
    #return np.where(ii==jj, -1, n*(n-1)//2 - (n-ii)*(n-ii-1)//2 + jj-ii-1)
    return n*ii - ii*(ii+1)//2 + jj-ii-1
    
      
# helper
def _rank_with_ties(a, weights=None):
    """Return sorted of array indices with tied values averaged"""
    a = np.asarray(a)
    shape = a.shape
    size = a.size
    
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != shape:
            raise ValueError('weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    
    sorting_index = a.argsort()
    sa = a[sorting_index]
    flag = np.concatenate(([True], sa[1:] != sa[:-1], [True]))
    if weights is None:
        # counts up to 
        cw = np.flatnonzero(flag)
    else:
        cw = np.concatenate(([0], weights[sorting_index].cumsum()))
        cw = cw[flag]
    cw = cw.astype(np.double)
    sr = (cw[1:] + cw[:-1] - 1) / 2
    
    iflag = np.cumsum(flag[:-1]) - 1
    r = np.empty(size, dtype=np.double)
    r[sorting_index] = sr[iflag]
    r = r.reshape(shape)
    return r

    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
