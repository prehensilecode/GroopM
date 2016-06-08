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
    if axis is None:
        return _rank_with_ties(array, weights=weights)
    return np.apply_along_axis(_rank_with_ties, axis, array, weights=weights)


def density_distance(Y, weights=None, minWt=None, minPts=None):
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
    minWt : ndarray
        Total cumulative neighbour weight used to compute density distance for individual points.
    minPts : int
        Number of neighbours used to compute density distance.
        
    Returns
    -------
    density_distance : ndarray
        Condensed distance matrix of pairwise density distances.
    """
    n = sp_distance.num_obs_y(Y)
    do_weights = weights is not None
    do_pts = minPts is not None
    if (do_weights and minWt is None) or not (do_weights or do_pts):
        raise ValueError("Specify either 'weights' and 'minWt' or 'minPts' parameter values")
        
    if do_weights:
        core_dists = core_distance_weighted(Y, weights, minWt)
        
    if do_pts:
        pts_dists = core_distance(Y, minPts)
        if do_weights:
            core_dists = np.minimum(core_dists, pts_dists)
        else:
            core_dists = pts_dists
    
    (i, j) = pairs(n)
    dd = np.maximum(np.minimum(core_dists[i], core_dists[j]), Y)
    return dd
        

def core_distance_weighted(Y, weights, minWt):
    """Compute core distance for data points, defined as the distance to the furtherest
    neighbour where the cumulative weight of closer points is less than minWt.

    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix containing distances for pairs of
        observations. See scipy's `squareform` function for details.
    weights : ndarray
        Condensed matrix containing pairwise weights.
    minWt : ndarray
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
    core_dist = np.empty(n, dtype=Y.dtype)
    for i in range(n):
        minPts = int(np.sum(wm[i, sorting_indices[i]].cumsum() < minWt[i]))
        core_dist[i] = dm[i, sorting_indices[i, np.minimum(n-1, minPts)]]
    
    return core_dist
            
        
def core_distance(Y, minPts):
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
    n = sp_distance.num_obs_y(Y)
    dm = sp_distance.squareform(Y)
    dm.sort(axis=1)
    return dm[:, np.minimum(n-1, minPts)]

    
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
        1-D array of indices of original observations in traversal order.
    d : ndarray
        1-D array. `d[i]` is the `i`th traversal distance.
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
    return (o, d[o])
    
    
def condensed_index_(n, i, j):
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
    print i, j
        
        
condensed_index = np.vectorize(condensed_index_, otypes=[np.intp])
    
    
def pairs(n):
    return np.triu_indices(n, k=1)
    
    
# helper
def _rank_with_ties(a, weights=None):
    """Return sorted of array indices with tied values averaged"""
    a = np.asanyarray(a)
    shape = a.shape
    size = a.size
    
    if weights is not None:
        weights = np.asanyarray(weights)
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
