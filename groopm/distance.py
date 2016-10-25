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
    
    
def iargrank(out, weights=None, axis=0):
    """Return the positions of elements of a when sorted along the specified axis"""
    if axis is None:
        _irank_with_ties(out, weights=weights)
        return
    np.apply_along_axis(_irank_with_ties, axis, out, weights=weights)

def density_distance_(Y, weights=None, minWt=None, minPts=None):
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
    do_weights = minWt is not None
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
        del pts_dists # mem opt
    
    (dists_i, dists_j) = tuple(core_dists[i] for i in pairs(n))
    dd = np.maximum(np.minimum(dists_i, dists_j), Y)
    return dd
    
def idensity_distance_(out, weights=None, minWt=None, minPts=None):
    """Compute pairwise density distance, defined as the max of the pairwise
    distance between two points and the minimum core distance of the two
    points.

    Parameters
    ----------
    out : ndarray
        Condensed distance matrix containing distances for pairs of
        observations. See scipy's `squareform` function for details.
    weights : ndarray
        Condensed matrix containing pairwise weights.
    minWt : ndarray
        Total cumulative neighbour weight used to compute density distance for individual points.
    minPts : int
        Number of neighbours used to compute density distance.
    """
    n = sp_distance.num_obs_y(out)
    do_weights = minWt is not None
    do_pts = minPts is not None
    if (do_weights and minWt is None) or not (do_weights or do_pts):
        raise ValueError("Specify either 'weights' and 'minWt' or 'minPts' parameter values")
    
    if do_weights:
        core_dists = core_distance_weighted(out, weights, minWt)
        
    if do_pts:
        pts_dists = core_distance(out, minPts)
        if do_weights:
            core_dists = np.minimum(core_dists, pts_dists)
        else:
            core_dists = pts_dists
        del pts_dists # mem opt
    
    m = np.empty(n, dtype=out.dtype)
    k = 0
    for i in range(n-1):
        out[k:k+n-1-i] = np.maximum(np.minimum(core_dists[i], core_dists[(i+1):n]), out[k:k+n-1-i])
        k = k+n-1-i
    #(dists_i, dists_j) = tuple(core_dists[i] for i in pairs(n))
    #dd = np.maximum(np.minimum(dists_i, dists_j), out)
    #assert np.all(dd==out)
    return out

def core_distance(Y, weights=None, minWt=None, minPts=None):
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
    minPts : int
        Number of neighbours used to compute density distance.
        
    Returns
    -------
    core_distance : ndarray
        Core distances for data points.
    """
    n = sp_distance.num_obs_y(Y)
    #dm_ = sp_distance.squareform(Y)
    core_dist = np.empty(n, dtype=Y.dtype)
    m = np.empty(n, dtype=Y.dtype) # store row distances
    minPts = n-1 if minPts is None else minPts
    if weights is None or minWt is None:
        #dm_.sort(axis=1)
        #x_ = dm_[:, np.minimum(n-1, minPts)]
        for (i, mp) in np.broadcast(np.arange(n), minPts):
            others = np.flatnonzero(np.arange(n)!=i)
            m[others] = Y[condensed_index(n, i, others)]
            m[i] = 0
            m.sort()
            #assert np.all(dm_[i] == m)
            core_dist[i] = m[np.minimum(n-1, mp)]
            #assert x_[i] == core_dist[i]
    else:
        #wm_ = sp_distance.squareform(weights)
        w = np.empty(n, dtype=weights.dtype) # store row weights
        for (i, mp, mw) in np.broadcast(np.arange(n), minPts, minWt):
            others = np.flatnonzero(np.arange(n)!=i)
            m[others] = Y[condensed_index(n, i, others)]
            m[i] = 0
            #assert np.all(m==dm_[i])
            w[others] = weights[condensed_index(n, i, others)]
            w[i] = 0
            #assert np.all(w==wm_[i])
            sorting_indices = m.argsort()
            minPts = np.minimum(int(np.sum(w[sorting_indices].cumsum() < mw)), mp)
            core_dist[i] = m[sorting_indices[np.minimum(n-1, minPts)]]
            #minPts_ = int(np.sum(wm_[i, sorting_indices].cumsum() < minWt[i]))
            #m_ = m[sorting_indices]
            #assert core_dist[i] == np.minimum(m_[np.minimum(n-1, mp)], m_[np.minimum(n-1, minPts_)])
    return core_dist
    
def core_distance_weighted_(Y, weights, minWt):
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
    #dm_ = sp_distance.squareform(Y)
    #wm_ = sp_distance.squareform(weights)
    core_dist = np.empty(n, dtype=Y.dtype)
    m = np.empty(n, dtype=Y.dtype) # store row distances
    w = np.empty(n, dtype=weights.dtype) # store row weights
    for i in range(n):
        others = np.flatnonzero(np.arange(n)!=i)
        m[others] = Y[condensed_index(n, i, others)]
        m[i] = 0
        #assert np.all(m==dm_[i])
        w[others] = weights[condensed_index(n, i, others)]
        w[i] = 0
        #assert np.all(w==wm_[i])
        sorting_indices = m.argsort()
        minPts = int(np.sum(w[sorting_indices].cumsum() < minWt[i]))
        #assert minPts == int(np.sum(wm_[i, sorting_indices].cumsum() < minWt[i]))
        core_dist[i] = m[sorting_indices[np.minimum(n-1, minPts)]]
        #assert core_dist[i] == dm_[i, sorting_indices[np.minimum(n-1, minPts)]]
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
    n = sp_distance.num_obs_y(Y)
    #dm_ = sp_distance.squareform(Y)
    #dm_.sort(axis=1)
    #x_ = dm_[:, np.minimum(n-1, minPts)]
    core_dist = np.empty(n, dtype=Y.dtype)
    m = np.empty(n, dtype=Y.dtype)
    for i in range(n):
        others = np.flatnonzero(np.arange(n)!=i)
        m[others] = Y[condensed_index(n, i, others)]
        m[i] = 0
        m.sort()
        #assert np.all(dm_[i] == m)
        core_dist[i] = m[np.minimum(n-1, minPts)]
        #assert x_[i] == core_dist[i]
    return core_dist
   
def reachability_order(Y, core_dist=None):
    """Traverse collection of nodes by choosing the closest unvisited node to
    a visited node at each step to produce a reachability plot.
    
    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix
    core_dist : ndarray
        Core distances for original observations of Y.
        
    Returns
    -------
    o : ndarray
        1-D array of indices of original observations in traversal order.
    d : ndarray
        1-D array. `d[i]` is the `i`th traversal distance.
    """
    n = sp_distance.num_obs_y(Y)
    #dm_ = sp_distance.squareform(Y)
    o = np.empty(n, dtype=np.intp)
    to_visit = np.ones(n, dtype=bool)
    closest = 0
    o[0] = 0
    to_visit[0] = False
    d = np.empty(n, dtype=Y.dtype)
    d[0] = 0
    d[1:] = np.maximum(Y[condensed_index(n, 0, np.arange(1, n))], core_dist[0])
    #assert np.all(d== dm_[0])
    for i in range(1, n):
        closest = np.flatnonzero(to_visit)[d[to_visit].argmin()]
        o[i] = closest
        to_visit[closest] = False
        m = np.maximum(Y[condensed_index(n, closest, np.flatnonzero(to_visit))], core_dist[closest])
        #assert np.all(m==dm_[closest, to_visit])
        d[to_visit] = np.minimum(d[to_visit], m)
    return (o, d[o])
    
    
def _condensed_index(n, i, j):
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
        
        
condensed_index = np.vectorize(_condensed_index, otypes=[np.intp])
    
    
def pairs(n):
    return np.triu_indices(n, k=1)
    
    
# helpers
def _rank_with_ties(a, weights=None):
    """Return sorted of array indices with tied values averaged"""
    a = np.asanyarray(a)
    size = a.size
    if a.shape != (size,):
        raise ValueError("a should be a 1-D array.")
    
    if weights is not None:
        weights = np.asanyarray(weights)
        if weights.shape != (size,):
            raise ValueError('weights should have the same shape as a.')
    
    sorting_index = a.argsort()
    sa = a[sorting_index]
    flag = np.concatenate(([True], sa[1:] != sa[:-1], [True]))
    del sa # optimise memory usage
    if weights is None:
        # counts up to 
        cw = np.flatnonzero(flag).astype(float)
    else:
        cw = np.concatenate(([0.], weights[sorting_index].cumsum())).astype(float)
        cw = cw[flag]
    iflag = np.cumsum(flag[:-1]) - 1
    del flag # mem optimisation
    sr = (cw[1:] + cw[:-1] - 1) * 0.5
    sr = sr[iflag]
    del iflag, cw # mem optimisation
    
    r = np.empty(size, dtype=np.double)
    r[sorting_index] = sr
    return r
    
def _irank_with_ties(a, weights=None):
    """Inplace-ish sorted of array indices with tied values averaged"""
    a = np.asanyarray(a)
    size = a.size
    if a.shape != (size,):
        raise ValueError("a should be a 1-D array.")
    
    if weights is not None:
        weights = np.asanyarray(weights)
        if weights.shape != (size,):
            raise ValueError('weights should have the same shape as a.')
    
    sorting_index = a.argsort() # copy!
    #a_ = a[sorting_index]
    a[:] = a[sorting_index] # sort a
    #assert np.all(a_==a)
    flag = np.empty(size+1, dtype=bool)
    flag[0] = flag[-1] = True
    if size > 1:
        flag[1:-1] = a[1:] != a[:-1]
    nnz = np.count_nonzero(flag)
    cw_first = 0 # initial cumulative sorted weight
    if weights is None:
        # counts up to 
        cw_rest = a[:nnz-1] # reserve part of buffer for rest of cumulative sorted weights
        cw_rest[:] = np.flatnonzero(flag)[1:]
        # a invalid
        #cw_ = np.flatnonzero(flag).astype(float)
    else:
        a[:] = weights[sorting_index]  # write sorted weights into buffer
        a[:] = a.cumsum()
        cw_rest = a[:nnz-1]
        cw_rest[:] = a[flag[1:]]
        # a invalid
        #cw_ = np.concatenate(([0.], weights[sorting_index].cumsum())).astype(float)
        #cw_ = cw_[flag]
    #assert cw_[0]==cw_first and np.all(cw_[1:]==cw_rest)
    
    sr = cw_rest
    if len(cw_rest) > 1:
        sr[1:] = cw_rest[1:] + cw_rest[:-1]
        sr[1:] -= 1
        sr[1:] *= 0.5
    sr[0] = (cw_rest[0] + cw_first - 1) * 0.5
    del cw_rest # cw_rest invalid
    
    #sr_ = (cw_[1:] + cw_[:-1] - 1) * 0.5
    #assert np.all(sr==sr_)
    
    iflag = np.cumsum(flag[:-1]) - 1 # another copy !
    a[sorting_index] = sr[iflag]
    
    #r_ = np.empty(size, dtype=np.double)
    #r_[sorting_index] = sr_[iflag]
    #assert np.all(r_==a)
    
    return True

    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
