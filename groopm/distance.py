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
import scipy.stats as sp_stats
import scipy.misc as sp_misc

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
    
    
def argrank(array, weight_fun=None, axis=0):
    """Return fractional ranks of elements of a when sorted along the specified axis"""
    if axis is None:
        return _fractional_rank(array, weight_fun=weight_fun)
    return np.apply_along_axis(_fractional_rank, axis, array, weight_fun=weight_fun)
    
    
def iargrank(out, weight_fun=None):
    """Replace elements with the fractional ranks when sorted"""
    _ifractional_rank(out, weight_fun=weight_fun)

    
def core_distance(Y, weight_fun=None, minWt=None, minPts=None):
    """Compute core distance for data points, defined as the distance to the furtherest
    neighbour where the cumulative weight of closer points is less than minWt.

    Parameters
    ----------
    Y : ndarray
        Condensed distance matrix containing distances for pairs of
        observations. See scipy's `squareform` function for details.
    weight_fun : ndarray
        Function to calculate pairwise weights for condensed distances.
    minWt : ndarray
        Total cumulative neighbour weight used to compute density distance for individual points.
    minPts : int
        Number of neighbours used to compute density distance.
        
    Returns
    -------
    core_distance : ndarray
        Core distances for data points.
    """
    (Y, _) = validate_y(Y, name="Y")
    n = sp_distance.num_obs_y(Y)
    core_dist = np.empty(n, dtype=Y.dtype)
    m = np.empty(n, dtype=Y.dtype) # store row distances
    minPts = n-1 if minPts is None else minPts
    if weight_fun is None or minWt is None:
        for (i, mp) in np.broadcast(np.arange(n), minPts):
            others = np.flatnonzero(np.arange(n)!=i)
            m[others] = Y[condensed_index(n, i, others)]
            m[i] = 0
            m.sort()
            core_dist[i] = m[np.minimum(n-1, mp)]
    else:
        w = np.empty(n, dtype=np.double) # store row weights
        for (i, mp, mw) in np.broadcast(np.arange(n), minPts, minWt):
            others = np.flatnonzero(np.arange(n)!=i)
            m[others] = Y[condensed_index(n, i, others)]
            m[i] = 0
            w[others] = weight_fun(i, others)
            w[i] = 0
            sorting_indices = m.argsort()
            minPts = np.minimum(int(np.sum(w[sorting_indices].cumsum() < mw)), mp)
            core_dist[i] = m[sorting_indices[np.minimum(n-1, minPts)]]
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
    Y = np.asanyarray(Y)
    n = sp_distance.num_obs_y(Y)
    if core_dist is not None:
        core_dist = np.asarray(core_dist)
        if core_dist.shape != (n,):
            raise ValueError("core_dist is not a 1-D array with compatible size to Y.")
    o = np.empty(n, dtype=np.intp)
    to_visit = np.ones(n, dtype=bool)
    closest = 0
    o[0] = 0
    to_visit[0] = False
    d = np.empty(n, dtype=Y.dtype)
    d[0] = 0
    d[1:] = Y[condensed_index(n, 0, np.arange(1, n))]
    if core_dist is not None:
        d = np.maximum(d, core_dist[0])
    for i in range(1, n):
        closest = np.flatnonzero(to_visit)[d[to_visit].argmin()]
        o[i] = closest
        to_visit[closest] = False
        m = Y[condensed_index(n, closest, np.flatnonzero(to_visit))]
        if core_dist is not None:
            m = np.maximum(m, core_dist[closest])
        d[to_visit] = np.minimum(d[to_visit], m)
    return (o, d[o])
    

def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    Based on scipy Cython function:
    https://github.com/scipy/scipy/blob/v0.17.0/scipy/cluster/_hierarchy.pyx
    """
    return np.where(i < j,
                    n*i - (i * (i + 1) // 2) + (j - i - 1),
                    n*j - (j * (j + 1) // 2) + (i - j - 1)
                   )
    
    
def squareform_coords(n, k):
    """
    Calculate the coordinates (i, j), i < j of condensed index k in full
    n x n distance matrix.
    """
    n = np.asarray(n)
    k = np.asarray(k)
    
    # i = np.floor(0.5*(2*n - 1 - np.sqrt((2*n - 1)**2 - 8*k)))
    i = -8.*k
    i += (2*n - 1)**2
    i **= 0.5
    i *= -1
    i += 2*n - 1
    i *= 0.5
    i = np.floor(i).astype(np.int)
    
    # j = k + i - (n * i - (i * (i + 1)) // 2 - 1)
    j = i + 1
    j *= i
    j //= 2
    j *= -1
    j += n*i - 1
    j *= -1
    j += i
    j += k
    j = np.asarray(j, dtype=np.int)*1
    return (i, j)
    
    
def pairs(n):
    return np.triu_indices(n, k=1)
    
    
# helpers
def _fractional_rank(ar, weight_fun=None):
    """
    Return sorted of array indices with tied values averaged.
    
    Code is loosely based on numpy's unique function:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/arraysetops.py#L112-L232
    """
    (ar, _) = validate_y(ar, name="ar")
    size = ar.size
    perm = ar.argsort()
    
    aux = ar[perm] # sorted ar
    # flag ends of streaks of consecutive equal values (numpy unique tracks
    # start of streaks)
    flag = np.concatenate((aux[1:] != aux[:-1], [True]))
    
    if weight_fun is None:
        # ranks of indices at ends of streaks
        rflag = np.flatnonzero(flag).astype(float)+1
    else:
        # cumulative weights of sorted values at ends of streaks
        rflag = weight_fun(perm).cumsum().astype(float)
        rflag = rflag[flag]
    # calculate an average rank for equal value streaks by averaging streak
    # start and end ranks
    rflag = np.concatenate((rflag[:1] - 1, rflag[1:] + rflag[:-1] - 1)) * 0.5
    
    # streak index / rank corresponding to sorted original values
    iflag = np.concatenate(([0.], np.cumsum(flag[:-1]))).astype(int)
    # put points back in original order
    r = np.empty(size, dtype=np.double)
    r[perm] = rflag[iflag]
    return r

    
def _ifractional_rank(ar, weight_fun=None):
    """
    Array value ranks with tied values averaged
    
    Optimised rank algortihm that reuses and mutates input array storage
    """
    (ar, _) = validate_y(ar, name="ar")
    size = ar.size
    out = ar # we will eventually write ranks to input array
    
    perm = ar.argsort() # <- copy
    ar[:] = ar[perm] # sort ar
    
    # identity indices of final values of streaks of consecutive equal values
    flag = np.concatenate((ar[1:] != ar[:-1], [True]))
    count = np.count_nonzero(flag) # number of uniques
    
    # create a buffer using ar storage
    buff = np.getbuffer(ar)
    del ar # ar invalid
    
    if weight_fun is None:
        # ranks of indices at ends of streaks
        rflag = np.frombuffer(buff, dtype=np.double, count=count) # reserve part of buffer for rest of cumulative sorted weights
        rflag[:] = np.flatnonzero(flag)+1
    else:
        wts = np.frombuffer(buff, dtype=np.double)
        wts[:] = weight_fun(perm) # write sorted weights into buffer
        wts[:] = wts.cumsum() # write cumulative weights into buffer
        rflag = np.frombuffer(buff, dtype=np.double, count=count)
        rflag[:] = wts[flag] # ranks of indices at ends of streaks
        del wts # cw invalid
    
    # calculate an average rank for equal value streaks by averaging streak
    # start and end ranks
    if len(rflag) > 1:
        rflag[1:] = rflag[1:] + rflag[:-1]
        rflag[1:] -= 1
        rflag[1:] *= 0.5
    rflag[0] = (rflag[0] - 1) * 0.5
    
    iflag = np.cumsum(flag[:-1]) # <- copy
    del flag # mem_opt
    top = rflag[0] # get this value first, as r and out share a buffer, and writing to out will overwrite r 
    out[perm[1:]] = rflag[iflag]
    out[perm[0]] = top
    

def validate_y(Y, weights=None, name="Y"):
    Y = np.asanyarray(Y)
    size = Y.size
    if Y.shape != (size,):
        raise ValueError("%s should be a 1-D array." % name)
    
    if weights is not None:
        weights = np.asanyarray(weights)
        if weights.shape != (size,):
            raise ValueError("weights should have the same shape as %s." % name)
    return (Y, weights)
    

def logratio(X, axis=-1, mode="centered"):
    X = np.asanyarray(X)
    if mode=="additive":
        fn = lambda log_x: log_x[:-1] - log_x[-1]
    elif mode=="centered":
        fn = lambda log_x: log_x - np.mean(log_x)
    elif mode=="isometric":
        n = X.shape[axis]
        fn = lambda log_x: (np.cumsum(log_x[:-1]) - np.arange(1,n)*log_x[1:]) / np.sqrt(np.arange(1,n)*np.arange(2,n+1))
    return np.apply_along_axis(fn, axis, np.log(X))
        
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
