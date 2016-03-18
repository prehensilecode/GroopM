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


np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class FeatureDistanceTool:
    """Computes euclidean distance between pairs of observations in feature
    spaces.
    
    Parameters
    ----------
    Xs : sequence of ndarrays
        A sequence of matrices with `m` rows corresponding to observations.
    """
    def __init__(self, Xs):
        self._Xs = tuple([np.asarray(X) for X in Xs])
        ns = [X.shape[0] for X in self._Xs]
        if np.any(ns != ns[0]):
            raise ValueError("Input arrays must have equal numbers of rows.")
        self._nobs = ns[0]
            
    def num_obs(self):
        return self._nobs
        
    def pdist(self, idx=None):
        if idx is None:
            idx = slice(None)
        return np.transpose([sp_distance.pdist(X[idx], metric="euclidean") for X in self._Xs])
        
    def cdist(self, idxA, idxB=None):
        if idxB is None:
            idxB = slice(None)
        return np.dstack([sp_distance.cdist(X[idxA], X[idxB], metric="euclidean") for X in self._Xs])


class FeatureGlobalDistanceRankTool:
    """Computes global rank of distances between pairs of observations in feature
    spaces.
    
    Parameters
    ----------
    Ys : sequence of ndarray
        A sequence of 1-D arrays of length `m` corresponding to condensed
        distance matrices in feature spaces.
    weights : ndarray, optional
        A 1-D array of length `m` containing pairwise observation weights in
        condensed form.
    """
    def __init__(self, Ys, weights=None):
        Ys = np.transpose(Ys)
        self._ranks = distance.argrank_weighted(Ys, weights=weights, axis=0)
        self._nobs = num_obs(self._ranks)
        
    def num_obs(self):
        return self._nobs
        
    def pdist(self, idx=None):
        if idx is None:
            return self._ranks
        cond_idx = pcoords(idx, self._nobs)
        return np.where(cond_idx==-1, 0, self._ranks[cond_idx])
        
    def cdist(self, idxA, idxB=None):
        if idxB is None:
            idxB = np.arange(self._nobs)
        cond_idx = ccords(idxA, idxB, self._nobs)
        return np.where(cond_idx==-1, 0, self._ranks[cond_idx])
        
        
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
    index = np.squareform(Y).sum(axis=1).argmin()

    return index

                
def pcoords(idx, n):
    (rows, cols) = np.ix_(idx, idx)
    iu = np.triu_indices(len(idx))
    return squareform_coords(rows[iu], cols[iu], n)
  
  
def ccoords(idxA, idxB, n):
    (rows, cols) = np.ix_(idxA, idxB)
    return squareform_coords(rows, cols, n)

    
def argrank(array, weights=None, axis=0):
    """Return the positions of elements of a when sorted along the specified axis"""
    if weights is None:
        return numpy.apply_along_axis(_rank_with_ties, axis, array)
    else:
        return multi_apply_along_axis(lambda (a, w): _rank_with_ties(a, w), axis, (array, weights))
 
     
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
 
    
def _rank_with_ties_(a, weights=None):
    """Return sorted of array indices with tied values averaged"""
    (codebook, codes) = np.unique(a, return_inverse=True) #codebook is in sorted order
    (n, _bins) = np.histogram(codes, bins=np.arange(codebook.size+1), weights=weights)
    w = np.concatenate(([0.], n.cumsum())).astype(np.double)
    r = (w[1:] + w[:-1] - 1) / 2
    
    return r[codes]

# Utility
def num_obs(Y):
    """Number of original observations in condensed distance matrix."""
    m = Y.shape[0]
    d = np.ceil(np.sqrt(2*m))
    if m != d * (d-1) // 2:
        raise ValueError("Argument is not a valid condensed distance matrix.")
    return d
    
    
# Utility
def multi_apply_along_axis(func1d, axis, tup, *args, **kwargs):
    """Multi-argument version of numpy's `apply_along_axis`. 
    
    Parameters
    ----------
    func1d : function
        This function should accept a tuple of 1-D arrays. It is applied to a
        tuple of 1D slices of arrays in `tup` along the specified axis.
    axis : integer
        Axis along which `tup` arrays are sliced.
    tup : tuple of ndarrays
        Tuple of input arrays. Arrays must have equal size in all dimensions
        except along the `axis` dimension.
    args : any
        Additional arguments to `func1d`.
    kwargs : any
        Additional named arguments to `func1d`
        
    Returns
    -------
    outarr : ndarray
        The output array. The shae of `outarr` is identical to the shapes of
        `tup` arrays, except along the `axis` dimension, where the length of
        `outarr` is equal to the size of the return value of `func1d`. If
        `func1d` returns a scalar `outarr` will have one fewer dimensions than
        `arr`.
    """
    tup = [np.asarray(t) for t in tup]
    ns = [t.shape[axis] for t in tup]
    a = np.concatenate(tup, axis=axis)
    edges = np.concatenate(([0, ], ns.cumsum()))
    
    def multi_func1d(arr): 
        splits = tuple([arr[s:e] for (s, e) in zip(edges[:-1], edges[1:])])
        return func1d(splits, *args, **kwargs)
        
    return numpy.apply_along_axis(multi_func1d, axis, a)
        
    
# Utility
def squareform_index(i, j, n):
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
    if not np.all(i < n) or not np.all(j < n) or np.any(i < 0) or np.any(j < 0):
        raise IndexError("Indices must be between 0 and %s-1." % n)
    return np.where(i==j, -1, n * i - i * (i + 1) // 2 + j - i -1)

    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
