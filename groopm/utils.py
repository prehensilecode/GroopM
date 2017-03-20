#!/usr/bin/env python
###############################################################################
#                                                                             #
#    utils.py                                                                 #
#                                                                             #
#    Utility classes                                                          #
#                                                                             #
#    Copyright (C) Michael Imelfort, Tim Lamberton                            #
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

__author__ = "Michael Imelfort, Tim Lamberton"
__copyright__ = "Copyright 2012-2015"
__credits__ = ["Michael Imelfort", "Tim Lamberton"]
__license__ = "GPL3"
__version__ = "0.2.11"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"
__status__ = "Development"

###############################################################################
import os
import sys
import errno
import numpy as np

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
class CSVReader:
    """Read tabular data from text files"""
    def readCSV(self, fp, separator):
        for l in fp:
            yield l.rstrip().split(separator)
            
            
class FastaReader:
    """Read in fasta files"""
    def readFasta(self, fp): # this is a generator function
        header = None
        seq = None
        for l in fp:
            if l[0] == '>': # fasta header line
                if header is not None:
                    # we have reached a new sequence
                    yield header, "".join(seq)
                header = l.rstrip()[1:].partition(" ")[0] # save the header we just saw
                seq = []
            else:
                seq.append(l.rstrip())
        # anything left in the barrel?
        if header is not None:
            yield header, "".join(seq)

            
def makeSurePathExists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
            
            
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
    #tup = tuple(np.asarray(t) for t in tup)
    ns = np.array([np.shape(t)[axis] for t in tup])
    a = np.concatenate(tup, axis=axis)
    edges = np.concatenate(([0], ns.cumsum()))
    
    def multi_func1d(arr): 
        splits = tuple([arr[s:e] for (s, e) in zip(edges[:-1], edges[1:])])
        return func1d(splits, *args, **kwargs)
        
    return np.apply_along_axis(multi_func1d, axis, a)
    
    
def group_iterator(grouping):
    """Returns an iterator of values and indices for a grouping variable."""
    group_dist = {}
    for (i, name) in enumerate(grouping):
        try:
            group_dist[name].append(i)
        except KeyError:
            group_dist[name] = [i]
    
    return group_dist.iteritems()
    
    
def split_contiguous(grouping, filter_groups=[]):
    """Find initial and final indices"""
    flag_first = np.concatenate(([True], grouping[1:] != grouping[:-1]))
    first_indices = np.flatnonzero(flag_first)
    last_indices = np.concatenate((first_indices[1:], [len(grouping)]))
    keep = np.in1d(grouping, filter_groups, invert=True)
    return (first_indices[keep[first_indices]], last_indices[keep[first_indices]])
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
