#!/usr/bin/env python
###############################################################################
#                                                                             #
#    stream.py                                                                #
#                                                                             #
#    Streaming version of GroopM operations                                   #
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

# local imports

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

_dbytes = np.dtype(np.double).itemsize
_ibytes = np.dtype(np.int).itemsize

def pdist_chunk(X, filename, chunk_size=None, metric="euclidean"):
    X = np.asarray(X)
    n = X.shape[0]
    size = n * (n - 1) // 2
    bytes = long(size*_dbytes)
    
    # setup storage
    with open(filename, 'w+b') as f:
        f.seek(bytes-1,0)
        f.write(np.compat.asbytes("\0"))
        f.flush()
    
        row = 0
        k = 0
        rem = size
        if chunk_size is not None:
            while rem > chunk_size:
                storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*dbytes, shape=(n-1-row,))
                storage[:] = sp_distance.cdist(X[row:row+1], X[row+1:], metric=metric)[:]
                storage.flush()
                k += n-1-row
                row += 1
                rem = size - k
        storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*_dbytes, shape=(rem,))
        storage[:] = sp_distance.pdist(X[row:], metric=metric)
        storage.flush()
        
    
def argsort_chunk(infilename, outfilename, chunk_size=None):
    with open(infilename, 'rb') as fin:
        fin.seek(0,2)
        bytes = fin.tell()
        if (bytes % _dbytes):
            raise ValueError("Size of available data is not multiple of data-type size.")
        size = bytes // _dbytes
    
        # set up index storage
        with open(outfilename, 'w+b') as fout:
            bytes = long(size*_ibytes)
            fout.seek(bytes-1, 0)
            fout.write(np.compat.asbytes('\0'))
            fout.flush()
            
            num_chunks = np.ceil(size / chunk_size)
            chunk_offsets = np.arange(0, size, chunk_size)
            chunk_sizes = np.array([chunk_size]*(num_chunks-1)+[size-chunk_offsets[-1]])
            chunk_lower = np.empty(chunk_size, dtype=np.double)
            chunk_upper = np.empty(chunk_size, dtype=np.double)
            
            # initial sorting of segments
            for i in range(num_chunks):
                val_storage = np.memmap(fin, dtype=np.double, mode="r+", offset=chunk_offsets[i]*_dbytes, shape=(chunk_sizes[i],))
                indices = np.argsort(val_storage)
                ind_storage = np.memmap(fout, dtype=np.int, mode="r+", offset=chunk_offsets[i]*_ibytes, shape=(chunk_sizes[i],))
                ind_storage[:] = indices+chunk_offsets[i]
                ind_storage.flush()
                val_storage[:] = val_storage[indices]
                val_storage.flush()
                chunk_lower[i] = val_storage[0]
                chunk_upper[i] = val_storage[-1]
            
            

            def quicksort_chunk(lo, hi):
                if lo < hi:
                    p = partition_chunk(lo, hi)
                    quicksort_chunk(lo, p)
                    quicksort_chunk(p+1, hi)
                    
            def partition_chunk(lo, hi):
                val_pivot = chunk_lower[lo]
                i = lo
                j = hi
                while True:
                    while chunk_upper[i] < pivot:
                        i += 1
                    
                    while chunk_lower[j] > pivot:
                        j -= 1
                      
                    if i >= j:
                        return i
                    
                    # sort and swap i and j
                    val_i_storage = np.memmap(fin, dtype=np.double, mode="r+", offset=chunk_offset[i]*_dbytes, size=(chunk_sizes[i],))
                    val_j_storage = np.memmap(fin, dtype=np.double, mode="r+", offset=chunk_offset[j]*_dbytes, size=(chunk_sizes[j],))
                    old_values = np.concatentate((val_i_storage, val_j_storage))
                    indices = np.argsort(old_values)
                    ind_i_storage = np.memmap(fout, dtype=np.int, mode="r+", offset=chunk_offset[i]*_ibytes, size=(chunk_sizes[i],))
                    ind_j_storage = np.memmap(fout, dtype=np.int, mode="r+", offset=chunk_offset[j]*_ibytes, size=(chunk_sizes[j],))
                    old_indices = np.concatenate((ind_i_storage, ind_j_storage))
                    ind_i_storage[:] = old_indices[indices[:chunk_sizes[i]]]
                    ind_j_storage[:] = old_indices[indices[chunk_sizes[i]:]]
                    ind_i_storage.flush()
                    ind_j_storage.flush()
                    new_values = old_values[indices]
                    val_i_storage[:] = old_values[indices[:chunk_sizes[i]]]
                    val_j_storage[:] = old_values[indices[chunk_sizes[i]:]]
                    chunk_lower[i] = val_i_storage[0]
                    chunk_upper[i] = val_i_storage[-1]
                    chunk_lower[j] = val_j_storage[0]
                    chunk_upper[j] = val_j_storage[-1]
                        
                    
                    
                        
                    
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
