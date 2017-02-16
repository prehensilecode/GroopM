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
import os

# local imports
from stream_ext import merge

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def pdist_chunk(X, filename, chunk_size=None, **kwargs):
    """
    Pairwise distances between observations in n-dimensional space. Output is
    written to the passed file, without loading all of the distances into memory.
    
    X and kwargs are passed to scipy `pdist` and `cdist` functions. See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy-spatial-distance-pdist
    """
    X = np.asarray(X)
    n = X.shape[0]
    size = n * (n - 1) // 2
    dbytes = np.dtype(np.double).itemsize
    bytes = long(size*dbytes)
    
    # setup storage
    with open(filename, 'w+b') as f:
        # Allocate required space on disk
        f.seek(bytes-1,0)
        f.write(np.compat.asbytes("\0"))
        f.flush()
        
        row = 0
        k = 0
        rem = size
        if chunk_size is not None:
            while rem > chunk_size:
                storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*dbytes, shape=(chunk_size,))
                pos_storage = 0
                while pos_storage + n-1-row < chunk_size:
                    storage[pos_storage:pos_storage+n-1-row] = sp_distance.cdist(X[row:row+1], X[row+1:], **kwargs)
                    pos_storage += n-1-row
                    row += 1
                storage.flush()
                k += pos_storage
                rem -= pos_storage
        storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*dbytes, shape=(rem,))
        storage[:] = sp_distance.pdist(X[row:], **kwargs)
        storage.flush()

        
def argsort_chunk_mergesort(infilename, outfilename, chunk_size=None, dtype=np.double):
    """
    Sort input file data and store sorting indices in an output file, without
    loading all input data into memory at once.
    """
    dbytes = np.dtype(dtype).itemsize
    ibytes = np.dtype(np.int).itemsize
    
    # load input
    fin = open(infilename, 'r+b')
    fin.seek(0,2)
    bytes = fin.tell()
    if (bytes % dbytes):
        raise ValueError("Size of available data is not multiple of data-type size.")
    size = bytes // dbytes
    
    # initialise index storage to correct size
    fout = open(outfilename, 'w+b')
    bytes = long(size*ibytes)
    fout.seek(bytes-1, 0)
    fout.write(np.compat.asbytes('\0'))
    fout.flush()
    
    # helper functions
    def get_val_storage(offset, size):
        return np.memmap(fin, dtype=np.double, mode="r+", offset=offset*dbytes, shape=(size,))
    
    def get_ind_storage(offset, size):
        return np.memmap(fout, dtype=np.int, mode="r+", offset=offset*ibytes, shape=(size,))
    
    if chunk_size is not None:
        # optimise chunk size so that the number of chunks is a power of 2
        num_rounds = np.ceil(np.log2(size * 1. / chunk_size))
        num_chunks = 2**num_rounds
        chunk_size = int(np.ceil(size * 1. / num_chunks))
                 
    # initial sorting of segments
    k = 0
    rem = size
    while rem > 0:
        l = rem if chunk_size is None or rem < chunk_size else chunk_size
        val_i_storage = get_val_storage(offset=k, size=l)
        indices = np.argsort(val_i_storage)
        ind_i_storage = get_ind_storage(offset=k, size=l)
        ind_i_storage[:] = indices+k
        val_i_storage[:] = val_i_storage[indices]
        ind_i_storage.flush()
        val_i_storage.flush()
        
        k += l
        rem -= l
    
    if chunk_size is None:
        assert rem == 0
        return 
        
    
    # standard mergesort applied to initial sorted segments
    # loop over and merge pairs of adjacent segments, double
    # segment size and repeat, stopping when segment size
    # contains the entire array.
        
    segment_size = chunk_size
    while segment_size < size:
        
        # Pairs of segments are sorted and stored into the same memory,
        # using a buffered, chunked mergesort. A chunk of the first segment,
        # 'i' is copied to temporary value and index buffers, then a chunk of
        # the temporary buffer and the second segment, 'j' are read and used to
        # sort a chunk of data into segment i, keeping track of how many values
        # from the buffer and segment j were merged. The process repeats, with
        # the next chunk from the first segment being buffered, and chunks from
        # the buffer and segment j being read starting from the first unmerged
        # elements.
        
        k = 0
        rem = size
        while rem > 0:
            assert rem > segment_size
            l = np.minimum(2*segment_size, rem) # size of the pair of segments
                        
            # we use two temporary files to buffer unsorted values and indices
            f2in = open(infilename+".2", "w+b")
            f2out = open(outfilename+".2", "w+b")
            
            def get_val_buff(offset, size):
                return np.memmap(f2in, dtype=np.double, mode="r+", offset=offset*dbytes, shape=(size,))
                
            def get_ind_buff(offset, size):
                return np.memmap(f2out, dtype=np.int, mode="r+", offset=offset*ibytes, shape=(size,))
            
            offset_i = 0 # segment i
            offset_j = segment_size # segment j
            offset_buff = 0 # buffer
            while offset_i < l:
                
                # get a chunk of values and indices from segment i
                il = np.minimum(chunk_size, l - offset_i)
                val_i_storage = get_val_storage(offset=k+offset_i, size=il)
                ind_i_storage = get_ind_storage(offset=k+offset_i, size=il)
                
                if offset_i < segment_size:
                    # append the values and indices to the buffer storage
                    val_buff = get_val_buff(offset=offset_i, size=il)
                    ind_buff = get_ind_buff(offset=offset_i, size=il)
                    val_buff[:] = val_i_storage
                    ind_buff[:] = ind_i_storage
                    val_buff.flush()
                    ind_buff.flush()
                    
                
                # next chunk from buffer to be merged
                buffl = np.minimum(chunk_size, segment_size - offset_buff)
                val_buff= get_val_buff(offset=offset_buff, size=buffl)
                ind_buff = get_ind_buff(offset=offset_buff, size=buffl)
                
                # next chunk from segment j to be merged
                jl = np.minimum(chunk_size, l - offset_j)
                val_j_storage = get_val_storage(offset=k+offset_j, size=jl)
                ind_j_storage = get_ind_storage(offset=k+offset_j, size=jl)
                                               
                (pos_buff, pos_j) = merge(val_buff,
                                          ind_buff,
                                          val_j_storage,
                                          ind_j_storage,
                                          val_i_storage,
                                          ind_i_storage)
                
                
                val_i_storage.flush()
                ind_i_storage.flush()
                val_buff.flush()
                ind_buff.flush()
                val_j_storage.flush()
                ind_j_storage.flush()
                
                offset_i += il # end of merged values
                offset_j += pos_j # first unmerged position in segment j
                offset_buff += pos_buff # first unmerged position in buffer
                
                
            f2in.close()
            os.remove(f2out.name)
            f2out.close()
            os.remove(f2in.name)
            
            k += l
            rem -= l
        
        segment_size = 2 * segment_size
        
    fin.close()
    fout.close()

        
def argrank_chunk(out_filename, indices_filename, weight_fun=None, chunk_size=None, dtype=np.double):
    """
    Reads a file of sorted values and a file of ordering indices, calculates
    fractional ranks and writes them to the first file, without loading all
    values into memory.
    
    Returns an array of ranks in the order specified by the ordering indices file.
    """
    
    argsort_chunk_mergesort(out_filename, indices_filename, chunk_size=chunk_size, dtype=dtype)
    
    ibytes = np.dtype(np.int).itemsize
    dbytes = np.dtype(dtype).itemsize
    
    # load input
    find = open(indices_filename, 'r+b')
    find.seek(0,2)
    bytes = find.tell()
    if (bytes % ibytes):
        raise ValueError("Size of available data is not multiple of data-type size.")
    size = bytes // ibytes
    
    fval = open(out_filename, 'r+b')
    fval.seek(0,2)
    if fval.tell() != size*dbytes:
        raise ValueError("The sizes of input files for indices and values must be equal.")
    
    # helpers
    def get_val_storage(offset, size):
        return np.memmap(fval, dtype=np.double, mode="r+", offset=offset*dbytes, shape=(size,))
    
    def get_ind_storage(offset, size):
        return np.memmap(find, dtype=np.int, mode="r+", offset=offset*ibytes, shape=(size,))
    
    def calc_fractional_ranks(inds, flag):
        """
        Calculate fractional ranks.
        
        Parameters
        ----------
        inds : ndarray
            Original observation indices in sorted order
        flag : ndarray
            Array of booleans indicating final positions for equal valued streaks
        """
        if weight_fun is None:
            rflag = np.flatnonzero(flag)+1
        else:
            wts = weight_fun(inds)
            wts[:] = wts.cumsum()
            rflag = wts[flag]
            del wts
        
        total = rflag[-1]
        if len(rflag) > 1:
            rflag[1:] = (rflag[1:] + rflag[:-1] + 1) * 0.5
        rflag[0] = (rflag[0] + 1) * 0.5
        
        # index in array of unique values
        iflag = np.concatenate(([False], flag[:-1])).cumsum()
        return (rflag[iflag], total)
       
    current_rank = 0
    k = 0
    rem = size
    if chunk_size is not None:
        while rem > chunk_size:
            
            # load chunk of sorted values
            val_storage = get_val_storage(offset=k, size=chunk_size)
            
            # identity final values in a streak of equal values
            flag = val_storage[1:] != val_storage[:-1]
            
            # drop items equal to the last value
            keep=len(flag)
            while not flag[keep-1]:
                keep -= 1
            
            ind_storage = get_ind_storage(offset=k, size=keep)
            flag = flag[:keep]
            
            (val_storage[:keep], total) = calc_fractional_ranks(ind_storage, flag)
            val_storage[:keep] += current_rank
            current_rank += total
            val_storage.flush()
            ind_storage.flush()
            
            k += keep
            rem -= keep
    
    # load and compute ranks for remaining
    val_storage = get_val_storage(offset=k, size=rem)
    flag = np.ones(rem, dtype=bool)
    np.not_equal(val_storage[1:], val_storage[:-1], out=flag[:-1])
    
    ind_storage = get_ind_storage(offset=k, size=rem)
    (val_storage[:], total) = calc_fractional_ranks(ind_storage, flag)
    val_storage[:] += current_rank
    current_rank += total
    val_storage.flush()
    ind_storage.flush()
    
    # output array
    out = np.empty(size, dtype=np.double)
    
    k = 0
    rem = size
    if chunk_size is not None:
        while rem > 0:
            l = np.minimum(rem, chunk_size)
            val_storage = get_val_storage(offset=k, size=l)
            ind_storage = get_ind_storage(offset=k, size=l)
            
            out[ind_storage] = val_storage
            val_storage.flush()
            ind_storage.flush()
            
            k += l
            rem -= l
    
    find.close()
    fval.close()
    return out
    
    
def iapply_func_chunk(outfilename, infilename, fun, chunk_size=None, dtype=np.double):
    """
    Apply a binary function to pairs of values stored in two files, writing
    the result to the first file.
    """
    
    dbytes = np.dtype(dtype).itemsize
    
    fin = open(infilename, 'rb')
    fin.seek(0,2)
    bytes = fin.tell()
    if (bytes % dbytes):
        raise ValueError("Size of available data is not multiple of data-type size.")
    size = bytes // dbytes
    
    fout = open(outfilename, 'r+b')
    fout.seek(0,2)
    if fout.tell() != bytes:
        raise ValueError("The size of input file must be equal to output array store.")
    
    def get_input_storage(offset, size):
        return np.memmap(fin, dtype=np.double, mode="r", offset=offset*dbytes, shape=(size,))
        
    def get_output_storage(offset, size):
        return np.memmap(fout, dtype=np.double, mode="r+", offset=offset*dbytes, shape=(size,))
        
    k = 0
    rem = size
    if chunk_size is not None:
        while rem > chunk_size:
            input_storage = get_input_storage(offset=k, size=chunk_size)
            output_storage = get_output_storage(offset=k, size=chunk_size)
            output_storage[:] = fun(output_storage, input_storage)
            
            input_storage.flush()
            output_storage.flush()
            
            k += chunk_size
            rem -= chunk_size
    
    input_storage = get_input_storage(offset=k, size=rem)
    output_storage = get_output_storage(offset=k, size=rem)
    output_storage[:] = fun(output_storage, input_storage)
    input_storage.flush()
    output_storage.flush()
    
    fin.close()
    fout.close()
    
        
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
