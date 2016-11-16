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
from heapq import merge as hq_merge

# local imports
from stream_ext import merge

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
                storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*_dbytes, shape=(n-1-row,))
                storage[:] = sp_distance.cdist(X[row:row+1], X[row+1:], metric=metric)[:]
                storage.flush()
                k += n-1-row
                row += 1
                rem = size - k
        storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*_dbytes, shape=(rem,))
        storage[:] = sp_distance.pdist(X[row:], metric=metric)
        storage.flush()

        
def argsort_chunk_mergesort(infilename, outfilename, chunk_size=None):
    # load input
    fin = open(infilename, 'r+b')
    fin.seek(0,2)
    bytes = fin.tell()
    if (bytes % _dbytes):
        raise ValueError("Size of available data is not multiple of data-type size.")
    size = bytes // _dbytes
    
    # set up index storage
    fout = open(outfilename, 'w+b')
    bytes = long(size*_ibytes)
    fout.seek(bytes-1, 0)
    fout.write(np.compat.asbytes('\0'))
    fout.flush()
    
    def get_val_storage(offset, size):
        return np.memmap(fin, dtype=np.double, mode="r+", offset=offset*_dbytes, shape=(size,))
    
    def get_ind_storage(offset, size):
        return np.memmap(fout, dtype=np.int, mode="r+", offset=offset*_ibytes, shape=(size,))
    
    if chunk_size is not None:
        # optimise chunk size
        num_rounds = np.ceil(np.log2(size * 1. / chunk_size))
        num_chunks = 2**num_rounds
        chunk_size = int(np.ceil(size * 1. / num_chunks))
                 
    # initial sorting of segments
    k = 0
    rem = size
    while rem > 0:
        l = rem if chunk_size is None or rem < chunk_size else chunk_size
        val_storage = get_val_storage(offset=k, size=l)
        indices = np.argsort(val_storage)
        ind_storage = get_ind_storage(offset=k, size=l)
        ind_storage[:] = indices+k
        ind_storage.flush()
        val_storage[:] = val_storage[indices]
        val_storage.flush()
        
        k += l
        rem -= l
    
    if chunk_size is None:
        return 
        
    segment_size = chunk_size
    while segment_size < size:
        
        # loop over pairs of adjacent segments
        k = 0
        rem = size
        while rem > 0:
            assert rem > segment_size
            l = np.minimum(2*segment_size, rem)
            
            #seg1 = get_val_storage(offset=k, size=segment_size)
            #assert np.all(seg1[1:]>=seg1[:-1])
            #seg2 = get_val_storage(offset=k+segment_size, size=l-segment_size)
            #assert np.all(seg2[1:]>=seg2[:-1])
            
            # set up buffers
            f2in = open(infilename+".2", "w+b")
            f2out = open(outfilename+".2", "w+b")
            
            def get_val_buff(offset, size):
                return np.memmap(f2in, dtype=np.double, mode="r+", offset=offset*_dbytes, shape=(size,))
                
            def get_ind_buff(offset, size):
                return np.memmap(f2out, dtype=np.int, mode="r+", offset=offset*_ibytes, shape=(size,))
            
            
            offset_i = 0
            offset_j = segment_size
            offset_buff = 0
            while offset_i < l:
                #print offset_i, offset_j - segment_size + offset_buff
                assert offset_j - segment_size + offset_buff == offset_i
                il = np.minimum(chunk_size, l - offset_i)
                val_i_storage = get_val_storage(offset=k+offset_i, size=il)
                ind_i_storage = get_ind_storage(offset=k+offset_i, size=il)
                
                if offset_i < segment_size:
                    # buffer output storage
                    val_buff = get_val_buff(offset=offset_i, size=il)
                    ind_buff = get_ind_buff(offset=offset_i, size=il)
                    val_buff[:] = val_i_storage
                    ind_buff[:] = ind_i_storage
                    val_buff.flush()
                    ind_buff.flush()
                    
                    #buff = get_val_buff(offset=0, size=offset_i+il)
                    #assert np.all(buff[1:]>=buff[:-1])
                    
                
                # refill buffers
                buffl = np.minimum(chunk_size, segment_size - offset_buff)
                val_buff= get_val_buff(offset=offset_buff, size=buffl)
                ind_buff = get_ind_buff(offset=offset_buff, size=buffl)
                
                jl = np.minimum(chunk_size, l - offset_j)
                val_j_storage = get_val_storage(offset=k+offset_j, size=jl)
                ind_j_storage = get_ind_storage(offset=k+offset_j, size=jl)
                
                #(pos_buff, pos_j) = merge(val_buff,
                                            #ind_buff,
                                            #val_j_storage,
                                            #ind_j_storage,
                                            #val_i_storage,
                                            #ind_i_storage)
                # extension loop
                #x = val_buff.copy()
                #x_ind = ind_buff.copy()
                #y = val_j_storage.copy()
                #y_ind = ind_j_storage.copy()
                #out = np.zeros(il, dtype=x.dtype)
                #out_ind = np.zeros(il, dtype=x_ind.dtype)
                #(i, j) = merge(x,
                               #x_ind,
                               #y,
                               #y_ind,
                               #out,
                               #out_ind)
                # numpy sort
                orig_indices = np.concatenate((ind_buff, ind_j_storage))
                orig_values = np.concatenate((val_buff, val_j_storage))
                #assert np.all(_orig_values[:buffl] == x)
                #assert np.all(_orig_values[buffl:] == y)
                low = orig_values.argpartition(il-1)[:il]
                orig_indices = orig_indices[low]
                orig_values = orig_values[low]
                indices = orig_values.argsort(kind="mergesort")
                orig_indices = orig_indices[indices]
                orig_values = orig_values[indices]
                low = low[indices]
                pos_buff = np.count_nonzero(low < buffl)
                pos_j = np.count_nonzero(low >= buffl)
                #assert np.all(orig_values == out)
                val_i_storage[:] = orig_values
                ind_i_storage[:] = orig_indices
                
                #assert pos_buff==i and pos_j==j
                #assert pos_buff + pos_j == il
                #assert np.all(val_i_storage[1:] >= val_i_storage[:-1])
                val_i_storage.flush()
                ind_i_storage.flush()
                
                offset_i += il
                offset_j += pos_j
                offset_buff += pos_buff
                
                
                #seg = get_val_storage(offset=k, size=offset_i)
                #assert np.all(seg[1:]>=seg[:-1])
                
            f2in.close()
            os.remove(f2out.name)
            f2out.close()
            os.remove(f2in.name)
            
            k += l
            rem -= l
        
        segment_size = 2 * segment_size
        
    fin.close()
    fout.close()

        
def argrank_chunk(indices_filename, values_filename, weight_fun=None, chunk_size=None):
    # load input
    find = open(indices_filename, 'r+b')
    find.seek(0,2)
    bytes = find.tell()
    if (bytes % _ibytes):
        raise ValueError("Size of available data is not multiple of data-type size.")
    size = bytes // _ibytes
    
    fval = open(values_filename, 'r+b')
    fval.seek(0,2)
    if fval.tell() != bytes:
        raise ValueError("The sizes of input files for indices and values must be equal.")
    
    # output array
    out = np.empty(size, dtype=np.double)
    
    def get_val_storage(offset, size):
        return np.memmap(fval, dtype=np.double, mode="r+", offset=offset*_dbytes, shape=(size,))
    
    def get_ind_storage(offset, size):
        return np.memmap(find, dtype=np.int, mode="r+", offset=offset*_ibytes, shape=(size,))
    
    
    # fractional ranks
    def calc_fractional_ranks(inds, flag, begin):
        if weight_fun is None:
            fractional_ranks = np.flatnonzero(flag)+1+begin
        else:
            cumulative_weights = weight_fun(inds)
            cumulative_weights[:] = cumulative_weights.cumsum()
            fractional_ranks = cumulative_weights[flag]+begin
            del cumulative_weights
        
        current_rank = fractional_ranks[-1]
        if len(fractional_ranks) > 1:
            fractional_ranks[1:] = (fractional_ranks[1:] + fractional_ranks[:-1] - 1) * 0.5
        fractional_ranks[0] = (fractional_ranks[0] + begin - 1) * 0.5
        
        # index in array of unique values
        iflag = np.concatenate(([False], flag[:-1])).cumsum()
        return (fractional_ranks[iflag], current_rank)
       
    current_rank = 0
    k = 0
    rem = size
    if chunk_size is not None:
        while rem > chunk_size:
            val_storage = get_val_storage(offset=k, size=chunk_size)
            
            # indicate whether a value is the last of a streak
            flag = val_storage[1:] != val_storage[:-1]
            
            # drop the last value
            keep=len(flag)
            while not flag[keep-1]:
                keep -= 1
            
            ind_storage = get_ind_storage(offset=k, size=keep)
            flag = flag[:keep]
            
            (out[ind_storage], current_rank) = calc_fractional_ranks(ind_storage, flag, begin=current_rank)
            
            k += keep
            rem -= keep
    
    val_storage = get_val_storage(offset=k, size=rem)
    flag = np.ones(rem, dtype=bool)
    np.not_equal(val_storage[1:], val_storage[:-1], out=flag[:-1])
    #flag = np.concatenate((val_storage[1:] != val_storage[:-1], [True]))
    ind_storage = get_ind_storage(offset=k, size=rem)
    (out[ind_storage], current_rank) = calc_fractional_ranks(ind_storage, flag, begin=current_rank)
    
    find.close()
    fval.close()
    return (out, current_rank)
    
    
def iapply_func_chunk(out, filename, fun, chunk_size=None):
    size = out.shape[0]
    bytes = size*_dbytes
    
    f = open(filename, 'rb')
    f.seek(0,2)
    if f.tell() != bytes:
        raise ValueError("The size of input file must be equal to output array store.")
    
    def get_storage(offset, size):
        return np.memmap(f, dtype=np.double, mode="r", offset=offset*_dbytes, shape=(size,))
        
    k = 0
    rem = size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
        while rem > chunk_size:
            storage = get_storage(offset=k, size=chunk_size)
            
            out[k:k+chunk_size] = fun(out[k:k+chunk_size], storage)
            
            k += chunk_size
            rem -= chunk_size
    
    storage = get_storage(offset=k, size=rem)
    out[k:] = fun(out[k:], storage)
    
    f.close()
    
        
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
