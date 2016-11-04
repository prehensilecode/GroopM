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

def pdist_chunk(X, filename, chunk_size=None, metric="euclidean"):
    X = np.asarray(X)
    n = X.shape[0]
    npairs = n * (n - 1) // 2
    bytes = long(npairs*_dbytes)
    
    # setup storage
    with open(filename, 'w+b') as f:
        f.seek(bytes-1,0)
        f.write(np.compat.asbytes("\0"))
        f.flush()
    
        row = 0
        k = 0
        if chunk_size is not None:
            while (npairs - k) > chunk_size:
                storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*dbytes, shape=(n-1-row,))
                storage[:] = sp_distance.cdist(X[row:row+1], X[row+1:], metric=metric)[:]
                storage.flush()
                k += n-1-row
                row += 1
        storage = np.memmap(f, dtype=np.double, mode="r+", offset=k*_dbytes, shape=(npairs-k,))
        storage[:] = sp_distance.pdist(X[row:], metric=metric)
        storage.flush()
    

def argsort_chunk(infilename, outfilename, chunk_size=None):
    with open(infilename, 'rb') as fin:
        fin.seek(0,2)
        bytes = fin.tell()
        if (bytes % _dbytes):
            raise ValueError("Size of available data is not multiple of data-type size.")
        size = bytes // _dbytes
        bytes = long(size*_dbytes)
        
        segments = 2 ** np.ceil(np.log2(size * 1. / chunk_size))
        segment_size = size // segments
    
        # set up storage
        with open(outfilename, 'w+b') as fout:
            fout.seek(bytes-1, 0)
            fout.write(np.compat.asbytes("\0"))
            fout.flush()
            
            with open(outfilename+"2", 'w+b') as ftmp:
                ftmp.seek(bytes-1, 0)
                ftmp.write(np.compat.asbytes("\0"))
                ftmp.flush()
        
                first_write = True
                while True:
                    for i in range(0, segments, 2):
                        if first_write:
                            left = np.memmap(fin, )
                        offset1 = i*segment_size
                        offset2 = (i+1)*segment_size
                        
                
                
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
