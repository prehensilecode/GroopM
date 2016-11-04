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

def pdist_chunks(X, filename, chunk_size=None, metric="euclidean"):
    n = len(X)
    npairs = n * (n - 1) // 2
    row = 0
    k = 0
    itemsize = np.dtype(np.double).itemsize
    if chunk_size is not None:
        while npairs > chunk_size:
            storage = np.memmap(filename, dtype=np.double, mode="w+", offset=k*itemsize, shape=(n-1-row,))
            storage[:] = sp_distance.cdist(X[row], X[row+1:], metric=metric)
            storage.flush()
            k += n-1-row
            row += 1
            npairs -= n-1-row
    storage = np.memmap(filename, dtype=np.double, mode="w+", offset=k*itemsize, shape=(npairs,))
    storage[:] = sp_distance.pdist(X[row:], metric=metric)
    storage.flush()
    
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
