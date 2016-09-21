#!/usr/bin/env python
###############################################################################
#                                                                             #
#    classification.py                                                        #
#                                                                             #
#    Manage marker gene hits classifications                                  #
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
class ClassificationManager:
    """Wraps a connectivity matrix and determines consensus classifications by
    finding `cliques` (fully-connected subgraphs) in the matrix.
    """
    
    def __init__(self, mapping):
        self._classification = mapping.classification
        self._d = 1
        Y = np.logical_or(mapping.makeDistances(), self._classification.makeDistances() >= self._d)
        self._mC = np.logical_not(sp_distance.squareform(Y)).astype(float)
        #self._mC = mapping.makeConnectivity(d=self._d)
        
    def BCubed(self, indices):
        """Compute BCubed metrics"""
        correct = self._mC[np.ix_(indices, indices)].sum(axis=1)
        prec = correct * 1. / len(indices)
        recall = correct * 1. / self._mC[indices].sum(axis=1)
        return (prec, recall)
        
    def maxClique(self, indices):
        """Compute a maximal set of indices such that `C[j,k] == True`
        for all pairs `j`,`k` from set"""
        if len(indices) == 0:
            return np.array([], dtype=np.intp)
        return greedy_clique_by_elimination(self._mC[np.ix_(indices, indices)])
        
    def consensusTag(self, indices):
        indices = np.asarray(indices)
        q = indices[self.maxClique(indices)]
        consensus_tag = ""
        level = 7
        for i in q:
            tags = [t for t in zip(range(7-self._d), self._classification.tags(i))]
            if len(tags) > 0:
                (o, t) = tags[-1]
                if level > o:
                    consensus_tag = t
                    level = o
        return consensus_tag
        
        
def greedy_clique_by_elimination(C):
    """Find clique from connectivity matrix by repeatedly removing least connected
    nodes. Efficient and should generally be accurate enough for our purposes.
    
    Parameters
    ----------
    C : (N, N) ndarray
        Connectivity matrix for graph with `N` nodes.
        
    Returns
    -------
    q : ndarray
        1-D arrray of node indices of clique.
    """
    C = np.asarray(C, dtype=bool)
    n = C.shape[0]
    if C.shape[1] != n:
        raise ValueError("Connectivity matrix must be square.")
    keep = np.ones(n, dtype=bool)
    while True:
        nkeep = np.count_nonzero(keep)
        if nkeep==0:
            break
        counts = np.sum(C[np.ix_(keep, keep)], axis=1)
        which_min = counts.argmin()
        if counts[which_min] == nkeep:
            break
        keep[keep] = np.arange(nkeep)!=which_min
        
    return np.flatnonzero(keep)
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################

