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
class Classification:
    """
    Class for carrying gene taxonomic classification data around, constructed 
    using Classification Manager class.
    """
    
    def __init__(self, taxstrings):
        """
        Fields
        ------
        # data
        _table: ndarray
            n-by-7 array where n is the number of mappings. `table[i]` contains
            indices into the `taxons` array corresponding to the taxon with the
            corresponding ranks for each column:
                0 - Domain
                1 - Phylum
                2 - Class
                3 - Order
                4 - Family
                5 - Genus
                6 - Species
        
        # metadata
        _taxons: ndarray
            Array of taxonomic classification strings.
        """
        n = len(taxstrings)
        taxon_dict = { 0: "" }
        counter = 0
        table = np.zeros((n, 7), dtype=int)
        for (i, s) in enumerate(taxstrings):
            for (j, rank) in enumerate(_parse_taxstring(s)):
                try:
                    table[i, j] = taxons[rank]
                except KeyError:
                    counter += 1
                    table[i, j] = counter
                    taxons[rank] = counter
                    
        taxons = np.array(taxon_dict.values())
        taxons[taxon_dict.keys()] = taxons.copy()
        
        self._table = table
        self._taxons = taxons
    
    def distances(self):
        return sp_distance.pdist(self._table, _classification_distance)
        
    def tags(self, index):
        """Return a list of taxonomic tags"""
        return [t+self._taxons[i] for (t, i) in zip(_TAGS, self._table[index]) if i!=0]

        
_TAGS = ['d__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']


def _classification_distance(a, b):
    for (d, s, o) in zip(range(7, 0, -1), a, b):
        if s==0 or o==0 or s!=o:
            return d
    return 0
    
    
def _parse_taxstring(taxstring):
    fields = taxstring.split('; ')
    if fields[0]=="Root":
        fields = fields[1:]
    ranks = []
    for (string, prefix) in zip(fields, _TAGS):
        try:
            if not string.startswith(prefix):
                raise ValueError("Error parsing field: '%s'. Missing `%s` prefix." % (string, prefix))
            ranks.append(string[len(prefix):].strip())
        except ValueError as e:
            print e, "Skipping remaining fields"
            break
    return ranks
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
class ClassificationConsensusFinder:
    """Wraps a connectivity matrix and determines consensus classifications by
    finding `cliques` (fully-connected subgraphs) in the matrix.
    """
    def __init__(self, mapping, level):
        self._classification = mapping.classification
        self._level = level
        self._mC = mapping.makeConnectivity(self._level)
        
    def maxClique(self, indices):
        """Compute a maximal set `P(i)` of indices j such that `C[j,k] == True`
        for all pairs `j`,`k` from `Q(i)"""
        if len(indices) == 0:
            return np.array([], dtype=np.intp)
        return greedy_clique_by_elimination(self._mC[np.ix_(indices, indices)])
        
    def disagreement(self, indices):
        """Compute size difference between 2 largest cliques"""
        if len(indices) == 0:
            return 0
        first_clique = greedy_clique_by_elimination(self._mC[np.ix_(indices, indices)])
        remaining = np.setdiff1d(indices, indices[first_clique])
        second_clique = greedy_clique_by_elimination(self._mC[np.ix_(remaining, remaining)])
        return abs(len(first_clique) - len(second_clique))
        
    def consensusTag(self, indices):
        indices = np.asarray(indices)
        q = indices[self.maxClique(indices)]
        consensus_tag = ""
        level = 7
        for i in q:
            tags = [t for t in zip(range(7-self._level), self._classification.tags(i))]
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

