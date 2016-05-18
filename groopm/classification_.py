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

# local imports
import distance
from utils import group_iterator

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
class ClassificationManager:
    """Class for managing marker gene hits for contigs.
    """
    def __init__(self, markers):
        self._mapping = markers
        self._classifications = [_Classification(s) for s in self._mapping.taxstrings]
        
    def makeDistances(self):
        return _classification_pdist(self._classifications)
        
    def makeConnectivity(self, level):
        """Condensed disconnectivity matrix"""
        dm = sp_distance.squareform(self.makeDistances() <= level)
        
        # disconnect members in the same group
        for (_, m) in self.itergroups():
            dm[np.ix_(m, m)] = False
            dm[m, m] = True 
        
        return dm
        
    def itergroups(self):
        """Returns an iterator of marker names and indices."""
        return group_iterator(self._mapping.markerNames)
        
    def iterindices(self):
        """Returns an iterator of profile and marker indices."""
        return group_iterator(self._mapping.rowIndices)
        
    def tags(self, index):
        """Return a classification tag iterator"""
        return self._classifications[index].tags()
    
    
class ClassificationConsensusFinder:
    def __init__(self, markers, level):
        self._mapping = markers
        self._level = level
        self._cm = ClassificationManager(self._mapping)
        self._mC = self._cm.makeConnectivity(self._level)
        
    def maxClique(self, indices):
        """Compute a maximal set `P(i)` of indices j such that `C[j,k] == True`
        for all pairs `j`,`k` from `Q(i)"""
        if len(indices) == 0:
            return np.array([], dtype=np.intp)
        return greedy_clique_by_elimination(self._mC[np.ix_(indices, indices)])
        
    def disagreement(self, indices):
        """Compute size difference between 2 largest maximal cliques"""
        if len(indices) == 0:
            return 0
        first_clique = greedy_clique_by_elimination(self._mC[np.ix_(indices, indices)])
        remaining = np.ones(len(indices), dtype=bool)
        remaining[first_clique] = False
        remaining = indices[remaining]
        second_clique = greedy_clique_by_elimination(self._mC[np.ix_(remaining, remaining)])
        return abs(len(first_clique) - len(second_clique))
        
    def consensusTag(self, indices):
        indices = np.asarray(indices)
        q = indices[self.maxClique(indices)]
        consensus_tag = ""
        level = 7
        for i in q:
            tags = [t for t in zip(range(7-self._level), self._cm.tags(i))]
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
        

class _Classification:
    """Taxonomic classification for a contig based on a marker gene hit."""
    
    TAGS = ['d__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    
    def parse_tag(self, string, prefix):
        if not string.startswith(prefix):
            raise ValueError("Error parsing field: '%s'. Missing `%s` prefix." % (string, prefix))
        return string[len(prefix):].strip()
    
    def parse(self, taxstring):
        fields = taxstring.split('; ')
        if fields[0]=="Root":
            fields = fields[1:]
        ranks = []
        for (field, prefix) in zip(fields, self.TAGS):
            try:
                ranks.append(self.parse_tag(field, prefix))
            except ValueError as e:
                print e, "Skipping remaining fields"
                break
        return ranks
    
    def __init__(self, taxstring):
        self.original_string = taxstring
        self.ranks = self.parse(taxstring)
    
    def taxon(self, taxon):
        try:
            level = ["domain", "phylum", "class", "order", "family", "genus", "species"].index(taxon)
        except ValueError:
            raise ValueError("Unrecognised `taxon` parameter value: `%s`" % taxon)
        if level >= len(self.ranks):
            return None
        return self.ranks[level]
        
    def tags(self):
        for (t, d) in zip(self.TAGS, self.ranks):
            yield t+d
            
    def distance(self, other):
        for (d, s, o) in zip(range(7, 0, -1), self.ranks, other.ranks):
            if s=='' or o=='' or s!=o:
                return d
        return 0
        

def _classification_pdist(clist):
    """Pairwise distances between classifications.
    
    Parameters
    ----------
    clist : list
        list of _Classification objects
        
    Returns
    -------
    Y : ndarray
        Condensed distance matrix for pairs of classifications.
    """
    n = len(clist)
    dm = np.zeros(n * (n - 1) // 2, dtype=np.double)
    
    k = 0
    for i in range(n-1):
        for j in range(i+1, n):
            dm[k] = clist[i].distance(clist[j])
            k = k + 1
            
    return dm
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################

