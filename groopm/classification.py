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
        taxstrings = self._mapping.taxstrings
        if taxstrings is not None:
            self._classifications = [_Classification(s) for s in taxstrings]
        
    def makeDistances(self):
        """Condensed distance matrix between pairs of marker hits"""
        if self._classifications is not None:
            return _classification_pdist(self._classifications)
        else:
            n = self._mapping.numMappings
            return np.zeros( n * (n - 1) // 2, dtype=np.double)
        
    def makeDisconnectivity(self, level):
        """Condensed disconnectivity matrix"""
        n = self._mapping.numMappings
        dm = self.makeDistances() > level
        
        # disconnect members in the same group
        for (_, m) in self.itergroups():
            idx = distance.pcoords(m, n)
            dm[idx[idx != -1]] = True 
        
        return dm
        
    def itergroups(self):
        """Returns an iterator of marker names and indices."""
        return group_iterator(self._mapping.markerNames)
    

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
        
    def domain(self):
        return self.ranks[0]
        
    def phylum(self):
        return self.ranks[1]
        
    def class_(self):
        return self.ranks[2]
        
    def order(self):
        return self.ranks[3]
        
    def family(self):
        return self.ranks[4]
        
    def genus(self):
        return self.ranks[5]
        
    def species(self):
        return self.ranks[6]
        
    def tags(self):
        for (t, s) in zip(self.TAGS, self.ranks):
            yield t++s
            
    def distance(self, other):
        for (d, s, o) in zip(range(7, 0, -1), self.ranks, other.ranks):
            if s=='' or o=='' or s!=0:
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

