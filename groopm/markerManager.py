#!/usr/bin/env python
###############################################################################
#                                                                             #
#    markerManager.py                                                         #
#                                                                             #
#    Manage a set of marker gene hits                                         #
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

class MarkerManager:
    """Class for managing marker gene hits for contigs.
    """
    
    def __init__(self,
                 pm,
                 rowIndices,
                 contigNames,
                 markerNames,
                 taxstrings=None):
        
        self.rowIndices = np.asarray(rowIndices)
        n = len(self.rowIndices)
        self.contigNames = np.asarray(contigNames)
        if len(self.contigNames) != n:
            raise ValueError("Lists of contig names and indices must have same length.")
        self.markerNames = np.asarray(markerNames)
        if len(self.markerNames) != n:
            raise ValueError("Lists of marker names and indices must have same length.")
               
        self.taxstrings = None 
        self._classifications = None
        if taxstrings is not None:
            self.taxstrings = np.asarray(taxstrings)
            if len(self.taxstrings) != n:
                raise ValueError("Lists of taxstrings and indices must have same length.")
            self._clist = [_Classification(s) for s in self.taxstrings]
            
        self.numMarkers = n
        
    def makeDistances(self):
        """Condensed distance matrix between pairs of marker hits"""
        if self._clist is None:
            n = self.numMarkers
            return np.zeros( n * (n - 1) // 2, dtype=np.double)
        
        return _classification_pdist(self._clist)
        
    def makeDisconnectivity(self, level):
        """Condensed disconnectivity matrix"""
        n = self.numMarkers
        
        dm = self.makeDistances() > level
        
        # disconnect members in the same group
        gm = np.zeros((n ,n), dtype=bool)
        for (_, m) in self.itergroups():
            gm[np.ix_(m, m)] = True
            gm[m, m] = False
        dm[sp_distance.squareform(gm)] = False
        
        return dm
        
    def itergroups(self):
        """Returns an iterator of marker names and indices."""
        return group_iterator(self.markerNames)


# Utility
def group_iterator(grouping):
    """Returns an iterator of values and indices for a grouping variable."""
    group_dist = {}
    for (i, name) in enumerate(grouping):
        try:
            group_dist[name].append(i)
        except KeyError:
            group_dist[name] = [i]
    
    return group_dist.iteritems()
    

class _Classification:
    """Taxonomic classification for a contig based on a marker gene hit."""
    
    TAGS = ['d__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    
    def parse_tag(self, string, tag):
        if not field.startswith(prefix):
            raise ValueError("Error parsing field: '%s'. Missing `%s` prefix." % (string, prefix))
        return string[len(prefix):].strip()
    
    def parse(self, taxstring):
        fields = taxstring.split('; ')
        if fields[0]=="Root":
            fields = fields[1:]
        ranks = []
        for (field, prefix) in zip(fields, TAGS):
            try:
                ranks.append(self.parse_tag(field, prefix))
            except ValueError:
                print "Skipping remaining fields"
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
        for (t, s) in zip(TAGS, self.ranks):
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

