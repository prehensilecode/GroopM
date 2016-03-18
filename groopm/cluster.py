#!/usr/bin/env python
###############################################################################
#                                                                             #
#    cluster.py                                                               #
#                                                                             #
#    A collection of classes / methods used when clustering contigs           #
#                                                                             #
#    Copyright (C) Michael Imelfort, Tim Lamberton                            #
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

__author__ = "Michael Imelfort, Tim Lamberton"
__copyright__ = "Copyright 2012/2013"
__credits__ = ["Tim Lamberton", "Michael Imelfort"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################

import numpy as np
import numpy.linalg as np_linalg
import scipy.cluster.hierarchy as sp_hierarchy

# local imports
import distance

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class HierachicalClusterEngine:
    """Hierarchical clustering algorthm"""
    def __init__(self, dm, pe):
        self._dm = dm #DistanceManager
        self._pe = pe #PartitionEngine
        
    def makeBins(self):
        """Run binning algorithm"""

        dists = self._dm.pdist()
        combined = np_linalg.norm(dists, axis=-1)
        Z = sp_hierarchy.average(combined)
        
        return pe.partition(Z)
        

class MediodsClusterEngine:
    """Iterative mediod clustering algorithm"""
    
    UNBINNED = -1
    
    def __init__(self, dm, re):
        self._dm = dm #DistanceManager
        self._re = re #RecruitEngine
    
    def makeBins(self, init):
        """Run binning algorithm
        
        Parameters
        ----------
        init : ndarray
            Array of indices used to determine starting points for new
            clusters.
        
        Returns
        -------
        T : ndarray
            `T[i]` is the cluster number for the `i`th observation.
        """
        mediod = None
        bin_counter = UNBINNED
        labels = np.full(self._dm.num_obs(), UNBINNED, dtype=int)
        queue = init

        while(True):
            if mediod is None:
                if len(queue) == 0:
                    break
                mediod = queue.pop()
                if labels[mediod] != UNBINNED:
                    mediod = None
                    continue
                round_counter = 0
                bin_counter += 1
                labels[mediod] = bin_counter

            round_counter += 1
            print "Recruiting bin %d, round %d." % (bin_counter, round_counter)
            
            print "Found %d unbinned." % np.count_nonzero(labels == UNBINNED)

            old_size = np.count_nonzero(labels == bin_counter)
            putative_members = np.flatnonzero(np.in1d(labels, [UNBINNED, bin_counter]))
            recruited = self._re.recruit(mediod, putative_members=putative_members)
            
            labels[recruited] = bin_counter
            members = np.flatnonzero(labels == bin_counter)
            
            print "Recruited %d members." % (members.size - old_size)
            
            if len(members)==1:
                new_mediod = members
            else:
                index = distance.mediod(self._dm.pdist(members))
                new_mediod = members[index]


            if new_mediod == mediod:
                print "Mediod is stable after %d rounds." % round_counter
                mediod = None
            else:
                mediod = new_mediod

        print " %d bins made." % bin_counter
        return labels
        

###############################################################################
###############################################################################
###############################################################################
###############################################################################
