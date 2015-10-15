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

import numpy
import scipy.spatial.distance as distance

# GroopM imports
from profileManager import ProfileManager
from binManager import BinManager

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class HybridMeasure:
    """Computes the following metric pair:
        cov  = euclidean distance in log coverage space,
        kmer = euclidean distance in kmer sig space. """
    def __init__(self, PM):
        self.PM = PM

    def get_distances(self, a_members, b_members=None):
        """Get distances between two sets of points for the metrics"""

        if b_members is None:
            cov = distance.squareform(distance.pdist(numpy.log10(self.PM.covProfiles[a_members]+1), metric="euclidean"))
            kmer = distance.squareform(distance.pdist(self.PM.kmerSigs[a_members], metric="euclidean"))
        else:
            cov = distance.cdist(numpy.log10(self.PM.covProfiles[a_members]+1), numpy.log10(self.PM.covProfiles[b_members]+1), metric="euclidean")
            kmer = distance.cdist(self.PM.kmerSigs[a_members], self.PM.kmerSigs[b_members], metric="euclidean")

        return numpy.array([cov, kmer])

    def get_mediod(self, members):
        """Get member index that minimises the sum rank euclidean distance to other members.

        The sum rank euclidean distance is the sum of the euclidean distances of distance ranks for the metrics"""

        # for each member, sum of distances to other members
        scores = [numpy.sum(d, axis=1) for d in self.get_distances(members)]
        ranks = argrank(scores, axis=1)

        # combine using euclidean distance between ranks
        combined = numpy.linalg.norm(ranks, axis=0)
        index = numpy.argmin(combined)

        return index

    def associate_with(self, a_members, b_members):
        """Associate b points with closest a point"""

        distances = self.get_distances(self, a_members, b_members)
        (_dims, a_num, b_num) = distances.shape

        # rank distances to a points
        ranks = argrank(distances, axis=1)

        # combine using euclidean distance between ranks
        combined = numpy.linalg.norm(ranks, axis=0)
        b_to_a = numpy.argmin(combined, axis=0)

        return b_to_a

    def get_dim_names(self):
        """Labels for distances returned by get_distances"""
        return ("log coverage euclidean", "kmer euclidean")


###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
#Ranking

def rank_with_ties(array):
    """Return sorted of array indices with tied values averaged"""
    ranks = numpy.asarray(numpy.argsort(numpy.argsort(array)), dtype=float)
    for val in set(array):
        g = array == val
        ranks[g] = numpy.mean(ranks[g])
    return ranks


def argrank(array, axis=0):
    """Return the positions of elements of a when sorted along the specified axis"""
    return numpy.apply_along_axis(rank_with_ties, axis, array)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
