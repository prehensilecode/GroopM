#!/usr/bin/env python
###############################################################################
#                                                                             #
#    coverageAndKmerDistance.py                                               #
#                                                                             #
#    Compute coverage / kmer hybrid distance measure                          #
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

# groopm imports
from corre import argrank

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class CoverageAndKmerDistanceTool:
    """Computes the following metric pair:
        cov  = euclidean distance in log coverage space,
        kmer = euclidean distance in kmer sig space.
    """
    def __init__(self, pm):
        self._pm = pm

    def getDistances(self, a_members, b_members=None):
        """Get distances between two sets of points for the metrics"""

        if b_members is None:
            cov = distance.squareform(distance.pdist(self._pm.covProfiles[a_members], metric="euclidean"))
            kmer = distance.squareform(distance.pdist(self._pm.kmerSigs[a_members], metric="euclidean"))
        else:
            cov = distance.cdist(self._pm.covProfiles[a_members], self._pm.covProfiles[b_members], metric="euclidean")
            kmer = distance.cdist(self._pm.kmerSigs[a_members], self._pm.kmerSigs[b_members], metric="euclidean")

        return numpy.array([cov, kmer])

    def getMediod(self, members):
        """Get member index that minimises the sum rank euclidean distance to other members.

        The sum rank euclidean distance is the sum of the euclidean distances of distance ranks for the metrics.
        """
        # for each member, sum of distances to other members
        scores = [numpy.sum(d, axis=1) for d in self.getDistances(members)]
        ranks = argrank(scores, axis=1)

        # combine using euclidean distance between ranks
        combined = numpy.linalg.norm(ranks, axis=0)
        index = numpy.argmin(combined)

        return index

    def associateWith(self, a_members, b_members):
        """Associate b points with closest a point"""

        distances = self.getDistances(a_members, b_members)
        (_dims, a_num, b_num) = distances.shape

        # rank distances to a points
        ranks = argrank(distances, axis=1)

        # combine using euclidean distance between ranks
        combined = numpy.linalg.norm(ranks, axis=0)
        b_to_a = numpy.argmin(combined, axis=0)

        return b_to_a

    def getDimNames(self):
        return ("log coverage euclidean", "kmer euclidean")


class CoverageAndKmerView:
    """Coverage and kmer distances relative to an origin point"""

    def __init__(self, pm, origin):
        hm = CoverageAndKmerDistanceTool(pm)
        self.origin = origin

        # coverage and kmer distances
        (covDists, kmerDists) = hm.getDistances([self.origin], slice(None))[:, 0, :]
        self.covDists = covDists
        self.kmerDists = kmerDists

        # coverage and kmer ranks
        (covRanks, kmerRanks) = argrank([covDists, kmerDists], axis=1)
        self.covRanks = covRanks
        self.kmerRanks = kmerRanks

        # dim names
        (covLabel, kmerLabel) = hm.getDimNames()
        self.covLabel = covLabel
        self.kmerLabel = kmerLabel

###############################################################################
###############################################################################
###############################################################################
###############################################################################
