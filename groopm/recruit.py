#!/usr/bin/env python
###############################################################################
#                                                                             #
#    recruit.py                                                               #
#                                                                             #
#    Recruit nearby contigs                                                   #
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

# GroopM imports
import corre
import distance

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class CorrelationRecruitEngine:
    """Recruit nearby contigs using a correlation threshold."""
    def __init__(self, distanceTool, threshold):
        self._dt = distanceTool # distances from origin point
        self._threshold = threshold
    
    def recruit(self, origin, putative_members=None):
        ranks = distance.argrank(self._dt(origin))
        return getMergers((ranks[:,0], ranks[:,1]), threshold=self._threshold, unmerged=putative_members)

###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
#Extrema mask partitioning

class PartitionSet:
    def __init__(self, size):
        self.ids = numpy.zeros(size, dtype=int)

    def group(self, members):
        ids = set(self.ids[members])
        current_max = max(ids)

        if current_max==0:
            # No members have been grouped previously, so start a new group
            self.ids[members] = max(self.ids) + 1
        else:
            # Merge partitions of all members
            self.ids[members] = current_max
            for j in ids:
                if j > 0:
                    self.ids[self.ids == j] = current_max


def isExtremaMask(points, scores, threshold, inners=None):
    """Find data points with scores higher than the inner minimum score or threshold value"""
    points = numpy.asarray(points)
    scores = self.get_scores(points)

    if inners is None:
        origin = numpy.flatnonzero(numpy.all(points == 0, axis=0))
        inners = [origin]

    (_num_dims, num_points) = points.shape

    is_mask = numpy.empty(num_points)
    is_inside_any_inner = numpy.zeros(num_points, dtype=True)
    for (i, j) in numpy.broadcast(range(num_points), inners):

        inner_point = points[:, j]
        outer_point = numpy.maximum(points[:, i], inner_point)

        is_inside = numpy.all([d <= c for (c, d) in zip(outer_point, points)], axis=0)
        is_inside[i] = False
        is_inside_inner = numpy.all([d <= c for (c, d) in zip(inner_point, points)], axis=0)
        is_inside_any_inner[is_inside_inner] = True
        is_inside[is_inside] = numpy.logical_not(is_inside_inner[is_inside])

        # Return the lower of the lowest internal score and the threshold
        # argument supplied.
        cutoff = threshold if not numpy.any(is_inside) else min(numpy.min(scores[is_inside]), threshold)

        # A `mask` point has a higher score than the lowest internal score.
        is_mask = scores[i] > cutoff

    is_mask[inners] = False

    return is_mask

def floodPartitionWithMask(points, is_mask):
    """Find partitions of unmasked points by joining a point to any outside point closer than an outside mask point"""
    points = numpy.asarray(points)
    (_num_dims, num_points) = points.shape

    partitions = PartitionSet(num_points)
    unmask = numpy.flatnonzero(numpy.logical_not(is_mask))
    for i in unmask:
        is_outside = numpy.any([d >= d[i] for d in points], axis=0)
        is_outside_mask = is_mask[is_outside]

        dist_outside = numpy.linalg.norm([c - d for (c, d) in zip(points[:, i], points[:, is_outside])], axis=0)
        if numpy.any(is_outside_mask):
            dist_cutoff = numpy.min(dist_outside[is_outside_mask])
        else:
            dist_cutoff = numpy.linalg.norm(points[:, i], axis=0)

        outside = numpy.flatnonzero(is_outside)
        members = outside[dist_outside < dist_cutoff]

        partitions.group(members)

    return partitions.ids

def getPartitionMembers(partitions):
    return [numpy.flatnonzero(partitions == i) for i in set(partitions)]

#------------------------------------------------------------------------------
#Merger workflow

def partitionByExtrema(ranks, scores, threshold, unmerged=None):
    """Return points in origin partition"""
    ranks = numpy.asarray(ranks)
    is_mask = isExtremaMask(ranks, scores, threshold)
    if unmerged is not None:
        is_mask[numpy.setdiff1d(numpy.arange(is_mask.size), unmerged)] = True
    return floodPartitionWithMask(ranks, is_mask)

def getOriginPartition(ranks, partitions):
    """Return points in origin partition"""
    ranks = numpy.asarray(ranks)
    is_origin = numpy.all(ranks == 0, axis=0)
    return numpy.flatnonzero(partitions == partitions[is_origin])

def getNearPNull(ranks, threshold, unmerged=None):
    """Get probability scores of points being near inner most partition"""

    scores = corre.getInsidePNull(ranks)
    inside_partition = getOriginPartition(ranks, partitionByExtrema(ranks, scores, threshold, unmerged))
    inside_cutoff = inside_partition[numpy.argmin(scores[inside_partition])]

    return corre.getOutsidePNull(ranks, inside_cutoff)

def getMergers(ranks, threshold, unmerged=None):
    """Recruit points with a significant rank correlation"""

    scores = getNearPNull(ranks, index)
    near_partition = getOriginPartition(ranks, partitionByExtrema(ranks, scores, threshold, unmerged))

    is_merger = numpy.zeros(samps, dtype=bool)
    for i in near_partition:
        is_inside = numpy.all([d <= d[i] for d in ranks], axis=0)
        is_merger = numpy.logical_or(is_merger, is_inside)

    return numpy.flatnonzero(is_merger)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
