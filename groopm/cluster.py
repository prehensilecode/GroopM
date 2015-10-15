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
import scipy.stats as stats

# GroopM imports
from profileManager import ProfileManager
from binManager import BinManager
from hybridMeasure import HybridMeasure

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ClusterEngine:
    """Top level interface for clustering contigs"""
    def __init__(self,
                 dbFileName,
                 timer,
                 minSize=5,
                 minVol=1000000):

        # Worker classes
        self.PM = ProfileManager(dbFileName) # store our data
        self.BM = BinManager(pm=self.PM, minSize=minSize, minVol=minVol)
        self.update = MediodClusterMaker(self.PM, init=numpy.argsort(self.PM.contigLengths), threshold=0.5)

        # Misc tools we'll need
        self.timer = timer

        # Configuration
        self.minSize = minSize
        self.minVol = minVol

    def run(self, coreCut):
        """Cluster the contigs to make bin cores"""
        # check that the user is OK with nuking stuff...
        if(not self.PM.promptOnOverwrite()):
            return False

        # get some data
        self.PM.loadData(self.timer, "length >= "+str(coreCut))
        print "    %s" % self.timer.getTimeStamp()

        # transform the data
        print "    Loading transformed data"
        self.PM.transformCP(self.timer)
        print "    %s" % self.timer.getTimeStamp()

        # cluster and bin!
        print "Create cores"
        self.makeBins(init=numpy.argsort(self.PM.contigLengths))
        print "    %s" % self.timer.getTimeStamp()

        # Now save all the stuff to disk!
        print "Saving bins"
        self.BM.saveBins(nuke=True)
        print "    %s" % self.timer.getTimeStamp()

    def make_bins(self, init):
        """Make the bins"""

        label_counter = -1
        labels = numpy.full(self.PM.numContigs, label_counter)
        round_counter = 0
        mediod = None

        while(True):
            if mediod is None:
                unbinned = init[labels[init] == -1]
                if len(unbinned) == 0:
                    break
                mediod = unbinned[0]
                label_counter += 1
                labels[mediod] = label_counter

            round_counter += 1
            print "Recruiting bin %d, round %d." % (label_counter, round_counter)
            print "Found %d unbinned." % numpy.count_nonzero(labels == -1)

            old_labels = labels
            (labels, new_mediod) = self.update(old_labels, mediod)

            print "Recruited %d members." % numpy.count_nonzero(labels - old_labels)

            if new_mediod == mediod:
                print "Mediod is stable after %d rounds." % count
                count = 0
                mediod = None
            else:
                mediod = new_mediod

        self.unbin_low_quality_assignments(labels)
        self.assign_bins(labels)

    def unbin_low_quality_assignments(bin_assignments):
        """Check bin assignment quality using Bin Manager"""
        low_quality = []
        for label in set(bin_assignments):
            # -1 == unbinned
            if label == -1:
                continue

            members = numpy.flatnonzero(numpy.asarray(bin_assignments) == label)
            total_BP = numpy.sum(self.PM.contigLengths[members])
            bin_size = len(members)

            if not self.BM.isGoodBin(total_BP, bin_size, ms=self.minSize, mv=self.minVol):
                # This partition is too small, ignore
                low_quality.append(label)

        print " Found %d low quality bins." % len(low_quality)
        bin_assignments[numpy.in1d(bin_assignments, low_quality)] = -1


    def assign_bins(self, bin_assignments):
        """Make bins from grouping list"""

        # number of clusters
        n_clusters = len(set(bin_assignments)) - (1 if -1 in bin_assignments else 0)

        bids_made = []
        for label in set(bin_assignments):
            # -1 == unbinned
            if label == -1:
                continue

            members = numpy.where(numpy.asarray(bin_assignments) == label)[0]
            # time to make a bin
            bin = self.BM.makeNewBin(rowIndices=members)

            # work out the distribution in points in this bin
            bin.makeBinDist(self.PM.transformedCP, self.PM.averageCoverages, self.PM.kmerNormPC1, self.PM.kmerPCs,
                self.PM.contigGCs, self.PM.contigLengths)

            # append this bins list of mapped rowIndices to the main list
            bids_made.append(bin.id)

        print " %d bins made." % len(bids_made)

class MediodClusterMaker:
    """Update cluster labels based on current mediod"""
    def __init__(self, PM, threshold):
        self.measure = HybridMeasure(PM)
        self.threshold = threshold

    def __call__(self, labels, mediod):
        is_unbinned = labels==-1
        num_unbinned = numpy.count_nonzero(is_unbinned)
        if num_unbinned==0:
            return (labels, mediod)

        current_label = labels[mediod]
        putative_members = numpy.flatnonzero(numpy.logical_and(labels==current_label, is_unbinned))

        distances = self.measure.get_distances([mediod], putative_members)[:, 0, :]
        ranks = argrank(distances, axis=0)
        recruited = get_mergers(ranks, threshold=self.threshold)
        labels[recruited] = current_labels
        members = numpy.flatnonzero(labels == current_label)
        if len(members)==1:
            mediod = members
        else:
            index = self.measure.get_mediod(members)
            mediod = members[index]

        return (labels, mediod)

###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
#Point counting

def get_inside_count(data, points=None):
    """For each data point return the number of data points that have lower value in all dimensions"""
    data = numpy.asarray(data)
    if points is None:
        points = data
    points = numpy.asarray(points)
    (_num_dims, num_points) = points.shape
    counts = numpy.empty(num_points, dtype=int)
    for i in range(num_points):
        is_inside = numpy.all([d <= c for (c, d) in zip(points[:, i], data)], axis=0)
        counts[i] = numpy.count_nonzero(is_inside) - 1 # discount origin

    return counts

def get_outside_count(data, inner_points, outer_points=None):
    """Return the number of data points that have values in a region between each data point and an internal point."""

    if outer_points is None:
        outer_points = get_bounding_points(inner_points, data)

    outer_counts = get_inside_count(data, outer_points)
    inner_counts = get_inside_count(data, inner_points)

    return outer_counts - inner_counts

def get_bounding_points(a_points, b_points):
    """Return a bounding region that contains both of a pair of points."""
    (num_a_dims, num_a_points) = a_points.shape
    (_num_b_dims, num_b_points) = b_points.shape

    swap = num_b_points > num_a_points
    (first, second, num_first, num_second) = (b_points, a_points, num_b_points, num_a_points) if num_b_points > num_a_points else (a_points, b_points, num_a_points, num_b_points)

    bounds = numpy.empty_like(first)
    for (i, j) in numpy.broadcast(range(num_first), range(num_second)):
        bounds[:, i] = numpy.maximum(first[:, i], second[:, j])

    return bounds

#------------------------------------------------------------------------------
#Rank correlation testing

def get_inside_p_null(ranks):
    """For each data point return the probability in uncorrelated data of having at least as high inner point count"""
    ranks = numpy.asarray(ranks, dtype=float)
    (_num_dims, num_points) = ranks.shape
    counts = get_inside_count(ranks)

    # For a point with ranks r, the probability of another point having a
    # lower rank in all dimensions is `prod(r / r_max)` where r_max is the
    # maximum rank, equal to the total number of points.
    r_max = num_points - 1
    p_inside = numpy.prod(ranks / r_max, axis=0)

    # Statistical test counts
    return binom_one_tailed_test(counts, r_max, p_inside)

def get_outside_p_null(ranks, inners):
    """For each data point return the probability in uncorrelated data of having at least as high inner point count"""
    ranks = numpy.asarray(ranks, dtype=float)
    (_num_dims, num_points) = ranks.shape
    outer_points = get_bounding_points(ranks[:, inners], ranks)
    counts = get_outside_count(ranks, ranks[:, inners], outer_points=outer_points)

    # For a pair of points r, s the probability of a point with all ranks
    # at least as low as r and higher s is
    # `prod(r / r_max) - prod(s / r_max)` where r_max is the highest rank,
    # equal to the total number of points
    r_max = num_points - 1
    p_outside = numpy.prod(outer_points / r_max, axis=0) - numpy.prod(ranks[:, inners] / r_max, axis=0)

    # Statistical test counts
    return binom_one_tailed_test(counts, r_max, p_outside)

def binom_one_tailed_test(counts, ns, ps):
    """Test counts against one-tailed binomial distribution"""
    return numpy.array([stats.binom.sf(c-1, n, p) for (c, n, p) in numpy.broadcast(counts, ns, ps)])

#------------------------------------------------------------------------------
#Extrema masking

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


def is_extrema_mask(points, scores, inners=None, threshold=0.5):
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

def flood_partition_with_mask(points, is_mask):
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

def get_origin_partition(ranks, scores, threshold):
    """Return points in origin partition"""
    ranks = numpy.asarray(ranks)
    is_origin = numpy.all(ranks == 0, axis=0)

    is_mask = is_extrema_mask(ranks, scores, threshold=threshold)
    partitions = flood_partition_with_mask(ranks, is_mask)

    return numpy.flatnonzero(partitions == partitions[is_origin])


#------------------------------------------------------------------------------
#Mergers

def get_mergers(ranks, threshold):
    """Recruit points with a significant rank correlation"""

    scores = get_inside_p_null(ranks)
    origin_partition = get_origin_partition(ranks, scores, threshold)
    index = origin_partition[numpy.argmin(scores[origin_partition])]

    scores = get_outside_p_null(ranks, index)
    origin_partition = get_origin_partition(ranks, scores, threshold)

    is_merger = numpy.zeros(samps, dtype=bool)
    for i in origin_partition:
        is_inside = numpy.all([d <= d[i] for d in ranks], axis=0)
        is_merger = numpy.logical_or(is_merger, is_inside)

    return numpy.flatnonzero(is_merger)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
