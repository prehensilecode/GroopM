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
from binManager import BinManager, BinOriginAPI
from hybridMeasure import HybridMeasure, HybridMeasurePlotter, argrank, PlotOriginAPI
from groopmUtils import makeSurePathExists

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ClusterEngine:
    """Top level interface for clustering contigs"""
    def __init__(self,
                 dbFileName,
                 minSize,
                 minBP,
                 threshold=0.5):

        # Worker classes
        self._pm = ProfileManager(dbFileName)
        self._bm = BinManager(self._pm, minSize=minSize, minBP=minBP)
        self.updateBin = MediodClusterMaker(self._pm, threshold=threshold)


    def run(self, timer, minLength=None, force=False):
        """Cluster the contigs to make bin cores"""
        # check that the user is OK with nuking stuff...
        if not force and not self._pm.promptOnOverwrite():
            return
        self._pm.loadData(timer, minLength=minLength)

        # cluster and bin!
        print "Create cores"
        self.makeBins(init=numpy.argsort(self._pm.contigLengths))
        print "    %s" % timer.getTimeStamp()

        # Now save all the stuff to disk!
        print "Saving bins"
        self._bm.saveBins(nuke=True)
        print "    %s" % timer.getTimeStamp()

    def makeBins(self, init):
        """Make the bins"""

        round_counter = 0
        mediod = None

        while(True):
            if mediod is None:
                to_bin = numpy.intersect1d(init, self._bm.getUnbinned())
                if len(to_bin) == 0:
                    break
                mediod = to_bin[0]
                self._bm.assignBin([mediod])

            round_counter += 1
            bid = self._bm.getCurrentBid()
            print "Recruiting bin %d, round %d." % (bid, round_counter)
            print "Found %d unbinned." % self._bm.getUnbinned().size

            old_size = self._bm.getBinIndices([bid]).size
            new_mediod = self.updateBin(self._bm, mediod)

            print "Recruited %d members." % self._bm.getBinIndices([bid]).size - old_size

            if new_mediod == mediod:
                print "Mediod is stable after %d rounds." % count
                count = 0
                mediod = None
            else:
                mediod = new_mediod

        print " %d bins made." % self._bm.currentBid
        self._bm.unbinLowQualityAssignments()


class MediodClusterMaker:
    """Update cluster labels based on current mediod"""
    def __init__(self, pm, threshold):
        self._hm = HybridMeasure(pm)
        self.threshold = threshold

    def __call__(self, bm, mediod):
        bid = bm.getBidsByIndex(mediod)
        putative_members = bm.getBinIndices([0, bid])

        distances = self._hm.getDistancesToPoint(mediod)
        ranks = argrank(distances, axis=0)
        recruited = getMergers(ranks, threshold=self.threshold, unmerged=putative_members)
        bm.assignBin(recruited, bid=bid)
        members = bm.getBinIndices(bid)
        if len(members)==1:
            mediod = members
        else:
            index = self._hm.getMediod(members)
            mediod = members[index]

        return mediod

###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
#Point counting

def getInsideCount(data, points=None):
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

def getOutsideCount(data, inner_points, outer_points=None):
    """Return the number of data points that have values in a region between each data point and an internal point."""

    if outer_points is None:
        outer_points = getBoundingPoints(inner_points, data)

    outer_counts = getInsideCount(data, outer_points)
    inner_counts = getInsideCount(data, inner_points)

    return outer_counts - inner_counts

def getBoundingPoints(a_points, b_points):
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

def getInsidePNull(ranks):
    """For each data point return the probability in uncorrelated data of having at least as high inner point count"""
    ranks = numpy.asarray(ranks, dtype=float)
    (_num_dims, num_points) = ranks.shape
    counts = getInsideCount(ranks)

    # For a point with ranks r, the probability of another point having a
    # lower rank in all dimensions is `prod(r / r_max)` where r_max is the
    # maximum rank, equal to the total number of points.
    r_max = num_points - 1
    p_inside = numpy.prod(ranks / r_max, axis=0)

    # Statistical test counts
    return binomOneTailedTest(counts, r_max, p_inside)

def getOutsidePNull(ranks, inners):
    """For each data point return the probability in uncorrelated data of having at least as high inner point count"""
    ranks = numpy.asarray(ranks, dtype=float)
    (_num_dims, num_points) = ranks.shape
    outer_points = getBoundingPoints(ranks[:, inners], ranks)
    counts = getOutsideCount(ranks, ranks[:, inners], outer_points=outer_points)

    # For a pair of points r, s the probability of a point with all ranks
    # at least as low as r and higher s is
    # `prod(r / r_max) - prod(s / r_max)` where r_max is the highest rank,
    # equal to the total number of points
    r_max = num_points - 1
    p_outside = numpy.prod(outer_points / r_max, axis=0) - numpy.prod(ranks[:, inners] / r_max, axis=0)

    # Statistical test counts
    return binomOneTailedTest(counts, r_max, p_outside)

def binomOneTailedTest(counts, ns, ps):
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

    scores = getInsidePNull(ranks)
    inside_partition = getOriginPartition(ranks, partitionByExtrema(ranks, scores, threshold, unmerged))
    inside_cutoff = inside_partition[numpy.argmin(scores[inside_partition])]

    return getOutsidePNull(ranks, inside_cutoff)

def getMergers(ranks, threshold, unmerged=None):
    """Recruit points with a significant rank correlation"""

    scores = getNearPNull(ranks, index)
    near_partition = getOriginPartition(ranks, partitionByExtrema(ranks, scores, threshold, unmerged))

    is_merger = numpy.zeros(samps, dtype=bool)
    for i in near_partition:
        is_inside = numpy.all([d <= d[i] for d in ranks], axis=0)
        is_merger = numpy.logical_or(is_merger, is_inside)

    return numpy.flatnonzero(is_merger)

#------------------------------------------------------------------------------
#Plotting tools

class PlotSurfaceAPI:
    """Computes derived surface in hybrid measure space.

    Requires / replaces argument dict values:
        {origin, surface_mode} -> {origin, z, label}
    """
    def __init__(self, pm):
        self._hm = HybridMeasure(pm)

    def __call__(self, surface_mode, **kwargs):
        """Derive surface values"""

        origin = kwargs["origin"]
        distances = self._hm.getDistancesToPoint(origin)
        ranks = argrank(distances, axis=1)
        if surface_mode=="corr_inside":
            z = numpy.log10(getInsidePNull(ranks))
            label = "Inside correlation"
        elif surface_mode=="corr_near":
            z = numpy.log10(getNearPNull(ranks))
            label = "Outside correlation"
        else:
            raise ValueError("Invaild surface mode: %s" % surface_mode)

        kwargs["z"] = z
        kwargs["label"] = label
        return kwargs

class PlotHighlightAPI:
    """"Get a tuple of sets of contigs to highlight.

    Requires / replaces argument dict values:
        {origin, threshold, highlight_mode} -> {origin, highlight}
    """
    def __init__(self, pm):
        self._hm = HybridMeasure(pm)

    def __call__(self, threshold, highlight_mode, **kwargs):

        if highlight_mode is None:
            return kwargs

        origin = kwargs["origin"]
        distances = self._hm.getDistancesToPoint(origin)
        ranks = argrank(distances, axis=1)
        if highlight_mode=="mergers":
            highlight = (getMergers(ranks, threshold),)
        elif highlight_mode=="partitions":
            highlight = tuple(getPartitionMembers(partitionByExtrema(ranks, getInsidePNull(ranks), threshold)))
        elif highlight_mode=="mask_inside":
            highlight = (numpy.flatnonzero(isExtremaMask(ranks, getInsidePNull(ranks), threshold)),)
        elif highlight_mode=="mask_near":
            highlight = (numpy.flatnonzero(isExtremaMask(ranks, getNearPNull(ranks), threshold)),)
        else:
            raise ValueError("Invalide mode: %s" % highlight_mode)

        kwargs["highlight"] = highlight
        return kwargs

class BinHighlightAPI:
    """Highlight contigs from a bin

    Requires / replaces argument dict values:
        {bid, origin_mode, threshold, highlight_mode} -> {origin, highlight}
    """
    def __init__(self, pm):
        self._plotHighlightApi = PlotHighlightAPI(pm)
        self._binOriginApi = BinOriginAPI(pm)
        self._bm = BinManager(pm)

    def __call__(self, highlight_mode, threshold, **kwargs):
        """"""
        if highlight_mode=="bin":
            bid = kwargs["bid"]
            highlight = (self._bm.getBinIndices(bid),)
            return self._binOriginApi(highlight=highlight, **kwargs)
        else:
            try:
                return self._plotHighlightApi(highlight_mode=highlight_mode,
                                              threshold=threshold,
                                              **self._binOriginApi(**kwargs))
            except:
                raise

class SurfaceHighlightAPI:
    """Combined surface + highlight api.

    Requires / replaces argument dict values:
        {origin, threshold, highlight_mode, surface_mode} -> {origin, highlight, z, label}
    """
    def __init__(self, pm):
        self._plotSurfaceApi = PlotSurfaceApi(pm)
        self._plotHighlightApi = PlotHighlightApi(pm)

    def __call__(self, **kwargs):
        return self._plotSurfaceApi(**self._plotHighlightApi(**kwargs))

class BinSurfaceHighlightAPI:
    """Combined surface + highlight + bin api.

    Requires / replaces argument dict values:
        {bid, origin_mode, threshold, highlight_mode, surface_mode} -> {origin, highlihght, z, label}
    """
    def __init__(self, pm):
        self._binOriginApi = BinOriginAPI(pm)
        self._surfaceHighlightApi = SurfaceHighlightAPI(pm)

    def __call__(self, **kwargs):
        return self._surfaceHighlightApi(**self._binOriginApi(**kwargs))


#------------------------------------------------------------------------------
#Plotting

class BinHighlightPlotter:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, bids=[], folder=None):
        self._pm = ProfileManager(dbFileName)
        self.bids = [] if bids is None else bids
        self.outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self.outDir is not None:
            makeSurePathExists(self.outDir)

    def loadData(self, timer, cutoff=0):
        self._pm.loadData(timer, loadBins=True, bids=self.bids, minLength=cutoff)

    def plot(self,
             timer,
             origin_mode="mediod",
             highlight_mode="mergers",
             threshold=None,
             plotRanks=False,
             colorMap="HSV",
             prefix="BIN"
            ):
        self.loadData(timer)
        hmPlot = HybridMeasurePlotter(self._pm)
        bm = BinManager(self._pm)
        binHighlightApi = BinHighlightAPI(self._pm)

        for bid in bm.getBids():
            fileName = "" if self.outDir is None else os.path.join(self.outDir, "%s_%d.png" % (prefix, bid))
            hmPlot.plot(**binHighlightApi(bid=bid,
                                          origin_mode=origin_mode,
                                          highlight_mode=highlight_mode,
                                          threshold=threshold,
                                          plotRanks=plotRanks,
                                          colorMap=colorMap,
                                          fileName=fileName))
            if self.outDir is None:
                break

        print "    %s" % timer.getTimeStamp()


###############################################################################
###############################################################################
###############################################################################
###############################################################################
