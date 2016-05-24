#!/usr/bin/env python
###############################################################################
#                                                                             #
#    binManager.py                                                            #
#                                                                             #
#    GroopM - High level bin data management                                  #
#                                                                             #
#    Copyright (C) Michael Imelfort                                           #
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
__credits__ = ["Michael Imelfort", "Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################
import numpy as np

# GroopM imports
from groopmExceptions import BinNotFoundException

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
class BinStats:
    """Class for carrying bin summary statistics, constructed using BinManager
    class.
    
    Fields
    ------
    # statistics
    bids: ndarray
        `bids[i]` is the bin id for the `i`th bin.
    sizes: ndarray
        `sizes[i]` is the total BP for the `i`th bin.
    numContigs: ndarray
        `numContigs[i]` is the number of contigs in the `i`th bin.
    lengthMedians: ndarray
        `lengthMedians[i]` is the median contig length in the `i`th bin.
    lengthRanges: ndarray
        `lengthRanges[i]` is a 2 element array of the min and max contig lengths in the `i`th bin.
    GCMeans: ndarray
        `GCMeans[i]` is the mean GC % in `i`th bin.
    GCStdDevs: ndarray
        `GCStdDevs[i]` is the standard deviation in GC % in `i`th bin.
    """
    pass
    

class BinManager:
    """Class used for manipulating bins
    
    Wraps a Profile object (see profileManager.py)
    """
    def __init__(self,
                 profile):
        self.profile = profile
        
    def getBinIndices(self, bids):
        """Get indices for bins"""
        self.checkBids(bids)
        return np.flatnonzero(np.in1d(self.profile.binIds, bids))
        
    def checkBids(self, bids):
        """Check if bids are valid"""
        is_not_bid = np.logical_not(np.in1d(bids, self.getBids()))
        if np.any(is_not_bid):
            raise BinNotFoundException("ERROR: "+",".join([str(bid) for bid in bids[is_not_bid]])+" are not bin ids")

    def getBids(self, binIds=None):
        """Return a sorted list of bin ids"""
        if binIds is None:
            binIds = self.profile.binIds
        return sorted(set(binIds).difference([0]))
        
    def getBinStats(self, binIds=None):
        if binIds is None:
            binIds = self.profile.binIds
        bids = self.getBids(binIds)
        sizes = []
        num_contigs = []
        length_medians = []
        length_ranges = []
        gc_means = []
        gc_stds = []
        for bid in bids:
            row_indices = np.flatnonzero(binIds == bid)
            num_contigs.append(len(row_indices))
            
            # contig lengths
            lengths = self.profile.contigLengths[row_indices]
            sizes.append(lengths.sum())
            length_medians.append(np.median(lengths))
            length_ranges.append([np.min(lengths), np.max(lengths)])
            
            # GC %
            gcs = self.profile.contigGCs[row_indices]
            gc_means.append(gcs.mean())
            if len(row_indices) > 1:
                gc_stds.append(gcs.std(ddof=1))
            else:
                gc_stds.append(np.nan)
            
        out = BinStats()
        out.bids = np.array(bids)
        out.sizes = np.array(sizes)
        out.numContigs = np.array(num_contigs)
        out.lengthMedians = np.array(length_medians)
        out.lengthRanges = np.array(length_ranges)
        out.GCMeans = np.array(gc_means)
        out.GCStdDevs = np.array(gc_stds)
        
        return out
        
    def unbinLowQualityAssignments(self, out_bins, minSize=0, minPts=0):
        """Check bin assignment quality"""
        low_quality = []
        stats = self.getBinStats(out_bins)
        for (i, bid) in enumerate(stats.bids):
            if not _isGoodBin(stats.sizes[i], stats.numContigs[i], minSize=minSize, minPts=minPts):
                # This partition is too small, ignore
                low_quality.append(bid)

        print "    Found %d low quality bins." % len(low_quality)
        out_bins[np.in1d(out_bins, low_quality)] = 0
        (_, new_bids) = np.unique(out_bins[out_bins != 0], return_inverse=True)
        out_bins[out_bins != 0] = new_bids+1 # bin ids start at 1
        print "    %s bins after removing low quality bins." % len(set(out_bins).difference([0]))
    
    
    
def _isGoodBin(binSize, binPts, minSize, minPts):
    """Does this bin meet my exacting requirements?"""

    # contains enough bp or enough contigs
    return binSize >= minSize or binPts >= minPts
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
