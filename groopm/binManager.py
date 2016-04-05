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
class BinManager:
    """Class used for manipulating bins
    
    Wraps an array of bin ids
    """
    
    def __init__(self,
                 profile,
                 minSize=0,
                 minBP=0):

        self._profile = profile
        self._minSize = minSize
        self._minBP = minBP
        
    def getBinIndices(self, bids):
        """Get indices for bins"""
        self.checkBids(bids)
        return np.flatnonzero(np.in1d(self._profile.binIds, bids))
        
    def checkBids(self, bids):
        """Check if bids are valid"""
        is_not_bid = np.logical_not(np.in1d(bids, self.getBids()))
        if np.any(is_not_bid):
            raise BinNotFoundException("ERROR: "+",".join([str(bid) for bid in bids[is_not_bid]])+" are not bin ids")

    def unbinLowQualityAssignments(self, out_bins):
        """Check bin assignment quality"""
        low_quality = []
        for bid in self.getBids(out_bins):
            is_in_bin = out_bins == bid
            total_BP = np.sum(self._profile.contigLengths[is_in_bin])
            bin_size = np.count_nonzero(is_in_bin)

            if not isGoodBin(total_BP, bin_size, minBP=self._minBP, minSize=self._minSize):
                # This partition is too small, ignore
                low_quality.append(bid)

        print "    Found %d low quality bins." % len(low_quality)
        out_bins[np.in1d(out_bins, low_quality)] = 0
        (_, new_bins) = np.unique(out_bins, return_inverse=True)
        out_bins[...] = new_bins
        

    def getBids(self, binIds=None):
        """Return a sorted list of bin ids"""
        if binIds is None:
            binIds = self._profile.binIds
        return sorted(set(binIds).difference([0]))


def isGoodBin(totalBP, binSize, minBP, minSize):
    """Does this bin meet my exacting requirements?"""

    # contains enough bp or enough contigs
    return totalBP >= minBP or binSize >= minSize

###############################################################################
###############################################################################
###############################################################################
###############################################################################
