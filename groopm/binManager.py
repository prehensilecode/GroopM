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
import numpy

# GroopM imports
from profileManager import ProfileManager
from groopmExceptions import BinNotFoundException

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class BinManager:
    """Class used for manipulating bins"""
    def __init__(self,
                 pm,
                 minSize=0,
                 minBP=0):

        self._pm = pm
        self._minSize = minSize
        self._minBP = minBP

    def getBidsByIndex(self, row_indices):
        """Return corresponding bin ids"""
        return self._pm.binIds[row_indices]

    def checkBids(self, bids):
        """Check if bids are valid"""
        is_not_bid = numpy.logical_not(numpy.in1d(bids, self.getBids()))
        if numpy.any(is_not_bid):
            raise BinNotFoundException("ERROR: "+",".join([str(bid) for bid in bids[is_not_bid]])+" are not bin ids")


    def getBinIndices(self, bids):
        """Return array of binned contig indices"""

        self.checkBids(bids)
        return numpy.flatnonzero(numpy.in1d(self._pm.binIds, bids))

    def getUnbinned(self):
        return self.getBinIndices([0])

    def unbinLowQualityAssignments(self):
        """Check bin assignment quality"""
        low_quality = []
        for bid in self.getBids():
            # 0 == unbinned
            if bid == 0:
                continue

            members = self.getBinIndices([bid])
            total_BP = numpy.sum(self._pm.contigLengths[members])
            bin_size = len(members)

            if not isGoodBin(total_BP, bin_size, minBP=self._minBP, minSize=self._minSize):
                # This partition is too small, ignore
                low_quality.append(label)

        print " Found %d low quality bins." % len(low_quality)
        self._PM.bidIds[self.getBinIndices(low_quality)] = 0

    def assignBin(self, row_indices, bid=None):
        """Make a new bin and add to the list of existing bins"""
        if bid is None:
            bid = max(self._pm.binIds) + 1

        self._PM.bidIds[row_indices] = bid

    def saveBins(self, nuke=False):
        """Save binning results

        binAssignments is a hash of LOCAL row indices Vs bin ids
        { row_index : bid }
        PM.setBinAssignments needs GLOBAL row indices
        """
        # save the bin assignments
        self._pm.setBinAssignments(self._getGlobalBinAssignments(), # convert to global indices
                                   nuke=nuke
                                  )

    def getBids(self):
        """Return a sorted list of bin ids"""
        return sorted(set(self._pm.binIds))

    def _getGlobalBinAssignments(self):
        """Merge the bids, raw DB indexes and core information so we can save to disk

        returns a hash of type:

        { global_index : bid }
        """
        # we need a mapping from cid (or local index) to to global index to binID
        return dict(zip(self._pm.indices, self._pm.binIds))


###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
# Helpers

def isGoodBin(totalBP, binSize, minBP, minSize):
    """Does this bin meet my exacting requirements?"""

    # contains enough bp or enough contigs
    return totalBP >= minBP or binSize >= minSize

###############################################################################
###############################################################################
###############################################################################
###############################################################################
