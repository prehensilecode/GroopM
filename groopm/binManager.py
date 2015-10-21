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

__author__ = "Michael Imelfort"
__copyright__ = "Copyright 2012/2013"
__credits__ = ["Michael Imelfort"]
__license__ = "GPL3"
__maintainer__ = "Michael Imelfort"
__email__ = "mike@mikeimelfort.com"

###############################################################################
import numpy

# GroopM imports
from groopmExceptions import BinNotFoundException

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class BinManager:
    """Class used for manipulating bins"""
    def __init__(self,
                 PM,
                 minSize=5,
                 minVol=1000000):

        self._PM = PM
        self.minSize = minSize
        self.minBP = minBP

        self.bids = self._PM.binIds[self._PM.indices] # bid -> bin
        self.current_bid = numpy.max(self.bids)

    def get_bin_indices(self, bids):
        """Return array of binned contig indices"""

        if not numpy.all(numpy.in1d(bids, self.get_bids())):
            raise BinNotFoundException("Cannot find: "+str(bid)+" in bins dicts")
        return numpy.flatnonzero(numpy.any([self.bids==bid for bid in bids], axis=0))

    def get_unbinned(self):
        return self.get_bin_indices([0])

    def unbin_low_quality_assignments(self):
        """Check bin assignment quality"""
        low_quality = []
        for label in self.get_bids():
            # -1 == unbinned
            if label == -1:
                continue

            members = numpy.flatnonzero(self.bids == label)
            total_BP = numpy.sum(self._PM.contigLengths[members])
            bin_size = len(members)

            if not isGoodBin(total_BP, bin_size, minBP=self.minBP, minSize=self.minSize):
                # This partition is too small, ignore
                low_quality.append(label)

        print " Found %d low quality bins." % len(low_quality)
        self.bids[numpy.in1d(self.bids, low_quality)] = 0

    def assign_bin(self, row_indices, bid=None):
        """Make a new bin and add to the list of existing bins"""
        if bid is None:
            self.current_bid +=1
            bid = self.current_bid

        self.PM.isLikelyChimeric[bid] = False
        self.bids[row_indices] = bid

    def save_bins(self, nuke=False):
        """Save binning results

        binAssignments is a hash of LOCAL row indices Vs bin ids
        { row_index : bid }
        PM.setBinAssignments needs GLOBAL row indices

        We always overwrite the bins table (It is smallish)
        """
        # save the bin assignments
        self._PM.setBinAssignments(
                                  self._get_global_bin_assignments(), # convert to global indices
                                  nuke=nuke
                                  )
        # overwrite the bins table
        self._PM.setBinStats(self._get_bin_stats())

    def get_bids(self):
        """Return a sorted list of bin ids"""
        return sorted(set(self.bids))

    def _get_bin_stats(self):
        """Update / overwrite the table holding the bin stats

        Note that this call effectively nukes the existing table
        """

        # create and array of tuples:
        # [(bid, size, likelyChimeric)]
        bin_stats = [(bid, numpy.count_nonzero(self.bids == bid), self._PM.isLikelyChimeric[bid]) for bid in self.get_bids()]
        return bin_stats

    def _get_global_bin_assignments(self):
        """Merge the bids, raw DB indexes and core information so we can save to disk

        returns a hash of type:

        { global_index : bid }
        """
        # we need a mapping from cid (or local index) to to global index to binID
        return dict(zip(self._PM.indices, self.bids))


###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
# Helpers

def isGoodBin(totalBP, binSize, minBP, minSize):
    """Does this bin meet my exacting requirements?"""

    # contains enough bp to pass regardless of number of contigs
    # or has enough contigs
    return totalBP >= minBP or binSize >= minSize

###############################################################################
###############################################################################
###############################################################################
###############################################################################
