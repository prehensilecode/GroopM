#!/usr/bin/env python
###############################################################################
#                                                                             #
#    plot.py                                                                  #
#                                                                             #
#    Data visualisation                                                       #
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
__copyright__ = "Copyright 2012-2015"
__credits__ = ["Michael Imelfort", "Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"
__status__ = "Development"

###############################################################################
import os
import sys
import numpy as np

# GroopM imports
from extract import makeSurePathExists
from binManager import BinManager
from coverageAndKmerDistance import CoverageAndKmerDistanceTool, CoverageAndKmerView
from mstore import ContigParser
from corre import getInsidePNull
from cluster import getNearPNull

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class BinPlotter:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, folder=None):
        self._pm = ProfileManager(dbFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)

    def loadData(self, timer):
        self._pm.loadData(timer, loadBins=True, minLength=0, removeBins=True, bids=[0])

    def plot(self,
             timer,
             bids=None,
             origin_mode="mediod",
             highlight_mode="mergers",
             threshold=None,
             plotRanks=False,
             colorMap="HSV",
             prefix="BIN"
            ):
        self.loadData(timer)

        bm = BinManager(self._pm)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)

        fplot = FeaturePlotter(self._pm, colorMap=colorMap)
        for bid in bids:
            fileName = "" if self._outDir is None else os.path.join(self._outDir, "%s_%d.png" % (prefix, bid))

            fplot.plot(fileName=fileName,
                       **makePlotArgs(pm=self._pm,
                                      bid=bid,
                                      origin_mode=origin_mode,
                                      highlight_mode=highlight_mode,
                                      threshold=threshold,
                                      plotRanks=plotRanks))
            if fileName=="":
                break

        print "    %s" % timer.getTimeStamp()


###############################################################################
# Helpers
###############################################################################

def makePlotArgs(pm, bid, origin_mode, highlight_mode, threshold, plotRanks):
    """Compute plot feature values and labels"""

    members = BinManager(pm).getBinIndices(bid)
    origin = getOrigin(pm, origin_mode, members)
    view = CoverageAndKmerView(pm, origin)
    highlight = getHighlight(mode=highlight_mode,
                             ranks=[view.covRanks, view.kmerRanks],
                             threshold=threshold,
                             members=members)
    (x, y) = (view.covRanks, view.kmerRanks) if plotRanks else (view.covDists, view.kmerDists)
    (x_label, y_label) = (view.covLabel, view.kmerLabel)
    if plotRanks:
        x_label += " rank"
        y_label += " rank"

    return {"x": x,
            "y": y,
            "x_label": x_label,
            "y_label": y_label,
            "highlight": highlight}


def getHighlight(mode, ranks=None, threshold=None, members=None):
    """"Get a tuple of sets of contigs to highlight.
    """
    if mode is None:
        highlight = None
    elif mode=="cluster":
        highlight = (members,)
    elif mode=="mergers":
        highlight = (cluster.getMergers(ranks, threshold),)
    else:
        raise ValueError("Invalid mode: %s" % mode)

    return highlight

def getOrigin(pm, mode, members):
    """Compute a view for a representative contig of a cluster"""

    if mode=="mediod":
        index = CoverageAndKmerDistanceTool(pm).getMediod(members)
    elif mode=="max_coverage":
        index = np.argmax(pm.normCoverages[members])
    elif mode=="max_length":
        index = np.argmax(pm.contigLengths[members])
    else:
        raise ValueError("Invalid mode: %s" % mode)

    return members[index]

def getSurface(mode, ranks):
    """Computes derived surface in hybrid measure space"""
    if mode=="corr_inside":
        z = np.log10(getInsidePNull(ranks))
        z_label = "Inside correlation"
    elif mode=="corr_near":
        z = np.log10(getNearPNull(ranks))
        z_label = "Outside correlation"
    else:
        raise ValueError("Invaild mode: %s" % mode)

    return (z, z_label)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
