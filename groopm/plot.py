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

    def loadProfile(self, timer):
        return self._pm.loadData(timer, loadBins=True, minLength=0, removeBins=True, bids=[0])

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
        
        profile = self.loadProfile(timer)

        bm = BinManager(profile)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)

        fplot = FeaturePlotter(profile, colorMap=colorMap)
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
        
        
class FeaturePlotter:
    """Plot contigs in feature space"""
    COLOURS = 'rbgcmyk'

    def __init__(self, profile, colorMap="HSV"):
        self._profile = profile
        self._cm = getColorMap(colorMap)

    def plot(self,
             x, y,
             x_label="", y_label="",
             keep=None, highlight=None, divide=None,
             plotContigLengths=False,
             fileName=""
            ):
        """Plot contigs in measure space"""
        fig = pyplot.figure()

        ax = fig.add_subplot(111)
        self.plotOnAx(ax, x, y,
                      x_label=x_label, y_label=y_label,
                      keep=keep, highlight=highlight,
                      plotContigLengths=plotContigLengths)

        if divide is not None:
            for (clr, coords) in zip(self.COLOURS, divide):
                fmt = '-'+clr
                for (x_point, y_point) in zip(*coords):
                    ax.plot([x_point, x_point], [0, y_point], fmt)
                    ax.plot([0, x_point], [y_point, y_point], fmt)

        if(fileName != ""):
            try:
                fig.set_size_inches(6,6)
                pyplot.savefig(fileName,dpi=300)
            except:
                print "Error saving image:", fileName, sys.exc_info()[0]
                raise
        else:
            print "Plotting contig features"
            try:
                pyplot.show()
            except:
                print "Error showing image", sys.exc_info()[0]
                raise

        pyplot.close(fig)
        del fig

    def plotSurface(self,
                    x, y, z,
                    x_label="", y_label="", z_label="",
                    keep=None, highlight=None,
                    plotContigLengths=False,
                    elev=None, azim=None,
                    fileName=""
                   ):
        """Plot a surface computed from coordinates in measure space"""
        fig = pyplot.figure()

        ax = fig.add_subplot(111, projection='3d')
        self.plotOnAx(ax,
                      x, y, z=z,
                      x_label=x_label, y_label=y_label, z_label=label,
                      keep=keep, highlight=highlight,
                      plotContigLengths=plotContigLengths,
                      elev=elev, azim=azim)

        if(fileName != ""):
            try:
                fig.set_size_inches(6,6)
                pyplot.savefig(fileName,dpi=300)
            except:
                print "Error saving image:", fileName, sys.exc_info()[0]
                raise
        else:
            print "Plotting contig features"
            try:
                pyplot.show()
            except:
                print "Error showing image", sys.exc_info()[0]
                raise

        pyplot.close(fig)
        del fig

    def plotOnAx(self, ax,
                 x, y, z=None,
                 x_label="", y_label="", z_label="",
                 keep=None, extents=None, highlight=None,
                 plotContigLengths=False,
                 elev=None, azim=None,
                 colorMap="HSV"
                ):

        # display values
        disp_vals = (x, y, z) if z is not None else (x, y)
        disp_cols = self._profile.contigGCs

        if highlight is not None:
            edgecolors=numpy.full_like(disp_cols, 'k', dtype=str)
            for (clr, hl) in zip(self.COLOURS, highlight):
                edgecolors[hl] = clr
            if keep is not None:
                edgecolors = edgecolors[keep]
        else:
            edgecolors = 'k'

        if plotContigLengths:
            disp_lens = numpy.sqrt(self._profile.contigLengths)
            if keep is not None:
                disp_lens = disp_lens[keep]
        else:
            disp_lens=30

        if keep is not None:
            disp_vals = [v[keep] for v in disp_vals]
            disp_cols = disp_cols[keep]

        sc = ax.scatter(*disp_vals,
                        c=disp_cols, s=disp_lens,
                        cmap=self._cm,
                        vmin=0.0, vmax=1.0,
                        marker='.')
        sc.set_edgecolors(edgecolors)
        sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if z is not None:
            ax.set_zlabel(z_label)

        if extents is not None:
            ax.set_xlim([extents[0], extents[1]])
            ax.set_ylim([extents[2], extents[3]])
            if z is not None:
                ax.set_zlim([extents[4], extents[5]])

        if z is not None:
            ax.view_init(elev=elev, azim=azim)


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
    
    
def getColorMap(colorMapStr):
    if colorMapStr == 'HSV':
        S = 1.0
        V = 1.0
        return matplotlib.colors.LinearSegmentedColormap.from_list('GC', [colorsys.hsv_to_rgb((1.0 + numpy.sin(numpy.pi * (val/1000.0) - numpy.pi/2))/2., S, V) for val in xrange(0, 1000)], N=1000)
    elif colorMapStr == 'Accent':
        return matplotlib.cm.get_cmap('Accent')
    elif colorMapStr == 'Blues':
        return matplotlib.cm.get_cmap('Blues')
    elif colorMapStr == 'Spectral':
        return matplotlib.cm.get_cmap('spectral')
    elif colorMapStr == 'Grayscale':
        return matplotlib.cm.get_cmap('gist_yarg')
    elif colorMapStr == 'Discrete':
        discrete_map = [(0,0,0)]
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))

        discrete_map.append((0,0,0))
        discrete_map.append((141/255.0,211/255.0,199/255.0))
        discrete_map.append((255/255.0,255/255.0,179/255.0))
        discrete_map.append((190/255.0,186/255.0,218/255.0))
        discrete_map.append((251/255.0,128/255.0,114/255.0))
        discrete_map.append((128/255.0,177/255.0,211/255.0))
        discrete_map.append((253/255.0,180/255.0,98/255.0))
        discrete_map.append((179/255.0,222/255.0,105/255.0))
        discrete_map.append((252/255.0,205/255.0,229/255.0))
        discrete_map.append((217/255.0,217/255.0,217/255.0))
        discrete_map.append((188/255.0,128/255.0,189/255.0))
        discrete_map.append((204/255.0,235/255.0,197/255.0))
        discrete_map.append((255/255.0,237/255.0,111/255.0))
        discrete_map.append((1,1,1))

        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        return matplotlib.colors.LinearSegmentedColormap.from_list('GC_DISCRETE', discrete_map, N=20)

    elif colorMapStr == 'DiscretePaired':
        discrete_map = [(0,0,0)]
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))

        discrete_map.append((0,0,0))
        discrete_map.append((166/255.0,206/255.0,227/255.0))
        discrete_map.append((31/255.0,120/255.0,180/255.0))
        discrete_map.append((178/255.0,223/255.0,138/255.0))
        discrete_map.append((51/255.0,160/255.0,44/255.0))
        discrete_map.append((251/255.0,154/255.0,153/255.0))
        discrete_map.append((227/255.0,26/255.0,28/255.0))
        discrete_map.append((253/255.0,191/255.0,111/255.0))
        discrete_map.append((255/255.0,127/255.0,0/255.0))
        discrete_map.append((202/255.0,178/255.0,214/255.0))
        discrete_map.append((106/255.0,61/255.0,154/255.0))
        discrete_map.append((255/255.0,255/255.0,179/255.0))
        discrete_map.append((217/255.0,95/255.0,2/255.0))
        discrete_map.append((1,1,1))

        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        return matplotlib.colors.LinearSegmentedColormap.from_list('GC_DISCRETE', discrete_map, N=20)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
