#!/usr/bin/env python
###############################################################################
#                                                                             #
#    profileManager.py                                                        #
#                                                                             #
#    GroopM - High level data management                                      #
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
import sys
import matplotlib
from matplotlib import pyplot
import colorsys

# GroopM imports
from mstore import GMDataManager as DataManager

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################


class ProfileManager:
    """Interacts with the groopm DataManager and local data fields

    Mostly a wrapper around a group of numpy arrays and a pytables quagmire
    """
    def __init__(self, dbFileName, force=False):
        # misc
        self._DM = DataManager()            # most data is saved to hdf
        self.dbFileName = dbFileName         # db containing all the data we'd like to use

        # --> NOTE: ALL of the arrays in this section are in sync
        # --> last axis (axis=-1) holds information for individuals contig
        # --> Think "struct of arrays".
        self.indices = numpy.array([])        # indices into the pytables data structure of selection
        self.covProfiles = numpy.array([])    # coverage based coordinates
        self.kmerSigs = numpy.array([])       # raw kmer signatures
        self.normCoverages = numpy.array([])  # norm of the raw coverage vectors
        self.contigGCs = numpy.array([])
        self.contigNames = numpy.array([])
        self.contigLengths = numpy.array([])
        self.binIds = numpy.array([])         # list of bin IDs
        # --> end section

        # meta
        self.numContigs = 0                   # corresponds to length of axis=-1 of above arrays
        self.stoitNames = numpy.array([])     # names of stoits for each row of covProfiles array
        self.numStoits = 0                    # corresponds to number of rows of covProfiles array and length of stoitColNames array

    def loadData(self,
                 timer,
                 verbose=True,              # many to some output messages
                 silent=False,              # some to no output messages
                 loadCovProfiles=True,
                 loadKmerSigs=True,
                 loadStoitNames=True,
                 loadContigNames=True,
                 loadContigLengths=True,
                 loadContigGCs=True,
                 loadBins=False,
                 minLength=None,
                 bids=[]
                ):
        """Load pre-parsed data"""

        if(silent):
            verbose=False
        if verbose:
            print "Loading data from:", self.dbFileName

        try:
            # Stoit names
            self.numStoits = self._DM.getNumStoits(self.dbFileName)
            if(loadStoitNames):
                self.stoitNames = numpy.array(self._DM.getStoitColNames(self.dbFileName).split(","))

            # Conditional filter
            condition = getConditionString(minLength=minLength, bids=bids)
            print condition
            self.indices = self._DM.getConditionalIndices(self.dbFileName,
                                                          condition=condition,
                                                          silent=silent)

            # Collect contig data
            if(verbose):
                print "    Loaded indices with condition:", condition
            self.numContigs = len(self.indices)

            if self.numContigs == 0:
                print "    ERROR: No contigs loaded using condition:", condition
                return

            if(not silent):
                print "    Working with: %d contigs" % self.numContigs

            if(loadCovProfiles):
                if(verbose):
                    print "    Loading coverage profiles"
                self.covProfiles = self._DM.getCoverageProfiles(self.dbFileName, indices=self.indices)
                self.normCoverages = self._DM.getNormalisedCoverageProfiles(self.dbFileName, indices=self.indices)

            if(loadKmerSigs):
                if(verbose):
                    print "    Loading RAW kmer sigs"
                self.kmerSigs = self._DM.getKmerSigs(self.dbFileName, indices=self.indices)

            if(loadContigNames):
                if(verbose):
                    print "    Loading contig names"
                self.contigNames = self._DM.getContigNames(self.dbFileName, indices=self.indices)

            if(loadContigLengths):
                self.contigLengths = self._DM.getContigLengths(self.dbFileName, indices=self.indices)
                if(verbose):
                    print "    Loading contig lengths (Total: %d BP)" % ( sum(self.contigLengths) )

            if(loadContigGCs):
                self.contigGCs = self._DM.getContigGCs(self.dbFileName, indices=self.indices)
                if(verbose):
                    print "    Loading contig GC ratios (Average GC: %0.3f)" % ( numpy.mean(self.contigGCs) )

            if(loadBins):
                if(verbose):
                    print "    Loading bin assignments"
                self.binIds = self._DM.getBins(self.dbFileName, indices=self.indices)
            else:
                # we need zeros as bin indicies then...
                self.binIds = numpy.zeros(self.numContigs, dtype=int)

        except:
            print "Error loading DB:", self.dbFileName, sys.exc_info()[0]
            raise

        if(not silent):
            print "    %s" % timer.getTimeStamp()

    def setBinAssignments(self, assignments, nuke=False):
        """Save our bins into the DB"""
        self._DM.setBinAssignments(self.dbFileName,
                                   assignments,
                                   nuke=nuke)


    def promptOnOverwrite(self, minimal=False):
        """Check that the user is ok with possibly overwriting the DB"""
        if(self._DM.isClustered()):
            input_not_ok = True
            valid_responses = ['Y','N']
            vrs = ",".join([str.lower(str(x)) for x in valid_responses])
            while(input_not_ok):
                if(minimal):
                    option = raw_input(" Overwrite? ("+vrs+") : ")
                else:
                    option = raw_input(" ****WARNING**** Database: '"+self.dbFileName+"' has already been clustered.\n" \
                                       " If you continue you *MAY* overwrite existing bins!\n" \
                                       " Overwrite? ("+vrs+") : ")
                if(option.upper() in valid_responses):
                    print "****************************************************************"
                    if(option.upper() == "N"):
                        print "Operation cancelled"
                        return False
                    else:
                        break
                else:
                    print "Error, unrecognised choice '"+option.upper()+"'"
                    minimal = True
            print "Will Overwrite database",self.dbFileName
        return True


###############################################################################
#Utility functions
###############################################################################

def getConditionString(minLength=None, maxLength=None, bids=None):
    """Simple condition generation"""

    conds = []
    if(minLength):
        conds.append("(length >= %d)" % minLength)
    if(maxLength):
        conds.append("(length <= %d)" % maxLength)
    if(bids):
        conds.append(" | ".join(["(bid == %d)" % bid for bid in bids]))

    if len(conds) == 0:
        return ""
    else:
        return "(" + " & ".join(conds) + ")"

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

class FeaturePlotter:
    """Plot contigs in feature space"""
    COLOURS = 'rbgcmyk'

    def __init__(self, pm, colorMap="HSV"):
        self._pm = pm
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
                      x_label=x_label, y_label=y_lable,
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
        disp_cols = self._pm.contigGCs

        if highlight is not None:
            edgecolors=numpy.full_like(disp_cols, 'k', dtype=str)
            for (clr, hl) in zip(self.COLOURS, highlight):
                edgecolors[hl] = clr
            if keep is not None:
                edgecolors = edgecolors[keep]
        else:
            edgecolors = 'k'

        if plotContigLengths:
            disp_lens = numpy.sqrt(self._pm.contigLengths)
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
###############################################################################
###############################################################################
###############################################################################

