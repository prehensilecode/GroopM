#!/usr/bin/env python
###############################################################################
#                                                                             #
#    groopmUtils.py                                                           #
#                                                                             #
#    Classes for non-clustering data manipulation and output                  #
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
__version__ = "0.2.11"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"
__status__ = "Development"

###############################################################################
import os
import sys
import errno
import numpy

# GroopM imports
from profileManager import ProfileManager, FeaturePlotter
from binManager import BinManager
from coverageAndKmerDistance import CoverageAndKmerDistanceTool, CoverageAndKmerView
from mstore import ContigParser
from corre import getInsidePNull
from cluster import getNearPNull

# other local imports
from bamm.bamExtractor import BamExtractor as BMBE

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

#------------------------------------------------------------------------------
#Extraction

class GMExtractor:
    """Used for extracting reads and contigs based on bin assignments"""
    def __init__(self, dbFilename,
                 bids=[],
                 folder='',
                 ):
        self.dbFileName = dbFileName
        self._pm = ProfileManager(self.dbFilename)
        self._bids = [] if bids is None else bids
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        makeSurePathExists(self._outDir)

    def loadData(self, timer, cutoff=0):
        self._pm.loadData(timer,
                          loadBins=True,
                          bids=self._bids,
                          minLength=cutoff
                         )

    def extractContigs(self,
                       timer,
                       fasta=[],
                       prefix='',
                       cutoff=0):
        """Extract contigs and write to file"""
        self.loadData(timer, cutoff=cutoff)
        bm = BinManager(self._pm)
        if prefix is None or prefix == '':
            prefix=os.path.basename(self.dbFileName) \
                            .replace(".gm", "") \
                            .replace(".sm", "")

        # load all the contigs which have been assigned to bins
        cp = ContigParser()
        # contigs looks like cid->seq
        contigs = {}
        import mimetypes
        try:
            for file_name in fasta:
                gm_open = open
                try:
                    # handle gzipped files
                    mime = mimetypes.guess_type(file_name)
                    if mime[1] == 'gzip':
                        import gzip
                        gm_open = gzip.open
                except:
                    print "Error when guessing contig file mimetype"
                    raise
                with gm_open(file_name, "r") as f:
                    contigs = cp.getWantedSeqs(f, self._pm.contigNames, storage=contigs)
        except:
            print "Could not parse contig file:",fasta[0],sys.exc_info()[0]
            raise

        # now print out the sequences
        print "Writing files"
        for bid in bm.getBids():
            file_name = os.path.join(self._outDir, "%s_bin_%d.fna" % (prefix, bid))
            try:
                with open(file_name, 'w') as f:
                    for cid in self._pm.contigNames[bm.getBinIndices(bid)]:
                        if(cid in contigs):
                            f.write(">%s\n%s\n" % (cid, contigs[cid]))
                        else:
                            print "These are not the contigs you're looking for. ( %s )" % (cid)
            except:
                print "Could not open file for writing:",file_name,sys.exc_info()[0]
                raise

    def extractReads(self,
                     timer,
                     bams=[],
                     prefix="",
                     mixBams=False,
                     mixGroups=False,
                     mixReads=False,
                     interleaved=False,
                     bigFile=False,
                     headersOnly=False,
                     minMapQual=0,
                     maxMisMatches=1000,
                     useSuppAlignments=False,
                     useSecondaryAlignments=False,
                     threads=1,
                     verbose=False):
        """Extract reads from bam files and write to file

        All logic is handled by BamM <- soon to be wrapped by StoreM"""
        # load data
        self.loadData()
        bm = BinManager(self._pm)   # bins

        print "Extracting reads"

        # work out a set of targets to pass to the parser
        targets = []
        group_names = []
        for bid in bm.getBids():
            group_names.append("BIN_%d" % bid)
            row_indices = bm.getBinIndices(bid)
            targets.append(list(self._pm.contigNames[row_indices]))

        # get something to parse the bams with
        bam_parser = BMBE(targets,
                          bams,
                          groupNames=group_names,
                          prefix=prefix,
                          outFolder=self._outDir,
                          mixBams=mixBams,
                          mixGroups=mixGroups,
                          mixReads=mixReads,
                          interleaved=interleaved,
                          bigFile=bigFile,
                          headersOnly=headersOnly,
                          minMapQual=minMapQual,
                          maxMisMatches=maxMisMatches,
                          useSuppAlignments=useSuppAlignments,
                          useSecondaryAlignments=useSecondaryAlignments)

        bam_parser.extract(threads=threads,
                           verbose=verbose)


#------------------------------------------------------------------------------
#Plotting

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
        index = numpy.argmax(pm.normCoverages[members])
    elif mode=="max_length":
        index = numpy.argmax(pm.contigLengths[members])
    else:
        raise ValueError("Invalid mode: %s" % mode)

    return members[index]

def getSurface(mode, ranks):
    """Computes derived surface in hybrid measure space"""
    if mode=="corr_inside":
        z = numpy.log10(getInsidePNull(ranks))
        z_label = "Inside correlation"
    elif mode=="corr_near":
        z = numpy.log10(getNearPNull(ranks))
        z_label = "Outside correlation"
    else:
        raise ValueError("Invaild mode: %s" % mode)

    return (z, z_label)

def makeSurePathExists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

###############################################################################
###############################################################################
###############################################################################
###############################################################################
