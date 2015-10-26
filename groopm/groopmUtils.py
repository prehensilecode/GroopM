#!/usr/bin/env python
###############################################################################
#                                                                             #
#    groopmUtils.py                                                           #
#                                                                             #
#    Classes for non-clustering data manipulation and output                  #
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
__version__ = "0.2.11"
__maintainer__ = "Michael Imelfort"
__email__ = "mike@mikeimelfort.com"
__status__ = "Released"

###############################################################################
import os
import sys
import errno
import numpy

# GroopM imports
from profileManager import ProfileManager, FeaturePlotter
from binManager import BinManager
from cluster import BinHighlightAPI
from mstore import ContigParser

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

    def loadData(self, timer, cutoff=0):
        self._pm.loadData(timer, loadBins=True, minLength=cutoff)

    def plot(self,
             timer,
             bids,
             origin_mode="mediod",
             highlight_mode="mergers",
             threshold=None,
             plotRanks=False,
             colorMap="HSV",
             prefix="BIN"
            ):
        self.loadData(timer)
        bm = BinManager(self._pm)
        fplot = FeaturePlotter(self._pm, colorMap=colorMap)
        binHighlightApi = BinHighlightAPI(self._pm)

        bm.checkBids(bids)
        for bid in bids:
            fileName = "" if self._outDir is None else os.path.join(self._outDir, "%s_%d.png" % (prefix, bid))
            fplot.plot(**getPlotArgs(**binHighlightApi(bid=bid,
                                                       origin_mode=origin_mode,
                                                       highlight_mode=highlight_mode,
                                                       threshold=threshold,
                                                       plotRanks=plotRanks,
                                                       fileName=fileName)))
            if self._outDir is None:
                break

        print "    %s" % timer.getTimeStamp()


###############################################################################
# Helpers
###############################################################################

def getPlotArgs(data, ranks, plotRanks=False, **kwargs):
    (x, y) = (ranks[0], ranks[1]) if plotRanks else (data[0], data[1])
    kwargs["x"] = x
    kwargs["y"] = y
    return kwargs

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
