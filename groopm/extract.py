#!/usr/bin/env python
###############################################################################
#                                                                             #
#    extract.py                                                               #
#                                                                             #
#    Classes for data extraction                                              #
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
import numpy as np
import scipy.spatial.distance as sp_distance
from bamm.bamExtractor import BamExtractor as BMBE

# local imports
from profileManager import ProfileManager
from binManager import BinManager
from data3 import ContigParser, MappingParser
from utils import makeSurePathExists
from cluster import MarkerCheckTreePrinter
import distance
import hierarchy


###############################################################################
###############################################################################
###############################################################################
###############################################################################

class BinExtractor:
    """Used for extracting reads and contigs based on bin assignments"""
    def __init__(self, dbFileName,
                 folder='',
                 ):
        self.dbFileName = dbFileName
        self._pm = ProfileManager(self.dbFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        makeSurePathExists(self._outDir)

    def loadProfile(self, timer, bids=[], cutoff=0):
        removeBins = bids is None or bids == []
        return self._pm.loadData(timer, 
                                 loadMarkers=False,
                                 loadBins=True,
                                 bids=[0] if removeBins else bids,
                                 removeBins=removeBins,
                                 minLength=cutoff
                                )

    def extractContigs(self,
                       timer,
                       bids=[],
                       fasta=[],
                       prefix='',
                       cutoff=0):
        """Extract contigs and write to file"""
        
        if prefix is None or prefix == '':
            prefix=os.path.basename(self.dbFileName) \
                            .replace(".gm", "") \
                            .replace(".sm", "")
                            
        profile = self.loadProfile(timer, bids, cutoff)
        bm = BinManager(profile)
        
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
                    cp.getWantedSeqs(f, profile.contigNames, out_dict=contigs)
        except:
            print "Could not parse contig file:",fasta[0],sys.exc_info()[0]
            raise

        # now print out the sequences
        print "Writing files"
        for bid in bm.getBids():
            file_name = os.path.join(self._outDir, "%s_bin_%d.fna" % (prefix, bid))
            try:
                with open(file_name, 'w') as f:
                    for cid in bm.profile.contigNames[bm.getBinIndices(bid)]:
                        if(cid in contigs):
                            f.write(">%s\n%s\n" % (cid, contigs[cid]))
                        else:
                            print "These are not the contigs you're looking for. ( %s )" % (cid)
            except:
                print "Could not open file for writing:",file_name,sys.exc_info()[0]
                raise

    def extractReads(self,
                     timer,
                     bids=[],
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
        profile = self.loadProfile(timer, bids)
        bm = BinManager(profile) # bins

        print "Extracting reads"

        # work out a set of targets to pass to the parser
        targets = []
        group_names = []
        for bid in bm.getBids():
            group_names.append("BIN_%d" % bid)
            row_indices = bm.getBinIndices(bid)
            targets.append(list(bm.profile.contigNames[row_indices]))

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

                           
class MarkerExtractor:
    def __init__(self,
                 dbFileName,
                 folder=''
                 ):
        self.dbFileName = dbFileName
        self._pm = ProfileManager(self.dbFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        makeSurePathExists(self._outDir)

    def loadProfile(self, timer, bids=[], cutoff=0):
        removeBins = bids is None or bids == []
        return self._pm.loadData(timer,
                                 loadBins=True,
                                 loadMarkers=True,
                                 loadTaxstrings=True,
                                 loadReachability=True,
                                 minLength=cutoff,
                                 bids=[0] if removeBins else bids,
                                 removeBins=removeBins,
                                )
        
    def extractMappingInfo(self,
                           timer,
                           bids=[],
                           prefix='',
                           separator='\t',
                           cutoff=0
                           ):
        """Extract markers from bins and write to file"""
        if prefix is None or prefix == '':
            prefix=os.path.basename(self.dbFileName) \
                            .replace(".gm", "") \
                            .replace(".sm", "")
        
        profile = self.loadProfile(timer, bids, cutoff)
        bm = BinManager(profile)
        mt = MarkerCheckTreePrinter(profile)
        
        # now print out the marker info
        print "Writing files"
        for bid in bm.getBids():
            file_name = os.path.join(self._outDir, "%s_bin_%d.txt" % (prefix, bid))
            
            bin_indices = bm.getBinIndices([bid])
            idx = np.flatnonzero(np.in1d(profile.mapping.rowIndices, bin_indices))
            
            labels = profile.mapping.markerNames[idx]
            cnames = profile.contigNames[profile.mapping.rowIndices[idx]]
            taxstrings = profile.mapping.taxstrings[idx]
            
            try:
                with open(file_name, 'w') as f:
                    #labels and lineages
                    f.write('#info table\n%s\n' % separator.join(['label', 'taxonomy', 'contig_name']))
                    for (label, taxstring, cname) in zip(labels, taxstrings, cnames):
                        f.write('%s\n' % separator.join([label, '\'%s\'' % taxstring, cname]))
                    
                    #distance table
                    f.write('\n#marker tree\n')
                    f.write(mt.printTree(profile.mapping.rowIndices[idx], leaves_list=bin_indices))
            except:
                print "Could not open file for writing:",file_name,sys.exc_info()[0]
                raise

                
                
class BinStatsDumper:
    def __init__(self,
                 dbFileName):
        self.dbFileName = dbFileName
        self._pm = ProfileManager(self.dbFileName)
        
    def loadProfile(self, timer):
        return self._pm.loadData(timer,
                                 loadMarkers=True,
                                 loadBins=True,
                                 bids=[0],
                                 removeBins=True,
                                )
                                
    def dumpBinStats(self,
                     timer,
                     fields,
                     outFile,
                     separator,
                     useHeaders
                    ):
        """Compute bin statistics"""
        
        # load all the contigs which have been assigned to bins
        profile = self.loadProfile(timer)
        bm = BinManager(profile)
        
        stats = bm.getBinStats()
        
        #data to output
        header_strings = []
        data_arrays = []
        data_converters = []
        
        for field in fields:
            if field == 'bins':
                header_strings.append('bid')
                data_arrays.append(stats.bids)
                data_converters.append(lambda x: str(x))
                
            elif field == 'points':
                header_strings.append('num_contigs')
                data_arrays.append(stats.numContigs)
                data_converters.append(lambda x: str(x))
            
            elif field == 'sizes':
                header_strings.append('size')
                data_arrays.append(stats.sizes)
                data_converters.append(lambda x: str(x))
                
            elif field == 'lengths':
                header_strings.append('length_min')
                header_strings.append('length_median')
                header_strings.append('length_max')
                data_arrays.append(stats.lengthRanges[:,0])
                data_arrays.append(stats.lengthMedians)
                data_arrays.append(stats.lengthRanges[:,1])
                data_converters.append(lambda x: str(x))
                data_converters.append(lambda x: str(x))
                data_converters.append(lambda x: str(x))
            
            elif field == 'gc':
                header_strings.append("GC%_mean")
                header_strings.append("GC%_std")
                data_arrays.append(stats.GCMeans)
                data_arrays.append(stats.GCStdDevs)
                data_converters.append(lambda x : "%0.4f" % x)
                data_converters.append(lambda x : "%0.4f" % x)
            
            elif field == 'coverage':
                stoits = profile.stoitNames
                header_strings.append(separator.join([separator.join([i + "_mean", i + "_std"]) for i in stoits]))
                interleaved = np.dsplit(np.transpose(np.dstack((stats.covMeans, stats.covStdDevs)), axes=[0, 2, 1]))
                data_arrays.append(interleaved)
                data_converters.append(lambda x: separator.join(["%0.4f"+separator+"%0.4f" % i for i in x]))

            elif field == 'tags':
                header_strings.append('taxonomy')
                data_arrays.append(stats.tags)
                data_converters.append(lambda x: x)
            
        # now print out the sequences
        try:
            with open(outFile, 'w') as f:
                if useHeaders:
                    header = separator.join(header_strings) + '\n'
                    f.write(header)
                
                num_rows = len(data_arrays[0])
                for i in range(num_rows):
                    row = separator.join([conv(arr[i]) for (conv, arr) in zip(data_converters, data_arrays)])
                    f.write(row+'\n')
        except:
            print "Could not open file for writing:",outFile,sys.exc_info()[0]
            raise
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
