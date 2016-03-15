#!/usr/bin/env python
###############################################################################
#                                                                             #
#    import_.py                                                               #
#                                                                             #
#    Data import                                                              #
#                                                                             #
#    Copyright (C) Tim Lamberton                                              #
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

__author__ = "Tim Lamberton"
__copyright__ = "Copyright 2016"
__credits__ = ["Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"
__status__ = "Development"

###############################################################################
import sys
from profileManager import ProfileManager

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class BinImporter:
    """Used for importing bin assignments"""
    def __init__(self, dbFilename):
        self.dbFileName = dbFileName
        self._pm = ProfileManager(self.dbFilename)

    def loadData(self, timer):
        self._pm.loadData(timer)

    def readBinAssignments(self,
                           timer,
                           infile,
                           separator,
                           binField=None,
                           nameField=None,
                           noHeaders=False):
        """Parse fasta files for bin contigs"""
        self.loadData(timer)

        br = BinReader()
        # looks like cid->bid
        contig_bins = {}
        try:
            (con_names, con_bins) = binReader.parse(infile)
            contig_bins = dict(zip(con_names, con_bins))
        except:
            print "Could not parse bin assignment file:",infile,sys.exc_info()[0]
            raise

        # now get the internal indices for contigs
        row_bin_assignments = {}
        for (global_index, cid) in zip(self._pm.indices, self._pm.contigNames):
            try:
                row_bin_assignments[global_index] = contig_bins[cid]
            except IndexError:
                row_bin_assignment[global_index] = 0

        self._pm.setBinAssignments(row_bin_assignments, nuke=False)

        
class BinReader(CSVReader):   
    """Read a file of tab separated contig name and bin groupings."""
    def parse(self, infile):
        con_names = []
        con_bins = []
        with open(infile, "r") as f:
            for l in f:
                (cid, bid) = self.readCSV(f, separator)

                con_names.append(cid)
                con_bins.append(bid)
        
        return (con_names, con_bins)
        
        
class MarkerReader(CSVReader):
    """Read a file of tab delimited contig names, marker names and optionally classifications."""
    def parse(self, infile, doclassifications=False):
        con_names = []
        con_markers = []
        if doclassifications:
            con_taxstrings = []
           
        with open(infile, "r") as f:
            for l in f:
                fields = self.readCSV(f, separator)

                con_names.append(fields[0])
                con_markers.append(fields[1])
                if doclassifications:
                    if len(fields) > 2:
                        con_taxstrings.append(fields[2])
                    else:
                        con_taxstrings.append("")
        
        if doclassifications:
            return (con_names, con_markers, con_taxstrings)
        else:
            return (con_names, con_markers)
       
       
# Utility
class CSVReader:
    """Read tabular data from text files"""
    def readCSV(self, fp, separtor):
        for l in fp:
            yield line.rstrip().split(separator)
            

###############################################################################
###############################################################################
###############################################################################
###############################################################################
