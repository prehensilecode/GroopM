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
import numpy as np

from utils import CSVReader
from profileManager import ProfileManager

###############################################################################
###############################################################################
###############################################################################
###############################################################################
class BinImporter:
    """Used for importing bin assignments"""
    def __init__(self,
                 dbFileName):
        self._pm = ProfileManager(dbFileName)
        
    def loadProfile(self, timer):
        return self._pm.loadData(timer)
        
    def importBinAssignments(self,
                             timer,
                             infile,
                             separator):
        """Parse assignment file for bin contigs"""
        
        profile = self.loadProfile(timer)
        br = BinReader()
        # looks like cid->bid
        contig_bins = {}
        try:
            with open(infile, "r") as f:
                try:
                    (con_names, con_bins) = br.parse(f, separator)
                    (_, con_bid) = np.unique(con_bins, return_inverse=True)
                    con_bid += 1 # bid zero is unbinned
                    contig_bins = dict(zip(con_names, con_bid))
                except:
                    print "Error parsing bin assignments"
                    raise
        except:
            print "Could not parse bin assignment file:",infile,sys.exc_info()[0]
            raise

        # now get the internal indices for contigs
        for (i, cid) in enumerate(profile.contigNames):
            try:
                profile.binIds[i] = contig_bins[cid]
            except KeyError:
                pass
        
        # Now save all the stuff to disk!
        print "Saving bins"
        self._pm.setBinAssignments(profile, nuke=True)
        print "    %s" % timer.getTimeStamp()

        
class BinReader:   
    """Read a file of tab separated contig name and bin groupings."""
    def parse(self, fp, separator):
        con_names = []
        con_bins = []
        
        reader = CSVReader()
        for (cid, bid) in reader.readCSV(fp, separator):
            con_names.append(cid)
            con_bins.append(bid)
        
        return (con_names, con_bins)    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
