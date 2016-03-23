#!/usr/bin/env python
###############################################################################
#                                                                             #
#    profileManager.py                                                        #
#                                                                             #
#    GroopM - High level data management                                      #
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

__author__ = "Michael Imelfort"
__copyright__ = "Copyright 2012/2013"
__credits__ = ["Michael Imelfort", "Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################
import numpy as np
import sys

# GroopM imports
from mstore import GMDataManager as DataManager

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
        
        
class Mapping:
    """Class for carrying gene mapping data around, constructed using
    ProfileManager class.
    
    Fields
    ------
    # mapping data
    profile: ndarray
        Profile instance, see ProfileManager class.
    rowIndices: ndarray
        `rowIndices[i]` is the row index in `profile` of the `i`th mapping.
    markerNames: ndarray
        `markerNames[i]` is the marker gene name hit for the `i`th mapping.
    taxstrings: ndarray
        `taxstrings[i]` is a taxonomic classification string for the `i`th 
        mapping.
        
    #metadata
    numMappings: int
        Corresponds to the lengths of above arrays.
    """
    pass
        
class Profile:
    """Class for carrying profile data around, construct using ProfileManager class.
    
    Fields
    ------
    # contig data
    indices: ndarray
        `indices[i]` is the index into the pytables data structure of contig `i`.
    covProfile: ndarray
        `covProfile[i, j]` is the trimmed mean coverage of contig `i` in stoit `j`.
    kmerSigs: ndarray
        `kmerSigs[i, j]` is the relative frequency of occurrence in contig `i` of
        the `j`th kmer.
    normCoverages: ndarray
        `normCoverage[i]` is the norm of all stoit coverages for contig `i`.
    contigGCs: ndarray
        `contigGCs[i]` is the percentage GC for contig `i`.
    contigNames: ndarray
        `contigNames[i]` is the fasta contig id of contig `i`.
    contigLengths: ndarray
        `contigLengths[i]` is the length in bp of contig `i`.
    binIds: ndarray
        `binIds[i]` is the bin id assigned to contig `i`.
        
    
    # metadata
    numContigs: int
        Number of contigs, corresponds to length of axis 1 in above arrays.
    stoitNames: ndarray
        Names of stoits for each column of covProfiles array.
    numStoits: int
        Corresponds to number of columns of covProfiles array.
    markers: Mapping
        Instance of Mapping class for marker gene hits
    """
    pass

    
class ProfileManager:
    """Interacts with the groopm DataManager and local data fields

    Mostly a wrapper around a group of numpy arrays and a pytables quagmire
    """
    def __init__(self, dbFileName, markerFileName=None):
        # misc
        self._dm = DataManager()            # most data is saved to hdf
        self.dbFileName = dbFileName         # db containing all the data we'd like to use
        self.markerFileName = markerFileName

    def loadData(self,
                 timer,
                 verbose=True,              # many to some output messages
                 silent=False,              # some to no output messages
                 loadCovProfiles=True,
                 loadKmerSigs=True,
                 #loadKmerPCs=False,
                 loadStoitNames=True,
                 loadContigNames=True,
                 loadContigLengths=True,
                 loadContigGCs=True,
                 loadBins=False,
                 loadMarkers=False,
                 minLength=None,
                 bids=[],
                 removeBins=False
                ):
        """Load pre-parsed data"""

        if(silent):
            verbose=False
        if verbose:
            print "Loading data from:", self.dbFileName

        try:
            prof = Profile()
            
            # Stoit names
            prof.numStoits = self._dm.getNumStoits(self.dbFileName)
            if(loadStoitNames):
                prof.stoitNames = np.array(self._dm.getStoitColNames(self.dbFileName).split(","))

            # Conditional filter
            condition = getConditionString(minLength=minLength, bids=bids, removeBins=removeBins)
            prof.indices = self._dm.getConditionalIndices(self.dbFileName,
                                                          condition=condition,
                                                          silent=silent)

            # Collect contig data
            if(verbose):
                print "    Loaded indices with condition:", condition
            prof.numContigs = len(prof.indices)

            if prof.numContigs == 0:
                print "    ERROR: No contigs loaded using condition:", condition
                return

            if(not silent):
                print "    Working with: %d contigs" % prof.numContigs

            if(loadCovProfiles):
                if(verbose):
                    print "    Loading coverage profiles"
                prof.covProfiles = self._dm.getCoverageProfiles(self.dbFileName, indices=prof.indices)
                prof.normCoverages = self._dm.getNormalisedCoverageProfiles(self.dbFileName, indices=prof.indices)

            if(loadKmerSigs):
                if(verbose):
                    print "    Loading RAW kmer sigs"
                prof.kmerSigs = self._dm.getKmerSigs(self.dbFileName, indices=prof.indices)

            if(False):
                prof.kmerPCs = self._dm.getKmerPCAs(self.dbFileName, indices=prof.indices)

                if(verbose):
                    print "    Loading PCA kmer sigs (" + str(len(prof.kmerPCs[0])) + " dimensional space)"

            if(loadContigNames):
                if(verbose):
                    print "    Loading contig names"
                prof.contigNames = self._dm.getContigNames(self.dbFileName, indices=prof.indices)

            if(loadContigLengths):
                prof.contigLengths = self._dm.getContigLengths(self.dbFileName, indices=prof.indices)
                if(verbose):
                    print "    Loading contig lengths (Total: %d BP)" % ( sum(prof.contigLengths) )

            if(loadContigGCs):
                prof.contigGCs = self._dm.getContigGCs(self.dbFileName, indices=prof.indices)
                if(verbose):
                    print "    Loading contig GC ratios (Average GC: %0.3f)" % ( np.mean(prof.contigGCs) )

            if(loadBins):
                if(verbose):
                    print "    Loading bin assignments"
                prof.binIds = self._dm.getBins(self.dbFileName, indices=prof.indices)
            else:
                # we need zeros as bin indicies then...
                prof.binIds = np.zeros(prof.numContigs, dtype=int)

        except:
            print "Error loading DB:", self.dbFileName, sys.exc_info()[0]
            raise
             
        if(loadMarkers):
            if verbose:
                print "Loading marker data from:", self.markerFileName
            
            try:
                markers = Mapping()
                reader = MappingReader()
                (con_names, con_markers, con_taxstrings) = reader.parse(markerFile, True)
                
                lookup = dict(zip(prof.contigNames, np.arange(prof.numContigs)))
                rowIndices = np.array([lookup[name] for name in con_names])
                
                markers.rowIndices = rowIndices
                markers.markerNames = np.asarray(con_markers)
                markers.taxstrings = np.asarray(con_taxstrings)
                markers.numMappings = len(con_names)
                
                prof.markers = markers
            except:
                print "Error opening marker file:", self.markerFileName, sys.exc_info()[0]
                raise
                

        if(not silent):
            print "    %s" % timer.getTimeStamp()
            
        return prof

    def setBinAssignments(self, profile, nuke=False):
        """Save bins into the DB
        
        dataManager.setBinAssignments needs GLOBAL row indices
        { global_index : bid }
        """
        assignments = dict(zip(profile.indices, profile.binIds))
        self._dm.setBinAssignments(self.dbFileName,
                                   assignments,
                                   nuke=nuke)

    def promptOnOverwrite(self, minimal=False):
        """Check that the user is ok with possibly overwriting the DB"""
        if(self._dm.isClustered()):
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


class MappingReader:
    """Read a file of tab delimited contig names, marker names and optionally classifications."""
    def parse(self, infile, doclassifications=False):
        con_names = []
        con_markers = []
        if doclassifications:
            con_taxstrings = []
           
        reader = CSVReader()
        with open(infile, "r") as f:
            for l in f:
                fields = reader.readCSV(f, "\t")

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
        
###############################################################################
#Utility functions
###############################################################################

def getConditionString(minLength=None, maxLength=None, bids=None, removeBins=False):
    """Simple condition generation"""

    conds = []
    if minLength is not None:
        conds.append("(length >= %d)" % minLength)
    if maxLength is not None:
        conds.append("(length <= %d)" % maxLength)
    if bids is not None and len(bids) > 0:
        if removeBins:
            conds.append(" | ".join(["(bid != %d)" % bid for bid in bids]))
        else:
            conds.append(" | ".join(["(bid == %d)" % bid for bid in bids]))

    if len(conds) == 0:
        return ""
    else:
        return "(" + " & ".join(conds) + ")"

###############################################################################
###############################################################################
###############################################################################
###############################################################################

