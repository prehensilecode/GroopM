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

__author__ = "Michael Imelfort, Tim Lamberton"
__copyright__ = "Copyright 2012/2013"
__credits__ = ["Michael Imelfort", "Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################
import numpy as np
import scipy.spatial.distance as sp_distance
import sys
import os

# GroopM imports
from data3 import DataManager, ClassificationEngine
from utils import group_iterator
from groopmExceptions import ContigNotFoundException
import distance

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class _Classification:
    """
    Class for carrying gene taxonomic classification data around, constructed 
    using ProfileManager class.
        
    Fields
    ------
    # data
    _table: ndarray
        n-by-7 array where n is the number of mappings. `table[i]` contains
        indices into the `taxons` array corresponding to the taxon with the
        corresponding ranks for each column:
            0 - Domain
            1 - Phylum
            2 - Class
            3 - Order
            4 - Family
            5 - Genus
            6 - Species
    
    # metadata
    _taxons: ndarray
        Array of taxonomic classification strings.
    """
    _ce = ClassificationEngine()
    
    def tags(self, index):
        """Return a list of taxonomic tags"""
        return [t+self._taxons[i] for (t, i) in zip(self._ce.TAGS, self._table[index]) if i!=0]
        
    def makeDistances(self):
        return sp_distance.pdist(self._table, self._ce.getDistance)
        
    def getPrefixed(self, taxstring):
        """Return indices of taxonomies that have the search taxonomy as a prefix"""
        ranks = self._ce.parse_taxstring(taxstring)
        taxon_ids = np.flatnonzero(np.in1d(self._taxons, ranks))
        taxon_dict = dict(zip(self._taxons[taxon_ids], taxon_ids))
        row = np.zeros(len(self._ce.TAGS), dtype=int)
        for (j, rank) in enumerate(ranks):
            try:
                row[j] = taxon_dict[rank]
            except KeyError:
                row[j] = 1 # novel ranks are set to 1's (match no other tags)
        
        return np.flatnonzero(np.apply_along_axis(self._ce.isPrefix, 0, self._table, row))
        

        
        
        
class _Mappings:
    """Class for carrying gene mapping data around, constructed using
    ProfileManager class.
    
    Fields
    ------
    # mapping data
    rowIndices : ndarray
        `rowIndices[i]` is the row index in `profile` of mapping `i`.
    mapIndices : ndarray
        `indices[i]` is the index into the pytables structure of mapping `i`
    markerNames : ndarray
        `markerNames[i]` is the marker id for mapping `i`.
    taxstrings : ndarray
        `taxstrings[i]` is the taxstring for mapping `i`.
    classification : _Classificaton object
        See above.
        
    #metadata
    numMappings : int
        Corresponds to the lengths of above arrays.
    """
    
    def itergroups(self):
        """Returns an iterator of marker group ids and indices."""
        return group_iterator(self.markerNames)
        
    def iterindices(self):
        """Returns an iterator of profile and marker indices."""
        return group_iterator(self.rowIndices)
        
    def makeDistances_(self):
        """"Distance matrix using shared group membership"""
        n = self.numMappings
        Y = np.zeros(n*(n-1)//2, dtype=bool)
        
        # larger distance between mappings to the same single copy marker
        for (_, m) in self.itergroups():
            m = np.array(m)
            (i, j) = distance.pairs(len(m))
            Y[distance.condensed_index(n, m[i], m[j])] = True
            
        return Y
                 
    def makeConnectivity_(self, d):
        """Connectivity matrix to specified distance"""
        dm = sp_distance.squareform(self.classification.makeDistances() <= d)
        
        # disconnect mappings to the same single copy marker
        for (_, m) in self.itergroups():
            dm[np.ix_(m, m)] = False
            dm[m, m] = True 
        
        return dm
        
class _Profile:
    """Class for carrying profile data around, construct using ProfileManager class.
    
    Fields
    ------
    # contig data
    indices : ndarray
        `indices[i]` is the index into the pytables data structure of contig `i`.
    covProfile : ndarray
        `covProfile[i, j]` is the trimmed mean coverage of contig `i` in stoit `j`.
    kmerSigs : ndarray
        `kmerSigs[i, j]` is the relative frequency of occurrence in contig `i` of
        the `j`th kmer.
    normCoverages : ndarray
        `normCoverage[i]` is the norm of all stoit coverages for contig `i`.
    contigGCs : ndarray
        `contigGCs[i]` is the percentage GC for contig `i`.
    contigNames : ndarray
        `contigNames[i]` is the fasta contig id of contig `i`.
    contigLengths : ndarray
        `contigLengths[i]` is the length in bp of contig `i`.
    binIds : ndarray
        `binIds[i]` is the bin id assigned to contig `i`.
        
    
    # metadata
    numContigs : int
        Number of contigs, corresponds to length of axis 1 in above arrays.
    stoitNames : ndarray
        Names of stoits for each column of covProfiles array.
    numStoits : int
        Corresponds to number of columns of covProfiles array.
    mapping : _Mappings object
        See above.
    reachOrder : ndarray
        `reachOrder[i]` is the contig index in position `i` of reachability order.
    reachDists : ndarray
        `reachDists[i]` is the reachability distance of position `i`.
    """
    
    
class ProfileManager:
    """Interacts with the groopm DataManager and local data fields

    Mostly a wrapper around a group of numpy arrays and a pytables quagmire
    """
    
    def __init__(self, dbFileName):
        # misc
        self.dbFileName = dbFileName         # db containing all the data we'd like to use

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
                 loadReachability=False,
                 loadMarkers=True,
                 loadTaxstrings=False,
                 minLength=None,
                 bids=[],
                 removeBins=False
                ):
        """Load pre-parsed data"""

        if(silent):
            verbose=False
        if verbose:
            print "Loading data from:", self.dbFileName
            
        dm = DataManager()
        dm.checkAndUpgradeDB(self.dbFileName, timer, silent=silent)
        try:
            prof = _Profile()
                 
            # Collect contig data
            if(loadReachability):
                (prof.indices, prof.reachDists) = dm.getReachabilityOrder(self.dbFileName)
                if(verbose):
                    print "    Loaded previously clustered contigs in reachability ordering"
                prof.numContigs = len(prof.indices)
                prof.reachOrder = np.arange(prof.numContigs)
                                
                if prof.numContigs == 0:
                    print "    WARNING: No previously clustered contigs. Please run `core` step before proceeding."
                    return prof
            else:
                # Conditional filter
                condition = _getConditionString(minLength=minLength, bids=bids, removeBins=removeBins)
                prof.indices = dm.getConditionalIndices(self.dbFileName,
                                                    condition=condition)
                if(verbose):
                    print "    Loaded indices with condition:", condition
                    
                prof.numContigs = len(prof.indices)
                prof.reachOrder = np.zeros(prof.numContigs, dtype=int)
                prof.reachDists = np.zeros(prof.numContigs, dtype=float)

                if prof.numContigs == 0:
                    print "    WARNING: No contigs loaded using condition:", condition
                    return prof

            if(not silent):
                print "    Working with: %d contigs" % prof.numContigs

            if(loadCovProfiles):
                if(verbose):
                    print "    Loading coverage profiles"
                prof.covProfiles = dm.getCoverages(self.dbFileName, indices=prof.indices)
                prof.normCoverages = dm.getNormCoverages(self.dbFileName, indices=prof.indices)

            if(loadKmerSigs):
                if(verbose):
                    print "    Loading kmer sigs"
                prof.kmerSigs = dm.getKmerSigs(self.dbFileName, indices=prof.indices)
                
            if(loadContigNames):
                if(verbose):
                    print "    Loading contig names"
                prof.contigNames = dm.getContigNames(self.dbFileName, indices=prof.indices)

            if(loadContigLengths):
                prof.contigLengths = dm.getContigLengths(self.dbFileName, indices=prof.indices)
                if(verbose):
                    print "    Loading contig lengths (Total: %d BP)" % ( sum(prof.contigLengths) )

            if(loadContigGCs):
                prof.contigGCs = dm.getContigGCs(self.dbFileName, indices=prof.indices)
                if(verbose):
                    print "    Loading contig GC ratios (Average GC: %0.3f)" % ( np.mean(prof.contigGCs) )

            if(loadBins):
                if(verbose):
                    print "    Loading bin assignments"
                prof.binIds = dm.getBins(self.dbFileName, indices=prof.indices)
            else:
                # we need zeros as bin indicies then...
                prof.binIds = np.zeros(prof.numContigs, dtype=int)

            if(loadMarkers):
                if verbose:
                    print "    Loading marker data"
                map_indices = dm.getMappingContigs(self.dbFileName)
                map_markers = dm.getMappingMarkers(self.dbFileName)
                
                indices_2_rows = dict(zip(prof.indices, range(prof.numContigs)))
                map_row_indices = []
                map_keep = []
                for (i, index) in enumerate(map_indices):
                    try:
                        row = indices_2_rows[index]
                    except KeyError:
                        continue
                    map_row_indices.append(row)
                    map_keep.append(i)
                    
                markers = _Mappings()
                markers.rowIndices = np.array(map_row_indices)
                markers.indices = np.array(map_keep, dtype=int)
                
                if verbose:
                    print "    Loading marker names"
                marker_names = dm.getMarkerNames(self.dbFileName)
                markers.markerNames = marker_names[map_markers][markers.indices]
                markers.numMappings = len(markers.indices)
                
                if loadTaxstrings:
                    if verbose:
                        print "    Loading marker taxonomies"
                    taxstrings = dm.getMappingTaxstrings(self.dbFileName)
                    markers.taxstrings = taxstrings[markers.indices]
                
                classif = _Classification()
                
                if verbose:
                    print "    Loading marker classifications"
                map_table = dm.getClassification(self.dbFileName)
                classif._table = map_table[map_keep]
                
                if verbose:
                    print "    Loading marker taxons"
                classif._taxons = dm.getTaxonNames(self.dbFileName)
                
                markers.classification = classif
                prof.mapping = markers
                
            # Stoit names
            prof.numStoits = dm.getNumStoits(self.dbFileName)
            if(loadStoitNames):
                print "    Loading stoit names"
                prof.stoitNames = np.array(dm.getCovColNames(self.dbFileName).split(","))
            
        except:
            print "Error loading DB:", self.dbFileName, sys.exc_info()[0]
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
        DataManager().setBinAssignments(self.dbFileName,
                                        assignments,
                                        nuke=nuke)
                                   
    def setReachabilityOrder(self, profile):
        """Save mapping distances
        
        dataManager.setReachabilityOrder needs GLOBAL indices
        [(global_index, distance)]
        """
        updates = zip(profile.indices[profile.reachOrder], profile.reachDists)
        DataManager().setReachabilityOrder(self.dbFileName, updates)

    def promptOnOverwrite(self, minimal=False):
        """Check that the user is ok with possibly overwriting the DB"""
        if(DataManager().isClustered(self.dbFileName)):
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

        
#------------------------------------------------------------------------------
# Helpers
def _getConditionString(minLength=None, maxLength=None, bids=None, removeBins=False):
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