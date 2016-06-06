#!/usr/bin/env python
###############################################################################
#                                                                             #
#    cluster.py                                                               #
#                                                                             #
#    A collection of classes / methods used when clustering contigs           #
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
__credits__ = ["Tim Lamberton", "Michael Imelfort"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################

import numpy as np
import numpy.linalg as np_linalg
import scipy.cluster.hierarchy as sp_hierarchy
import scipy.spatial.distance as sp_distance
import scipy.stats as sp_stats
import operator

# local imports
import distance
import recruit
import hierarchy
from binManager import BinManager
from profileManager import ProfileManager
from classification import ClassificationManager

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class CoreCreator:
    def __init__(self, dbFileName):
        self._pm = ProfileManager(dbFileName)
        self._dbFileName = dbFileName
        
    def loadProfile(self, timer, dsFileName, minLength, force):
        return self._pm.loadDistances(timer,
                                      dsFileName,
                                      minLength=minLength,
                                      loadMarkers=True,
                                      loadBins=False,
                                      force=force)
        
    def run(self,
            timer,
            minLength,
            minSize,
            minPts,
            useDsFile=None,
            newDsFile=None,
            force=False):
        # check that the user is OK with nuking stuff...
        if not force and not self._pm.promptOnOverwrite():
            return
        
        if useDsFile is not None:
            dsFileName = useDsFile
            force_dists = False
            keep_dists = True
        elif newDsFile is not None:
            dsFileName = newDsFile
            force_dists = True
            keep_dists = True
        else:
            dsFileName = self._dbFileName+".ds"
            force_dists = True
            keep_dists = False
            
        profile = self.loadProfile(timer,
                                   dsFileName,
                                   minLength=minLength,
                                   force=force_dists
                                   )
        
        ce = ClassificationClusterEngine(profile)
        ce.makeBins(timer, out_bins=profile.binIds, out_reach_pos=profile.reachPos, out_reach_dist=profile.reachDist)
        
        bm = BinManager(profile)
        bm.unbinLowQualityAssignments(out_bins=profile.binIds, minSize=minSize, minPts=minPts)

        # Now save all the stuff to disk!
        print "Saving bins"
        self._pm.setBinAssignments(profile, nuke=True)
        print "    %s" % timer.getTimeStamp()
        
        if not keep_dists:
            self._pm.cleanUpDistances(dsFileName)
        
        
# Hierarchical clustering
class HierarchicalClusterEngine:
    """Hierarchical clustering algorthm"""
    def makeBins(self, timer, out_bins, out_reach_pos, out_reach_dist):
        """Run binning algorithm"""
        
        print "Computing cluster hierarchy"
        dists = self.distances()
        print "Clustering 2^%f.2 pairs" % np.log2(len(dists))
        (o, d) = distance.reachability_order(dists)
        Z = hierarchy.linkage_from_reachability(o, d)
        print "    %s" % timer.getTimeStamp()
        
        print "Finding cores"
        T = self.fcluster(Z)
        out_bins[...] = T+1 #bins start from 1
        out_reach_pos[...] = o
        out_reach_dist[...] = d
        print "    %s bins made." % len(set(out_bins).difference([0]))
        print "    %s" % timer.getTimeStamp()
            
    def distances(self):
        """computes pairwise distances of observations"""
        pass #subclass to override
        
    def fcluster(self, Z):
        """finds flat clusters from linkage matrix"""
        pass #subclass to override
        
        
class ClassificationClusterEngine(HierarchicalClusterEngine):
    """Cluster using hierarchical clusturing with feature distance ranks and marker taxonomy"""
    def __init__(self, profile):
        self._profile = profile
    
    def distances(self):
        return self._profile.distances.denDists
    
    def fcluster(self, Z):
        cf = ClassificationManager(self._profile.mapping)
        return hierarchy.fcluster_coeffs(Z,
                                         dict(self._profile.mapping.iterindices()),
                                         cf.disagreement)
     
###############################################################################
###############################################################################
###############################################################################
###############################################################################
   
# Mediod clustering
class MediodsClusterEngine:
    """Iterative mediod clustering algorithm"""
    
    def makeBins(self, timer, init, out_bins):
        """Run binning algorithm
        
        Parameters
        ----------
        init : ndarray
            Array of indices used to determine starting points for new
            clusters.
        out_bins: ndarray
            1-D array of initial bin ids. An id of 0 is considered unbinned. 
            The bin id for the `i`th original observation will be stored in
            `out_bins[i]`.
        """
        
        bin_counter = np.max(out_bins)
        mediod = None
        queue = init

        while(True):
            if mediod is None:
                if len(queue) == 0:
                    break
                mediod = queue.pop()
                if out_bins[mediod] != 0:
                    mediod = None
                    continue
                round_counter = 0
                bin_counter += 1
                out_binds[mediod] = bin_counter

            round_counter += 1
            print "    Recruiting bin %d, round %d." % (bin_counter, round_counter)
            
            is_unbinned = out_bins == 0
            print "    Found %d unbinned." % np.count_nonzero(is_unbinned)

            is_old_members = out_bins == bin_counter
            putative_members = np.flatnonzero(np.logical_and(is_unbinned, is_old_members))
            recruited = self.recruit(mediod, putative_members=putative_members)
            
            out_bins[recruited] = bin_counter
            members = np.flatnonzero(out_bins == bin_counter)
            
            print "   Recruited %d members." % (members.size - old_members.size)
            
            if len(members)==1:
                new_mediod = members
            else:
                index = self.mediod(members)
                new_mediod = members[index]


            if new_mediod == mediod:
                print "    Mediod is stable after %d rounds." % round_counter
                mediod = None
            else:
                mediod = new_mediod

        print "    %d bins made." % bin_counter
        print "    %s" % timer.getTimeStamp()
        
    def recruit(self, mediod, putative_members):
        """recruit contigs close to a mediod contig"""
        pass #subclass to override
        
    def mediod(self, indices):
        """computes pairwise distances of observations"""
        pass #subclass to override
        
        
class CorrelationClusterEngine(MediodsClusterEngine):
    """Cluster using mediod feature distance rank correlation"""
    def __init__(self, profile, threshold=0.5):
        self._profile = profile
        self._features = (profile.covProfiles, profile.kmerSigs)
        self._threshold = threshold
        
    def feature_ranks(self, indices):
        return tuple(distance.argrank(sp_distance.cdist(f[indices], f, metric="euclidean"), axis=1) for f in self._features)
        
    def mediod(self, indices):
        intracluster_dists = tuple(dm[:, indices] for dm in self.feature_ranks(indices))
        return np_linalg.norm(intracluster_dists, axis=0).sum(axis=1).argmin()
        
    def recruit(self, origin, putative_members):
        (covRanks, kmerRanks) = tuple(dm[0] for dm in self.feature_ranks([origin]))
        return recruit.getMergers((covRanks, kmerRanks), threshold=self._threshold, unmerged=putative_members)
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################

   
