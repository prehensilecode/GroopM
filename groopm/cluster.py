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
import os

# local imports
import distance
import recruit
import hierarchy
from utils import split_contiguous
from binManager import BinManager
from profileManager import ProfileManager
from classification import ClassificationManager
from groopmExceptions import SavedDistancesInvalidNumberException

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class CoreCreator:
    """Top level class for making bins"""
    
    def __init__(self, dbFileName):
        self._pm = ProfileManager(dbFileName)
        self._dbFileName = dbFileName
        
    def loadProfile(self, timer, minLength):
        return self._pm.loadData(timer,
                                 minLength=minLength,
                                 loadMarkers=True,
                                 loadBins=False)
        
    def run(self,
            timer,
            minLength,
            minSize,
            minPts,
            #savedDistsPrefix="",
            #keepDists=False,
            force=False):
        # check that the user is OK with nuking stuff...
        if not force and not self._pm.promptOnOverwrite():
            return
            
        profile = self.loadProfile(timer,
                                   minLength=minLength
                                   )
        
        
        ce = ClassificationClusterEngine(profile,
                                         minPts=minPts,
                                         minSize=minSize
                                         )
        ce.makeBins(timer,
                    out_bins=profile.binIds,
                    out_reach_order=profile.reachOrder,
                    out_reach_dists=profile.reachDists
                    )

        # Now save all the stuff to disk!
        print "Saving bins"
        self._pm.setReachabilityOrder(profile)
        self._pm.setBinAssignments(profile, nuke=True)
        print "    %s" % timer.getTimeStamp()
            
        
        
# Hierarchical clustering
class HierarchicalClusterEngine:
    """Hierarchical clustering algorthm"""
    
    def makeBins(self, timer, out_bins, out_reach_order, out_reach_dists):
        """Run binning algorithm"""
        
        print "Getting distance info"
        dists = self.distances()
        print "    %s" % timer.getTimeStamp()
        
        print "Computing cluster hierarchy"
        (o, d) = distance.reachability_order(dists)
        print "    %s" % timer.getTimeStamp()
        
        print "Finding cores"
        T = self.fcluster(o, d)
        out_bins[...] = T
        out_reach_order[...] = o
        out_reach_dists[...] = d
        print "    %s bins made." % len(set(out_bins).difference([0]))
        print "    %s" % timer.getTimeStamp()
            
    def distances(self):
        """computes pairwise distances of observations"""
        pass #subclass to override
        
    def fcluster(self, o, d):
        """finds flat clusters from linkage matrix"""
        pass #subclass to override
        
        
class ClassificationClusterEngine(HierarchicalClusterEngine):
    """Cluster using hierarchical clusturing with feature distance ranks and marker taxonomy"""
    
    def __init__(self, profile, minPts, minSize):
        self._profile = profile
        self._minPts = minPts
        self._minSize = minSize
    
    def distances(self):
        de = ProfileDistanceEngine()
        den_dists = de.makeDensityDistances(self._profile.covProfiles,
                                            self._profile.kmerSigs,
                                            self._profile.contigLengths, 
                                            minPts=self._minPts,
                                            minSize=self._minSize)
        return den_dists
    
    def fcluster(self, o, d):
        Z = hierarchy.linkage_from_reachability(o, d)
        fce = MarkerCheckFCE(self._profile, minPts=self._minPts, minSize=self._minSize)
        bins = fce.makeClusters(Z)
        return bins
            
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ProfileDistanceEngine:
    """Simple class for computing profile feature distances"""

    def makeScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        n = len(contigLengths)
        if(not silent):
            print "Computing pairwise contig distances for 2^%.2f pairs" % np.log2(n*(n-1)//2)
        (lens_i, lens_j) = tuple(contigLengths[i] for i in distance.pairs(n))
        weights = lens_i * lens_j
        scale_factor = 1. / weights.sum()
        (cov_ranks, kmer_ranks) = tuple(distance.argrank(sp_distance.pdist(feature, metric="euclidean"), weights=weights, axis=None) * scale_factor for feature in (covProfiles, kmerSigs))
        return (cov_ranks, kmer_ranks, weights)
    
    def makeNormRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        (cov_ranks, kmer_ranks, weights) = self.makeScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=silent)
        rank_norms = np.sqrt(cov_ranks**2 + kmer_ranks**2)
        return (rank_norms, weights)
    
    def makeDensityDistances(self, covProfiles, kmerSigs, contigLengths, minSize=None, minPts=None, silent=False):
        (rank_norms, weights) = self.makeNormRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        if not silent:
            print "Reticulating splines"
        if minSize is None:
            minWt = None
        else:
            v = np.full(len(contigLengths), contigLengths.min())
            #v = contigLengths
            minWt = np.maximum(minSize - v, 0) * v
        den_dist = distance.density_distance(rank_norms, weights=weights, minWt=minWt, minPts=minPts)
        return den_dist
        
        
class CachingProfileDistanceEngine:
    """Simple class for computing profile feature distances"""

    def __init__(self, savedCovDists, savedKmerDists, savedWeights):
        self._savedCovDists = savedCovDists
        if not self._savedCovDists.endswith(".npy"):
            self._savedCovDists += ".npy"
        self._savedKmerDists = savedKmerDists
        if not self._savedCovDists.endswith(".npy"):
            self._savedCovDists += ".npy"
        self._savedWeights = savedWeights
        if not self._savedWeights.endswith(".npy"):
            self._savedWeights += ".npy"
    
    
    def _getWeights(self, contigLengths):
        try:
            weights = np.load(self._savedWeights)
            assert_num_obs(len(contigLengths), weights)
        except IOError:        
            (lens_i, lens_j) = tuple(contigLengths[i] for i in distance.pairs(len(contigLengths)))
            weights = 1. * lens_i * lens_j
            #weights = sp_distance.pdist(contigLengths[:, None], operator.mul)
            np.save(self._savedWeights, weights)
        return weights
    
    
    def _getScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        n = len(contigLengths)
        if(not silent):
            print "Computing pairwise contig distances for 2^%.2f pairs" % np.log2(n*(n-1)//2)
        cached_weights = None
        scale_factor = None
        try:
            cov_ranks = np.load(self._savedCovDists)
            assert_num_obs(n, cov_ranks)
        except IOError:
            cached_weights = self._getWeights(contigLengths)
            scale_factor = 1. / cached_weights.sum()
            cov_ranks = distance.argrank(sp_distance.pdist(covProfiles, metric="euclidean"), weights=cached_weights, axis=None) * scale_factor
            np.save(self._savedCovDists, cov_ranks)
        try:
            kmer_ranks = np.load(self._savedKmerDists)
            assert_num_obs(n, kmer_ranks)
        except IOError:
            del cov_ranks # save a bit of memory
            if cached_weights is None:
                cached_weights = self._getWeights(contigLengths)
                scaled_factor = 1. / cached_weights.sum()
            kmer_ranks = distance.argrank(sp_distance.pdist(kmerSigs, metric="euclidean"), weights=cached_weights, axis=None) * scale_factor
            np.save(self._savedKmerDists, kmer_ranks)
            cov_ranks = np.load(self._savedCovDists)
        return (cov_ranks, kmer_ranks, cached_weights)
        
    def makeScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        (cov_ranks, kmer_ranks, cached_weights) = self._getScaledRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        if cached_weights is None:
            cached_weights = self._getWeights(contigLengths)
        return (cov_ranks, kmer_ranks, cached_weights)
    
    def makeNormRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        (cov_ranks, kmer_ranks, w) = self._getScaledRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        del w # save some memory
        rank_norms = np.sqrt(cov_ranks**2 + kmer_ranks**2)
        w = self._getWeights(contigLengths)
        return (rank_norms, w)
    
    
    def makeDensityDistances(self, covProfiles, kmerSigs, contigLengths, minSize=None, minPts=None, silent=False):
        (rank_norms, weights) = self.makeNormRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        if not silent:
            print "Reticulating splines"
        if minSize is None:
            minWt = None
        else:
            v = np.full(len(contigLengths), contigLengths.min())
            #v = contigLengths
            minWt = np.maximum(minSize - v, 0) * v
        den_dist = distance.density_distance(rank_norms, weights=weights, minWt=minWt, minPts=minPts)
        return den_dist
        

def assert_num_obs(n, y):
    if n != sp_distance.num_obs_y(y):
        raise SavedDistancesInvalidNumberException("Saved distances for different number of observations")
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class FlatClusterEngine:
    """Flat clustering pipeline"""
    
    def unbinClusters(self, unbin, out_bins):
        out_bins[unbin] = 0
        (_, new_bids) = np.unique(out_bins[out_bins != 0], return_inverse=True)
        out_bins[out_bins != 0] = new_bids+1
    
    def makeClusters(self,
                     Z,
                     return_leaders=False,
                     return_low_quality=False,
                     return_coeffs=False,
                     return_support=False,
                     return_seeds=False,
                     return_conservative_bins=False,
                     return_conservative_leaders=False):
        Z = np.asarray(Z)
        n = Z.shape[0]+1
        
        flat_ids = hierarchy.flatten_nodes(Z)
        scores = self.getScores(Z)
        scores[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = 0 # always propagate descendent scores to equal height parents
        support = scores[n:] - hierarchy.maxscoresbelow(Z, scores, operator.add)
        support = support[flat_ids] # map values from parents to descendents of equal height
        node_support = np.concatenate((scores, support))
        is_low_quality_cluster = self.isLowQualityCluster(Z)
        is_low_quality_cluster[n:] = is_low_quality_cluster[n+flat_ids]
        
        # get leaders of supported bins
        (conservative_bins, conservative_leaders) = hierarchy.fcluster_merge(Z, support>0, return_nodes=True)
        conservative_bins += 1 # bin ids start at 1
        self.unbinClusters(is_low_quality_cluster[conservative_leaders], out_bins=conservative_bins)
        
        # We want to recruit nonsupported clusters to bins if splitting would result in a low quality bin
        is_seed_cluster = np.logical_not(is_low_quality_cluster)
        is_seed_cluster[hierarchy.descendents(Z, conservative_leaders)] = False
        is_seed_cluster[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = False # always propagate descendent scores to equal height parents
        to_merge = hierarchy.maxscoresbelow(Z, is_seed_cluster.astype(int), fun=operator.add)<=1
        to_merge = to_merge[flat_ids]
        (bins, leaders) = hierarchy.fcluster_merge(Z, to_merge, return_nodes=True)
        bins += 1 # bin ids start at 1
        self.unbinClusters(is_low_quality_cluster[leaders], out_bins=bins)
        bins[is_low_quality_cluster[leaders]] = 0
        (_, new_bids) = np.unique(bins[bins != 0], return_inverse=True)
        bins[bins != 0] = new_bids+1
                     
        if not (return_leaders or return_low_quality or return_support or return_coeffs or
                return_seeds or return_conservative_bins or return_conservative_leaders):
            return bins
            
        out = (bins,)
        if return_leaders:
            out += (leaders,)
        if return_low_quality:
            out += (is_low_quality_cluster,)
        if return_support:
            out += (support,)
        if return_coeffs:
            out += (scores,)
        if return_seeds:
            out += (is_seed_cluster,)
        if return_conservative_bins:
            out += (conservative_bins,)
        if return_conservative_leaders:
            out += (conservative_leaders,)
        return out
    
    
    def getMergeNodes_(self,
                      Z,
                      return_low_quality=False,
                      return_support=False,
                      return_child_quality=False,
                      return_support_quality=False):
        Z = np.asarray(Z)
        n = sp_hierarchy.num_obs_linkage(Z)
        
        flat_ids = hierarchy.flatten_nodes(Z)
        scores = self.getScores(Z)
        scores[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = 0 # always propagate descendent scores to equal height parents
        support = scores[n:] - hierarchy.maxscoresbelow(Z, scores, operator.add)
        support = support[flat_ids] # map values from parents to descendents of equal height
        
        # want to merge nonsupported clusters if splitting would result in a low quality bin
        isLowQualityCluster = self.isLowQualityCluster(Z)
        isLowQualityCluster[n:] = isLowQualityCluster[n+flat_ids]
        isQualityCluster = np.logical_not(isLowQualityCluster)
        isQualityCluster[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = False # always propagate descendent quality to equal height parents
        hasAllQualityChildClusters = hierarchy.maxscoresbelow(Z, isQualityCluster, fun=min)
        hasAllQualityChildClusters = hasAllQualityChildClusters[flat_ids]
        
        #hasAllQualityChildClusters_ = isQualityCluster[Z[:, :2].astype(int)].all(axis=1)
        #print np.count_nonzero(hasAllQualityChildClusters_ != hasAllQualityChildClusters)
        
        isQualityNonsupportedCluster_ = np.zeros(2*n-1, dtype=bool)
        isQualityNonsupportedCluster_[n:] = np.logical_and(support==0,
                                                           hasAllQualityChildClusters)
        isOrHasBelowQualityNonsupportedCluster_ = np.logical_or(isQualityNonsupportedCluster_[n:],
                                                               hierarchy.maxscoresbelow(Z, isQualityNonsupportedCluster_, fun=max))
        isOrHasBelowQualityNonsupportedCluster_ = isOrHasBelowQualityNonsupportedCluster_[flat_ids]
        
        to_merge = support>0
        to_merge_while = np.logical_and(support==0, 
                                        np.logical_not(hasAllQualityChildClusters))
        
        if not (return_low_quality or return_support or return_child_quality or return_support_quality):
            return (to_merge, to_merge_while)
            
        out = (to_merge, to_merge_while)
        if return_low_quality:
            out += (isLowQualityCluster,)
        if return_support:
            out += (support,)
        if return_child_quality:
            out += (hasAllQualityChildClusters,)
        if return_support_quality:
            out += (isOrHasBelowQualityNonsupportedCluster_,)
        return out
        
    def makeClusters_(self, Z):
        Z = np.asarray(Z)
        (to_merge, to_merge_while, low_quality) = self.getMergeNodes(Z, return_low_quality=True)
        (bins, leaders) = hierarchy.fcluster_merge(Z, to_merge, merge_while=to_merge_while, return_nodes=True)
        bins += 1 # bin ids start at 1
        bins[low_quality[leaders]] = 0
        (_, new_bids) = np.unique(bins[bins != 0], return_inverse=True)
        bins[bins != 0] = new_bids+1
        return bins
    
    def makeClusters_(self, reach_order, reach_dists):
        reach_order = np.asarray(reach_order)
        reach_dists = np.asarray(reach_dists)
        Z = hierarchy.linkage_from_reachability(reach_order, reach_dists)
        n = sp_hierarchy.num_obs_linkage(Z)
        
        flat_ids = hierarchy.flatten_nodes(Z)
        scores = self.getScores(Z)
        scores[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = 0
        support = scores[n:] - hierarchy.maxscoresbelow(Z, scores, operator.add)
        support = support[flat_ids]
        
        isLowQualityCluster = self.isLowQualityCluster(Z)
        isLowQualityCluster[n:] = isLowQualityCluster[n+flat_ids]
        (greedy_bins, greedy_leaders) = hierarchy.fcluster_merge(Z, support>=0, return_nodes=True)
        greedy_bins += 1 # bin ids start at one
        greedy_bins[isLowQualityCluster[greedy_leaders]] = 0
        
        (conservative_bins, conservative_leaders) = hierarchy.fcluster_merge(Z, support>0, return_nodes=True)
        conservative_bins += 1 # bin ids start at one
        node_support = np.concatenate((scores[:n], support))
        conservative_bins[node_support[conservative_leaders]==0] = 0
        conservative_bins[greedy_bins==0] = 0
        
        # absorb low quality bins greedily
        has_low_quality_child = hierarchy.maxscoresbelow(Z, np.logical_not(isLowQualityCluster), fun=lambda a, b: not a or not b)
        
        
        
        (min_reaches, max_reaches) = hierarchy.reachability_ranges(reach_order, reach_dists)
        reach_slopes = max_reaches / min_reaches
        
        peaks = hierarchy.reachability_peaks(reach_slopes, conservative_bins[reach_order])
        splits = hierarchy.reachability_splits(reach_dists)
        indices_2_nodes = np.empty(n, dtype=int)
        indices_2_nodes[splits] = np.arange(n)
        links = indices_2_nodes[peaks]
        min_peak_slope = reach_slopes[peaks].min()
        
        reach_ratios = hierarchy.reachability_ratios(Z, min_reaches, max_reaches)
        reach_ratios = reach_ratios[flat_ids]
        min_link_ratio = reach_ratios[links].min()
        to_merge = reach_ratios < 1. / min_peak_slope
        to_merge = to_merge[flat_ids]
        
        (recruited_bins, recruited_leaders) = hierarchy.fcluster_merge(Z, to_merge, return_nodes=True)
        recruited_bins[isLowQualityCluster[recruited_leaders]] = 0
        return (greedy_bins, conservative_bins, recruited_bins)
        
    def getScores(self, Z):
        pass #subclass to override
        
    def isLowQualityCluster(self, Z):
        pass #subclass to overrride
        
        
class MarkerCheckFCE(FlatClusterEngine):
    """Seed clusters using taxonomy and marker completeness"""
    
    def __init__(self, profile, minPts=None, minSize=None, filter_leaves=[]):
        self._profile = profile
        self._minSize = minSize
        self._minPts = minPts
        self._filterLeaves = filter_leaves
        if self._minSize is None and self._minPts is None:
            raise ValueError("'minPts' and 'minSize' cannot both be None.")
    
    def getScores(self, Z):
        return MarkerCheckCQE(self._profile, filter_leaves=self._filterLeaves).makeScores(Z)
        
    def isLowQualityCluster(self, Z):
        Z = np.asarray(Z)
        n = sp_hierarchy.num_obs_linkage(Z)
        doMinSize = self._minSize is not None
        doMinPts = self._minPts is not None
        if doMinSize:
            weights = np.concatenate((self._profile.contigLengths, np.zeros(n-1)))
            weights[n:] = hierarchy.maxscoresbelow(Z, weights, fun=operator.add)
            is_low_quality = weights < self._minSize
            
        if doMinPts:
            is_below_minPts = np.concatenate((np.full(self._profile.numContigs, 1 < self._minPts), Z[:, 2] < self._minPts))
            if doMinSize:
                is_low_quality = np.logical_and(is_low_quality, is_below_minPts)
            else:
                is_low_quality = is_below_minPts
                
        return is_low_quality
        
              
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ClusterQualityEngine:
    """Cluster using disagreement of leaf data"""
  
    def makeScores(self, Z):
        """Compute coefficients for hierarchical clustering"""
        Z = np.asarray(Z)
        n = Z.shape[0]+1
        
        node_data = dict(self.getLeafData())
        coeffs = np.zeros(2*n-1, dtype=float)
        
        # Compute leaf clusters
        for (i, indices) in node_data.iteritems():
            coeffs[i] = self.getScore(indices)
            
        # Bottom-up traversal
        for i in range(n-1):
            left_child = int(Z[i, 0])
            right_child = int(Z[i, 1])
            current_node = n+i
            
            # update leaf cache
            try:
                left_data = node_data[left_child]
                del node_data[left_child]
            except:
                left_data = []
            try:
                right_data = node_data[right_child]
                del node_data[right_child]
            except:
                right_data = []
                
            current_data = left_data + right_data
            if current_data != []:
                node_data[current_node] = current_data
            
            # We only compute a new coefficient for new sets of data points, i.e. if
            # both left and right child clusters have data points.
            if left_data == []:
                coeffs[current_node] = coeffs[right_child]
            elif right_data == []:
                coeffs[current_node] = coeffs[left_child]
            else:
                coeffs[current_node] = self.getScore(current_data)
            
        return coeffs
        
    def getLeafData(self):
        pass #subclass to override
        
    def getScore(self, node_data):
        """Compute coefficients using concatenated leaf data"""
        pass # subclass to override
        
        
class MarkerCheckCQE(ClusterQualityEngine):
    """Cluster coefficient using taxonomy and marker completeness"""
    
    def __init__(self, profile, filter_leaves=[]):
        self._alpha = 0.5
        self._d = 1
        self._mapping = profile.mapping
        self._mdists = sp_distance.squareform(self._mapping.classification.makeDistances()) < self._d
        (_mnames, self._mgroups) = np.unique(self._mapping.markerNames, return_inverse=True)
        self._mcounts = np.array([len(np.unique(self._mgroups[row])) for row in self._mdists])
        self._mscalefactors = 1./self._mcounts
        self._filterLeaves = filter_leaves
        
    def getLeafData(self):
        filter_leaf_set = set(self._filterLeaves)
        return dict([(i, data) for (i, data) in self._mapping.iterindices() if i not in filter_leaf_set])
        
    def getScore(self, indices):
        """Compute modified completeness and precision scores"""
        # number of unique markers that are taxonomically coherence with each item in cluster
        indices = np.asarray(indices)
        correct = np.array([len(np.unique(self._mgroups[indices[row]])) for row in self._mdists[np.ix_(indices, indices)]])
        # item precision is fraction of cluster that is correct
        prec = correct * 1. / len(indices)
        # item completeness is fraction of taxonomically coherent markers in data set in cluster
        compl = (correct * self._mscalefactors[indices])
        f = self._alpha * prec.sum() + (1 - self._alpha) * compl.sum()
        return f
        
        
class DisagreementCQE_(ClusterQualityEngine):
    """Cluster using disagreement of leaf data"""
    
    def __init__(self, profile):
        self._profile = profile
        self.getScore = ClassificationManager(self._profile.mapping).disagreement
        
    def getLeafData(self):
        return dict(self._profile.mapping.iterindices())
     
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
