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

__author__ = "Tim Lamberton"
__copyright__ = "Copyright 2016"
__credits__ = ["Tim Lamberton"]
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
import tables
import sys

# local imports
import distance
import recruit
import hierarchy
from profileManager import ProfileManager
from groopmExceptions import SavedDistancesInvalidNumberException, CacheUnavailableException
from utils import group_iterator

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
            savedDistsPrefix="",
            keepDists=False,
            force=False):
        # check that the user is OK with nuking stuff...
        if not force and not self._pm.promptOnOverwrite():
            return
            
        profile = self.loadProfile(timer,
                                   minLength=minLength
                                   )
        
        if savedDistsPrefix=="":
            savedDistsPrefix = self._dbFileName
        cacher = FileCacher(savedDistsPrefix+".dists")
        
        ce = ClassificationClusterEngine(profile,
                                         minPts=minPts,
                                         minSize=minSize,
                                         cacher=cacher,
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
        
        # Remove created files
        if not keepDists:
            try:
                cacher.cleanup()
            except:
                raise
            
        
        
# Hierarchical clustering
class HierarchicalClusterEngine:
    """Abstract hierarchical clustering pipeline.
    Subclass should provide `distances` and `fcluster` methods to the
    interface outlined below.
    """
    
    def makeBins(self, timer, out_bins, out_reach_order, out_reach_dists):
        """Run binning algorithm"""
        
        print "Getting distance info"
        (pdists, core_dists) = self.distances()
        print "    %s" % timer.getTimeStamp()
        
        print "Computing cluster hierarchy"
        (o, d) = distance.reachability_order(pdists, core_dists)
        print "    %s" % timer.getTimeStamp()
        
        print "Finding cores"
        T = self.fcluster(o, d)
        out_bins[...] = T
        out_reach_order[...] = o
        out_reach_dists[...] = d
        print "    %s bins made." % len(set(out_bins).difference([0]))
        print "    %s" % timer.getTimeStamp()
            
    def distances(self):
        """Computes pairwise distances of observations.
        
        Returns
        -------
        Y : ndarray
            Returns a condensed distance matrix for pairs of observations.
            See `squareform` from the `scipy` documentation.
        
        """
        pass #subclass to override
        
    def fcluster(self, o, d):
        """Find flat clusters from reachability summary.
        
        Parameters
        ----------
        o : ndarray
            1-D array of indices of original observations in traversal order.
        d : ndarray
            1-D array. `d[i]` is the `i`th traversal distance.
            
        Returns
        -------
        T : ndarray
            1-D array. `T[i]` is the flat cluster number to which original
            observation `i` belongs.
        """
        pass #subclass to override
        
        
class ClassificationClusterEngine(HierarchicalClusterEngine):
    """Cluster using hierarchical clusturing with feature distance ranks and marker taxonomy"""
    
    def __init__(self, profile, minPts=None, minSize=None, cacher=None):
        if (minSize is None) and (minPts is None):
            raise ValueError("Specify at least one of 'minWt' or 'minPts' parameter values")
        self._profile = profile
        self._minPts = minPts
        self._minSize = minSize
        self._cacher = cacher
    
    def distances(self, silent=False):
        if(not silent):
            n = len(self._profile.contigLengths)
            print "Computing pairwise contig distances for 2^%.2f pairs" % np.log2(n*(n-1)//2)
        de = ProfileDistanceEngine() if self._cacher is None else CachingProfileDistanceEngine(cacher=self._cacher)
        #de = CachingWeightlessProfileDistanceEngine(distStore=self._distStore)
        (rank_norms, w) = de.makeRankNorms(self._profile.covProfiles,
                                           self._profile.kmerSigs,
                                           self._profile.contigLengths, 
                                           silent=silent)
        #(rank_norms, w) = DistanceStatEngine(de, mode="triangular").makeStat(self._profile.covProfiles,
                                                                             #self._profile.kmerSigs,
                                                                             #self._profile.contigLengths,
                                                                             #silent=silent)
        if not silent:
            print "Reticulating splines"
            
        # Convert the minimum size in bp of a bin to the minimum weighted density
        # used to compute the density distance. For a contig of size L, the sum of
        # nearest neighbour weights will be W=L*{sum of nearest neighbour lengths}. The
        # corresponding size of the bin including the nearest neighbours will be
        # S=L+{sum of nearest neighbour lengths}. Applying the size constraint S>minSize
        # yields the weight constraint W>L*(minSize - L).
        if self._minSize:
            v = np.full(len(self._profile.contigLengths), self._profile.contigLengths.min())
            #v = contigLengths
            minWt = np.maximum(self._minSize - v, 0) * v
        else:
            minWt = None
        core_dists = distance.core_distance(rank_norms, w, minWt=minWt, minPts=self._minPts)
        
        #if minWt is not None:
        #    x = distance.core_distance_weighted_(rank_norms, w, minWt)
            
        #if self._minPts is not None:
        #    p = distance.core_distance_(rank_norms, self._minPts)
        #    if minWt is not None:
        #        x = np.minimum(x, p)
        #    else:
        #        x = p
        #assert np.all(core_dists==x)
        
        return (rank_norms, core_dists)
    
    def fcluster(self, o, d):
        Z = hierarchy.linkage_from_reachability(o, d)
        fce = MarkerCheckFCE(self._profile, minPts=self._minPts, minSize=self._minSize)
        bins = fce.makeClusters(Z)
        return bins

        
###############################################################################
###############################################################################
###############################################################################
###############################################################################        
   
class DistanceStatEngine:
    def __init__(self, de, mode="radial"):
        self._de = de
        if mode=="radial":
            self._n = 2
            self._area = _iradial_area
        elif mode=="triangular":
            self._n = 1
            self._area = _itriangular_area
        else:
            raise ValueError("Parameter value for argument 'mode' must be one of: 'radial', 'triangular'.")
        
    def makeStat(self, covProfiles, kmerSigs, contigLengths, silent=False):
            
        (norms, w) = self._de.makeRankNorms(covProfiles, kmerSigs, contigLengths, silent=silent, n=self._n)
        self._area(out=norms)
        norms *= w.sum()
            
        # normalise to actual count
        norms /= distance.iargrank(norms.copy(), weights=w, axis=None)
        
        return (norms, w)
    
    
def _iradial_area(out):
    l = out<=1; nl = np.logical_not(l)
    
    # 2-norm to radial area in 1x1 square
    out **= 2
    out /= 2 #angular area
    out[l] *= np.pi / 2
    o = np.sqrt(2*out[nl] - 1)
    al = np.pi / 2 - 2*out.arctan(o)
    out[nl] *= al
    out[nl] += o
        
def _itriangular_area(out):
    l = out<=1; nl = np.logical_not(l)
    
    # 1-norm to lower-triangular area in 1x1 square
    out[l] **= 2
    out[l] /= 2
    out[nl] = 2 - out[nl]
    out[nl] **= 2
    out[nl] = 2 - out[nl]
    out[nl] /= 2
    
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ProfileDistanceEngine:
    """Simple class for computing profile feature distances"""
    
    def makeScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        """Compute pairwise rank distances separately for coverage profiles and
        kmer signatures, and give rank distances as a fraction of the largest rank.
        """
        n = len(contigLengths)
        weights = np.empty( n * (n-1) // 2, dtype=np.double)
        k = 0
        for i in range(n-1):
            weights[k:(k+n-1-i)] = contigLengths[i]*contigLengths[(i+1):n]
            k = k+n-1-i
        scale_factor = 1. / weights.sum()
        (cov_ranks, kmer_ranks) = tuple(distance.iargrank(sp_distance.pdist(feature, metric="euclidean"), weights=weights, axis=None) * scale_factor for feature in (covProfiles, kmerSigs))
        return (cov_ranks, kmer_ranks, weights)
    
    def makeRankNorms(self, covProfiles, kmerSigs, contigLengths, silent=False):
        """Compute norms in {coverage rank space x kmer rank space}
        """
        (cov_ranks, kmer_ranks, weights) = self.makeScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=silent)
        dists = np.sqrt(cov_ranks**2 + kmer_ranks**2)
        return (dists, weights)

        
class CachingProfileDistanceEngine:
    """Class for computing profile feature distances. Does caching to disk to keep memory usage down."""

    def __init__(self, cacher):
        self._cacher = cacher
    
    def _getWeights(self, contigLengths, silent=False):
        n = len(contigLengths)
        try:
            weights = self._cacher.getWeights()
            assert_num_obs(n, weights)
        except CacheUnavailableException:
            if not silent:
                print "Calculating distance weights"
            #(lens_i, lens_j) = tuple(contigLengths[i] for i in distance.pairs(len(contigLengths)))
            #weights = 1. * lens_i * lens_j
            weights = np.empty( n * (n-1) // 2, dtype=np.double)
            k = 0
            for i in range(n-1):
                weights[k:(k+n-1-i)] = contigLengths[i]*contigLengths[(i+1):n]
                k = k+n-1-i
            #weights = sp_distance.pdist(contigLengths[:, None], operator.mul)
            self._cacher.storeWeights(weights)
        return weights
    
    def _getScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        """Compute pairwise rank distances separately for coverage profiles and
        kmer signatures, and give rank distances as a fraction of the largest rank.
        """
        n = len(contigLengths)
        cached_weights = None
        scale_factor = None
        try:
            cov_ranks = self._cacher.getCovDists()
            assert_num_obs(n, cov_ranks)
        except CacheUnavailableException:
            cached_weights = self._getWeights(contigLengths, silent=silent)
            scale_factor = 1. / cached_weights.sum()
            if not silent:
                print "Calculating coverage distance ranks"
            cov_ranks = distance.iargrank(out=sp_distance.pdist(covProfiles, metric="euclidean"), weights=cached_weights, axis=None)
            cov_ranks *= scale_factor
            #x = distance.argrank(sp_distance.pdist(covProfiles, metric="euclidean"), weights=cached_weights, axis=None) * scale_factor
            #assert np.all(x==cov_ranks)
            self._cacher.storeCovDists(cov_ranks)
        try:
            kmer_ranks = self._cacher.getKmerDists()
            assert_num_obs(n, kmer_ranks)
        except CacheUnavailableException:
            if cached_weights is None:
                cached_weights = self._getWeights(contigLengths, silent=silent)
                scale_factor = 1. / cached_weights.sum()
            #kmer_ranks = cov_ranks # mem opt, reuse cov_ranks memory
            del cov_ranks
            if not silent:
                print "Calculating tetramer distance ranks"
            kmer_ranks = distance.iargrank(sp_distance.pdist(kmerSigs, metric="euclidean"), weights=cached_weights, axis=None)
            kmer_ranks *= scale_factor
            #x = distance.argrank(sp_distance.pdist(kmerSigs, metric="euclidean"), weights=cached_weights, axis=None) * scale_factor
            #assert np.all(x==kmer_ranks)
            self._cacher.storeKmerDists(kmer_ranks)
            cov_ranks = self._cacher.getCovDists()
        return (cov_ranks, kmer_ranks, cached_weights)
    
    def makeScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        (cov_ranks, kmer_ranks, weights) = self._getScaledRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        if weights is None:
            weights = self._getWeights(contigLengths, silent=silent)
        return (cov_ranks, kmer_ranks, weights)
        
    def makeRankNorms(self, covProfiles, kmerSigs, contigLengths, silent=False, n=2):
        """Compute norms in {coverage rank space x kmer rank space}
        """
        (cov_ranks, kmer_ranks, weights) = self._getScaledRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        #x = cov_ranks * kmer_ranks / (cov_ranks + kmer_ranks)
        rank_norms = cov_ranks
        del cov_ranks # invalidated
        rank_norms **= n
        kmer_ranks **= n
        rank_norms += kmer_ranks
        rank_norms **= 1. / n
        if weights is None:
            weights = self._getWeights(contigLengths)
        return (rank_norms, weights)
                        
                        
class CachingWeightlessProfileDistanceEngine:
    """Class for computing profile feature distances. Does caching to disk to keep memory usage down."""

    def __init__(self, cacher):
        self._cacher = cacher
            
    def _getWeights(self, contigLengths):
        n = len(contigLengths)
        try:
            weights = self._cacher.getWeigths()
            assert_num_obs(n, weights)
        except CacheUnavailableException:
            #(lens_i, lens_j) = tuple(contigLengths[i] for i in distance.pairs(len(contigLengths)))
            #weights = 1. * lens_i * lens_j
            weights = np.empty( n * (n-1) // 2)
            k = 0
            for i in range(n-1):
                weights[k:(k+n-1-i)] = contigLengths[i]*contigLengths[(i+1):n]
                k = k+n-1-i
            #weights = sp_distance.pdist(contigLengths[:, None], operator.mul)
            self._cacher.storeWeights(weights)
        return weights
        
    def _getScaledRanks(self, covProfiles, kmerSigs, silent=False):
        """Compute pairwise rank distances separately for coverage profiles and
        kmer signatures, and give rank distances as a fraction of the largest rank.
        """
        n = len(covProfiles)
        if(not silent):
            print "Computing pairwise contig distances for 2^%.2f pairs" % np.log2(n*(n-1)//2)
        cached_weights = None
        scale_factor = 1. / n
        try:
            cov_ranks = self._cacher.getCovDists()
            assert_num_obs(n, cov_ranks)
        except CacheUnavailableException:
            #cached_weights = self._getWeights(contigLengths)
            cov_ranks = distance.iargrank(out=sp_distance.pdist(covProfiles, metric="euclidean"), axis=None)
            cov_ranks *= scale_factor
            #x = distance.argrank(sp_distance.pdist(covProfiles, metric="euclidean"), weights=cached_weights, axis=None) * scale_factor
            #assert np.all(x==cov_ranks)
            self._cacher.storeCovDists(cov_ranks)
        try:
            kmer_ranks = self._cacher.getKmerDists()
            assert_num_obs(n, kmer_ranks)
        except tables.exceptions.NoSuchNodeError:
            #kmer_ranks = cov_ranks # mem opt, reuse cov_ranks memory
            del cov_ranks
            kmer_ranks = distance.iargrank(sp_distance.pdist(kmerSigs, metric="euclidean"), axis=None)
            kmer_ranks *= scale_factor
            #x = distance.argrank(sp_distance.pdist(kmerSigs, metric="euclidean"), weights=cached_weights, axis=None) * scale_factor
            #assert np.all(x==kmer_ranks)
            self._cacher.storeKmerDists(kmer_dists)
            cov_ranks = self._cacher.getCovDistS()
        return (cov_ranks, kmer_ranks)
    
    def makeScaledRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        (cov_ranks, kmer_ranks) = self._getScaledRanks(covProfiles, kmerSigs, silent=silent)
        w = self._getWeights(contigLengths)
        return (cov_ranks, kmer_ranks, w)
    
    def makeRankNorms(self, covProfiles, kmerSigs, contigLengths, silent=False, n=2):
        """Compute norms in {coverage rank space x kmer rank space}
        """
        (cov_ranks, kmer_ranks) = self._getScaledRanks(covProfiles, kmerSigs, silent=silent)
        #x = cov_ranks**2 + kmer_ranks**2
        rank_norms = cov_ranks
        del cov_ranks # invalidated
        rank_norms **= n
        kmer_ranks **= n
        rank_norms += kmer_ranks
        rank_norms **= 1. / n
        #assert np.all(rank_norms==np.sqrt(x))
        w = self._getWeights(contigLengths)
        return (rank_norms, w)
        
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################  
class Cacher:
    """Class for caching profile feature distances"""
    
    def cleanup():
        pass
    
    def getWeights():
        pass
        
    def storeWeights(weights):
        pass
        
    def getCovDists():
        pass
        
    def storeCovDists(cov_dists):
        pass
        
    def getKmerDists():
        pass
        
    def storeKmerDists(kmer_dists):
        pass
        
        
class FileCacher(Cacher):
    """Cache using numpy to/fromfile"""
    
    def __init__(self, distStorePrefix):
        self._weightsStore = distStorePrefix+".weights"
        self._covDistStore = distStorePrefix+".cov"
        self._kmerDistStore = distStorePrefix+".kmer"
        
    def cleanup(self):
        os.remove(self._weightsStore)
        os.remove(self._covDistStore)
        os.remove(self._kmerDistStore)
        
    def getWeights(self):
        try:
            weights = np.fromfile(self._weightsStore, dtype=np.double)
        except IOError:
            raise CacheUnavailableException()
        return weights
        
    def storeWeights(self, weights):
        np.asanyarray(weights, dtype=np.double).tofile(self._weightsStore)
        
    def getCovDists(self):
        try:
            cov_dists = np.fromfile(self._covDistStore, dtype=np.double)
        except IOError:
            raise CacheUnavailableException()
        return cov_dists
        
    def storeCovDists(self, cov_dists):
        np.asanyarray(cov_dists, dtype=np.double).tofile(self._covDistStore)
        
    def getKmerDists(self):
        try:
            kmer_dists = np.fromfile(self._kmerDistStore, dtype=np.double)
        except IOError:
            raise CacheUnavailableException()
        return kmer_dists
        
    def storeKmerDists(self, kmer_dists):
        np.asanyarray(kmer_dists, dtype=np.double).tofile(self._kmerDistStore)
        
        
class TablesCacher(Cacher):
    """Cache using pytable"""

    def __init__(self, distStore):
        self._distStoreFile = distStore
        try:
            with tables.open_file(self._distStoreFile, mode="a", title="Distance store") as h5file:
                pass
        except:
            print "Error creating database:", self._distStoreFile, sys.exc_info()[0]
            raise
            
    def cleanup(self):
        os.remove(self._distStoreFile)
    
    def getWeights(self):
        try:
            with tables.open_file(self._distStoreFile, mode="r") as h5file:
                weights = h5file.get_node("/", "weights").read()
        except tables.exceptions.NoSuchNodeError:
            raise CacheUnavailableException()
        return weights
        
    def storeWeights(self, weights):
        with tables.open_file(self._distStoreFile, mode="a") as h5file:
            h5file.create_array("/", "weights", weights, "Distance weights")
    
    def getCovDists(self):
        try:
            with tables.open_file(self._distStoreFile, mode="r") as h5file:
                cov_dists = h5file.get_node("/", "coverage").read()
        except tables.exceptions.NoSuchNodeError:
            raise CacheUnavailableException()
        return cov_dists
        
    def storeCovDists(self, cov_dists):
        with tables.open_file(self._distStoreFile, mode="a") as h5file:
            h5file.create_array("/", "coverage", cov_dists, "Coverage distances")
            
    def getKmerDists(self):
        try:
            with tables.open_file(self._distStoreFile, mode="r") as h5file:
                kmer_dists = h5file.get_node("/", "kmer").read()
            assert_num_obs(n, kmer_ranks)
        except tables.exceptions.NoSuchNodeError:
            raise CacheUnavailableException()
        return kmer_dists
        
    def storeKmerDists(self, kmer_dists):
        with tables.open_file(self._distStoreFile, mode="a") as h5file:
            h5file.create_array("/", "kmer", kmer_dists, "Tetramer distances")
        
        

def assert_num_obs(n, y):
    if n != sp_distance.num_obs_y(y):
        raise SavedDistancesInvalidNumberException("Saved distances for different number of observations")
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
class FlatClusterEngine_:
    """Flat clustering pipeline.
    
    Subclass should provide `getScores` and `isLowQuality` methods with the
    described interfaces.
    """
    support_tol = 0.
    
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
        """Implements algorithm for cluster formation.
        
        The set of flat clusters returned (should) satisfy the following
        constraints:
            1. All clusters exceed a minimum standard (as reported by
               `isLowQuality` method);
            2. As allowed by 1, the number of clusters is the maximimum such
               that no smaller number of clusters has a higher sum quality (as
               reported by `getScore` method);
            3. As allowed by 1 and 2, the total size of clusters is maximum.
        
        The strategy used is:
            - Find a minimal set of minimum standard clusters which maximise the
              sum quality
            - Grow clusters by greedily merging only below standard clusters
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering of a set of
            original observations.
        
        Returns
        -------
        T : ndarray
            1-D array. `T[i]` is the flat cluster number to which original
            observation `i` belongs.
        """
        Z = np.asarray(Z)
        n = Z.shape[0]+1
        
        # NOTE: Cases of nested clusters where the child and parent cluster heights
        # are equal are ambiguously encoded in hierarchical clusterings. 
        # This is handled by finding the row of the highest ancestor node of equal height
        # and computing scores as if all equal height descendents were considered as the
        # same node as the highest ancestor, with children corresponding to the union of
        # the combined nodes' children.
        flat_ids = hierarchy.flatten_nodes(Z)
        scores = np.asarray(self.getScores(Z))
        scores[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = np.min(scores) # always propagate descendent scores to equal height parents
        # NOTE: support is a measure of the degree to which a cluster quality
        # improves on the combined quality of the best clusters below (computed
        # by maxscorebelow). Positive values indicate that the parent cluster 
        # should be favoured, zero values indicate no preference, and negative
        # values indicate that the best clusters below should be prefered.
        support = scores[n:] - hierarchy.maxscoresbelow(Z, scores, operator.add)
        support = support[flat_ids] # map values from parents to descendents of equal height
        node_support = np.concatenate((scores[:n], support))
        
        is_low_quality_cluster = np.asarray(self.isLowQualityCluster(Z))
        is_low_quality_cluster[n:] = is_low_quality_cluster[n+flat_ids]
        
        # NOTE: conservative bins are a minimal set of clusters that have
        # maximum combined quality. The returned conservative bins have below
        # standard bins dissolved.
        (conservative_bins, conservative_leaders) = hierarchy.fcluster_merge(Z, support>0, return_nodes=True)
        conservative_bins += 1 # bin ids start at 1
        self.unbinClusters(is_low_quality_cluster[conservative_leaders], out_bins=conservative_bins)
        
        (bins, leaders) = hierarchy.fcluster_merge(Z, support>=0, return_nodes=True)
        bins += 1 # bin ids start at 1
        self.unbinClusters(is_low_quality_cluster[leaders], out_bins=bins)
                     
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
    
    def getScores(self, Z):
        """Compute cluster quality scores for nodes in a hierarchical clustering.
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering. See 
            `linkage` in `scipy` documentation.
        
        Returns
        -------
        scores : ndarray
            1-D array. `scores[i]` where `i < n` is the `quality` score for the
            singleton cluster consisting of original observation `i`. 
            `scores[i]` where `i >= n` is the `quality` score for the cluster
            consisting of the original observations below the node represented
            by the `i-n`th row of `Z`.
        """
        pass #subclass to override
        
    def isLowQualityCluster(self, Z):
        """Bit of a hack. Indicate clusters that are intrinsically low quality
        e.g. too small,  not enough bp, but don't 'infect' the quality of higher
        clusters.
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering.
        
        Returns
        -------
        l : ndarray
            1-D boolean array. `l[i]` is True for clusters that are
            low quality and otherwise False. Here `i` is a singleton 
            cluster for `i < n` and an internal cluster for `i >= n`.
        """
        pass #subclass to overrride
        
        
class FlatClusterEngine:
    """Flat clustering pipeline.
    
    Subclass should provide `getScores` and `isLowQuality` methods with the
    described interfaces.
    """
    support_tol = 0.
    
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
        """Implements algorithm for cluster formation.
        
        The set of flat clusters returned (should) satisfy the following
        constraints:
            1. All clusters exceed a minimum standard (as reported by
               `isLowQuality` method);
            2. As allowed by 1, the number of clusters is the maximimum such
               that no smaller number of clusters has a higher sum quality (as
               reported by `getScore` method);
            3. As allowed by 1 and 2, the total size of clusters is maximum.
        
        The strategy used is:
            - Find a minimal set of minimum standard clusters which maximise the
              sum quality
            - Grow clusters by greedily merging only below standard clusters
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering of a set of
            original observations.
        
        Returns
        -------
        T : ndarray
            1-D array. `T[i]` is the flat cluster number to which original
            observation `i` belongs.
        """
        Z = np.asarray(Z)
        n = Z.shape[0]+1
        
        # NOTE: Cases of nested clusters where the child and parent cluster heights
        # are equal are ambiguously encoded in hierarchical clusterings. 
        # This is handled by finding the row of the highest ancestor node of equal height
        # and computing scores as if all equal height descendents were considered as the
        # same node as the highest ancestor, with children corresponding to the union of
        # the combined nodes' children.
        flat_ids = hierarchy.flatten_nodes(Z)
        scores = np.asarray(self.getScores(Z))
        scores[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = np.min(scores) # always propagate descendent scores to equal height parents
        # NOTE: support is a measure of the degree to which a cluster quality
        # improves on the combined quality of the best clusters below (computed
        # by maxscorebelow). Positive values indicate that the parent cluster 
        # should be favoured, zero values indicate no preference, and negative
        # values indicate that the best clusters below should be prefered.
        support = scores[n:] - hierarchy.maxscoresbelow(Z, scores, operator.add)
        support = support[flat_ids] # map values from parents to descendents of equal height
        node_support = np.concatenate((scores[:n], support))
        
        is_low_quality_cluster = np.asarray(self.isLowQualityCluster(Z))
        is_low_quality_cluster[n:] = is_low_quality_cluster[n+flat_ids]
        
        # NOTE: conservative bins are a minimal set of clusters that have
        # maximum combined quality. The returned conservative bins have below
        # standard bins dissolved.
        (conservative_bins, conservative_leaders) = hierarchy.fcluster_merge(Z, support>0, return_nodes=True)
        conservative_bins += 1 # bin ids start at 1
        self.unbinClusters(is_low_quality_cluster[conservative_leaders], out_bins=conservative_bins)
        
        # NOTE: A seed cluster is any putative cluster that meets the minimum
        # standard, is not already part of a conservative bin, and is within
        # support tolerance of conservative bins below. 
        is_seed_cluster = np.logical_not(is_low_quality_cluster)
        is_seed_cluster[Z[support<-self.support_tol, :2].astype(int)] = True
        #for i in range(n-1):
        #    if (node_support[n+i]<-self.support_tol):
        #        is_seed_cluster[Z[i,0]] = is_seed_cluster[Z[i,1]] = True
        is_seed_cluster[hierarchy.descendents(Z, conservative_leaders)] = False
        is_seed_cluster[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = False # always propagate descendent scores to equal height parents
        # NOTE: to_merge is true for nodes that have at most 1 seed cluster
        # below them. We greedily merge these nodes into clusters to obtain the
        # largest possible clusters, without merging any two seed clusters. 
        to_merge = hierarchy.maxscoresbelow(Z, is_seed_cluster.astype(int), fun=operator.add)<=1
        to_merge = to_merge[flat_ids]
        (bins, leaders) = hierarchy.fcluster_merge(Z, to_merge, return_nodes=True)
        bins += 1 # bin ids start at 1
        # NOTE: any new clusters that are below minimum standard are dissolved
        # here. 
        self.unbinClusters(is_low_quality_cluster[leaders], out_bins=bins)
                     
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
    
    def getScores(self, Z):
        """Compute cluster quality scores for nodes in a hierarchical clustering.
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering. See 
            `linkage` in `scipy` documentation.
        
        Returns
        -------
        scores : ndarray
            1-D array. `scores[i]` where `i < n` is the `quality` score for the
            singleton cluster consisting of original observation `i`. 
            `scores[i]` where `i >= n` is the `quality` score for the cluster
            consisting of the original observations below the node represented
            by the `i-n`th row of `Z`.
        """
        pass #subclass to override
        
    def isLowQualityCluster(self, Z):
        """Bit of a hack. Indicate clusters that are intrinsically low quality
        e.g. too small,  not enough bp, but don't 'infect' the quality of higher
        clusters.
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering.
        
        Returns
        -------
        l : ndarray
            1-D boolean array. `l[i]` is True for clusters that are
            low quality and otherwise False. Here `i` is a singleton 
            cluster for `i < n` and an internal cluster for `i >= n`.
        """
        pass #subclass to overrride
        
        
class MarkerCheckFCE(FlatClusterEngine):
    """Seed clusters using taxonomy and marker completeness"""
    def __init__(self, profile, minPts=None, minSize=None):
        self._profile = profile
        self._minSize = minSize
        self._minPts = minPts
        if self._minSize is None and self._minPts is None:
            raise ValueError("'minPts' and 'minSize' cannot both be None.")
            
    support_tol = 1.
    
    def getScores(self, Z):
        return MarkerCheckCQE(self._profile).makeScores(Z)
        
    def isLowQualityCluster(self, Z):
        Z = np.asarray(Z)
        n = sp_hierarchy.num_obs_linkage(Z)
        flat_ids = hierarchy.flatten_nodes(Z)
        
        # Quality clusters have total contig length at least minSize (if
        # defined) or a number of contigs at least minPts (if defined)
        doMinSize = self._minSize is not None
        doMinPts = self._minPts is not None
        if doMinSize:
            weights = np.concatenate((self._profile.contigLengths, np.zeros(n-1)))
            weights[n:] = hierarchy.maxscoresbelow(Z, weights, fun=operator.add)
            weights[n:] = weights[flat_ids+n]
            is_low_quality = weights < self._minSize   
        if doMinPts:
            is_below_minPts = np.concatenate((np.full(self._profile.numContigs, 1 < self._minPts, dtype=bool), Z[flat_ids, 3] < self._minPts))
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
    """Assign cluster qualities using leaf observation attributes.
    Subclass to provide `getLeafData` and `getScore` functions with the
    interfaces specified below.
    """
  
    def makeScores(self, Z):
        """Compute coefficients for hierarchical clustering
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering.
        
        Returns
        -------
        scores : ndarray
            1-D array. `scores[i]` returns the score for cluster `i`
            computed from the concatenated data of leaves below the cluster
            using `getScore`.
        """
        Z = np.asarray(Z)
        n = Z.shape[0]+1
        
        node_data = dict(self.getLeafData())
        coeffs = np.zeros(2*n-1, dtype=float)
        
        # Compute leaf clusters
        for (i, leaf_data) in node_data.iteritems():
            coeffs[i] = self.getScore(leaf_data)
            
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
        """Data associated with leaf nodes
        
        Returns
        -------
        data : dict
            A dictionary with original observation ids as keys and lists of associated leaf node data as values.
        """
        pass #subclass to override
        
    def getScore(self, node_data):
        """Compute cluster quality using concatenated leaf data.
        
        Parameters
        ----------
        node_data : list
            List of associated data for leaf nodes of a cluster.
        
        Returns
        -------
        score : float
            Value representing the `quality` score for the cluster.
        """
        pass # subclass to override
        
        
class MarkerCheckCQE(ClusterQualityEngine):
    """Cluster quality scores using taxonomy and marker completeness.
    
    We use extended BCubed metrics where precision and recall metrics are
    defined for each observation in a cluster.
    
    Traditional BCubed metrics assume that extrinsic categories are known 
    for the clustered items. In our case we have two sources of information - 
    mapping taxonomies and mapping markers. Mapping taxonomies can be used to
    group items that are 'similar enough'. Single copy markers can be used to
    distinguish when items should be in different bins.
    
    With these factors in mind, we have used the following adjusted BCubed 
    metrics:
        - Precision represents for an item how many of the item's cluster 
          are taxonomically similar items with different markers. Analogous to
          an inverted genome 'contamination' measure.
        - Recall represents for an item how many of taxonomically similar items with
          different markers are found in the item's cluster. Analogous to genome
          'completeness' measure.
          
    These metrics are combined in a naive linear manner using a mixing fraction
    alpha which can be combined additively when assessing multiple clusters.
    """
    
    def __init__(self, profile):
        self._d = 1
        self._alpha = 0.5
        self._mapping = profile.mapping
        
        # Connectivity matrix: M[i,j] = 1 where mapping i and j are 
        # taxonomically 'similar enough', otherwise 0.
        self._mdists = sp_distance.squareform(self._mapping.classification.makeDistances()) < self._d
        
        # Compute the number of copies for each marker
        markerNames = self._mapping.markerNames
        (_name, bin, copies) = np.unique(markerNames, return_inverse=True, return_counts=True)
        self._mscalefactors = 1. / copies[bin]
        
        # Compute the compatible marker group sizes
        #gsizes = np.array([len(np.unique(markerNames[row])) for row in self._mdists])
        gsizes = np.array([np.count_nonzero(row) / np.unique(markerNames[row], return_counts=True)[1].max() for row in self._mdists])
        self._gscalefactors = 1. / gsizes
        
    def getLeafData(self):
        """Leaf data is a list of indices of mappings."""
        return dict([(i, data) for (i, data) in self._mapping.iterindices()])
        
    def getScore(self, indices):
        """Compute modified BCubed completeness and precision scores."""
        indices = np.asarray(indices)
        markerNames = self._mapping.markerNames[indices]
        gsizes = np.array([len(np.unique(markerNames[row])) for row in (self._mdists[index, indices] for index in indices)])
        # item precision is the average number of compatible unique markers in
        # cluster
        prec = (gsizes * 1. / len(indices)).sum()
        # item completeness / recall is the percentage of compatible genes in
        # cluster for good markers in cluster
        recall = ((gsizes - 1) * gsizes * self._gscalefactors[indices] * 1. / len(indices)).sum()
        f = self._alpha * recall + (1 - self._alpha) * prec
        return f
        
       
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class _TreeRecursivePrinter:
    def __init__(self, Z, indices, leaf_labeller, node_labeller):
        self._Z = np.asarray(Z)
        self._n = self._Z.shape[0] + 1
        self._flat_ids = hierarchy.flatten_nodes(self._Z)
        self._embed_ids = hierarchy.embed_nodes(self._Z, indices)
        self._indices = indices
        # map from flat_id -> closest embed_id
        self._flat_embed_ids = dict(zip(self._flat_ids, self._embed_ids))
        self._leaf_labeller = leaf_labeller
        self._node_labeller = node_labeller

    def getLines(self, node_id=None):
        n = self._n
        if node_id is None:
            node_id = self._embed_ids[-1]
            
        flat_id = self._flat_ids[node_id-n]+n if node_id >= n else node_id
        embed_id = self._embed_ids[node_id-n] if node_id >= n else node_id
        embed = node_id < n or self._flat_embed_ids[flat_id-n] == embed_id
        if embed:
            node_id = embed_id
            
        if node_id < n:
            if node_id==-1 or not np.any(node_id == self._indices): # not embedded
                return []
            else:   
                # embedded leaf 
                return [self._node_labeller(flat_id)+self._leaf_labeller(embed_id)]
        else:
            left_child = int(self._Z[node_id-n,0])
            right_child = int(self._Z[node_id-n,1])
            child_lines = self.getLines(left_child)+self.getLines(right_child)
            if embed:
                return [self._node_labeller(flat_id)]+['--'+l for l in child_lines]
            else:
                return child_lines
        

class MarkerTreePrinter:
    def printTree(self, indices, leaves_list=None):
        Z = self.getLinkage()
        Z = np.asarray(Z)
        rp = _TreeRecursivePrinter(Z,
                                   indices,
                                   self.getLeafLabel,
                                   self.getNodeLabel
                                  )
        root = None if leaves_list is None else hierarchy.embed_nodes(Z, leaves_list)[-1]
        return '\n'.join([l.replace('-', '  |', 1) if l.startswith('-') else l for l in rp.getLines(root)])

    def getLinkage(self):
        pass
        
    def getLeafLabel(self, node_id):
        pass
        
    def getNodeLabel(self, node_id):
        pass

        
class MarkerCheckTreePrinter(MarkerTreePrinter):
    def __init__(self, profile):
        self._profile = profile
        Z = hierarchy.linkage_from_reachability(self._profile.reachOrder, self._profile.reachDists)
        n = Z.shape[0] + 1
        self._Z = Z
        self._n = n
        ce = MarkerCheckFCE(self._profile, minPts=20, minSize=1000000)
        self._scores = ce.getScores(self._Z)
        self._quals = ce.isLowQualityCluster(self._Z)
        weights = np.concatenate((self._profile.contigLengths, np.zeros(n-1)))
        weights[n:] = hierarchy.maxscoresbelow(Z, weights, fun=np.add)
        flat_ids = hierarchy.flatten_nodes(Z)
        weights[n:] = weights[flat_ids+n]
        self._weights = weights
        self._counts = np.concatenate((np.ones(n), Z[flat_ids, 3]))
        
    def getLinkage(self):
        return self._Z
        
    def getLeafLabel(self, node_id):
        return "'%s" % self._profile.contigNames[node_id]
        
    def getNodeLabel(self, node_id):
        return (":%.2f[%dbp,n=%d]" + ("L" if self._quals[node_id] else "")) % (self._scores[node_id], self._weights[node_id], self._counts[node_id])


###############################################################################
###############################################################################
###############################################################################
###############################################################################
