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
import os.path
import tables
import sys
import tempfile

# local imports
import distance
import hierarchy
import stream
from profileManager import ProfileManager
from groopmExceptions import SavedDistancesInvalidNumberException, CacheUnavailableException

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
            savedDistsPrefix = self._dbFileName+".dists"
        cacher = FileCacher(savedDistsPrefix)
        
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
        d : ndarray
            1-D array. `d[i]` is the core distance of the `i`th observation.
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
        self._cacher = cacher # None to disable streaming / caching
    
    def distances(self, silent=False, fun=lambda a: a):
        if(not silent):
            n = len(self._profile.contigLengths)
            print "Computing pairwise contig distances for 2^%.2f pairs" % np.log2(n*(n-1)//2)

        if self._cacher is None:
            de = ProfileDistanceEngine()
        else:
            de = StreamingProfileDistanceEngine(cacher=self._cacher, size=int(2**31-1))
            de_ = ProfileDistanceEngine()
        
        # add psuedo-counts
        #covProfiles = self._profile.covProfiles + 100. / self._profile.contigLengths[:, None]
        #covProfiles = distance.logratio(covProfiles, axis=1, mode="centered")
        #kmerSigs = self._profile.kmerSigs + 1. / (self._profile.contigLengths[:, None] - 3)
        kmerSigs = self._profile.kmerSigs * (self._profile.contigLengths[:, None] - 3) + 1
        kmerSigs = distance.logratio(kmerSigs, axis=1, mode="centered")
        stat = de.makeRankStat(self._profile.covProfiles,
                               kmerSigs,
                               self._profile.contigLengths,
                               silent=silent,
                               fun=fun,
                               )
        
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
        core_dists = distance.core_distance(stat, weight_fun=lambda i,j: self._profile.contigLengths[i]*self._profile.contigLengths[j], minWt=minWt, minPts=self._minPts)
        
        return (stat, core_dists)
    
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
    
    def makeRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        """Compute pairwise rank distances separately for coverage profiles and
        kmer signatures, and give rank distances as a fraction of the largest rank.
        """
        n = len(contigLengths)
        weights = np.empty( n * (n-1) // 2, dtype=np.double)
        k = 0
        for i in range(n-1):
            weights[k:(k+n-1-i)] = contigLengths[i]*contigLengths[(i+1):n]
            k = k+n-1-i
        weight_fun = lambda i: weights[i]
        cov_ranks = distance.argrank(sp_distance.pdist(covProfiles, metric="euclidean"), weight_fun=weight_fun)
        kmer_ranks = distance.argrank(sp_distance.pdist(kmerSigs, metric="euclidean"), weight_fun=weight_fun)
        return (cov_ranks, kmer_ranks)
    
    def makeRankStat(self, covProfiles, kmerSigs, contigLengths, silent=False, fun=lambda a: a):
        """Compute norms in {coverage rank space x kmer rank space}
        """
        (cov_ranks, kmer_ranks) = self.makeRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        dists = fun(cov_ranks) + fun(kmer_ranks)
        return dists
        

class StreamingProfileDistanceEngine:
    """Class for computing profile feature distances. Does caching to disk to keep memory usage down."""

    def __init__(self, cacher, size):
        self._cacher = cacher
        self._size = size
        self._store = TempFileStore()
            
    def _getWeightFun(self, contigLengths):
        n = len(contigLengths)
        def weight_fun(k):
            (i, j) = distance.squareform_coords(n, k)
            weights = i
            weights[:] = contigLengths[i]
            weights[:] *= contigLengths[j]
            return weights
        return weight_fun
            
    def _calculateRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        """Compute pairwise rank distances separately for coverage profiles and
        kmer signatures, and give rank distances as a fraction of the largest rank.
        """
        n = len(contigLengths)
        weight_fun = None
        try:
            cov_ranks = self._cacher.get("cov")
            assert_num_obs(n, cov_ranks)
        except CacheUnavailableException:
            weight_fun = self._getWeightFun(contigLengths)
            if not silent:
                print "Calculating coverage distance ranks"
            
            cov_filename = self._store.getWorkingFile()
            covind_filename = self._store.getWorkingFile()
            stream.pdist_chunk(covProfiles, cov_filename, chunk_size=2*self._size, metric="euclidean")
            cov_ranks = stream.argrank_chunk(cov_filename, covind_filename, weight_fun=weight_fun, chunk_size=self._size)
            self._store.cleanupWorkingFiles()
            self._cacher.store("cov", cov_ranks)
        del cov_ranks
        try:
            kmer_ranks = self._cacher.get("kmer")
            assert_num_obs(n, kmer_ranks)
        except CacheUnavailableException:
            if weight_fun is None:
                weight_fun = self._getWeightFun(contigLengths)
            if not silent:
                print "Calculating tetramer distance ranks"
            kmer_filename = self._store.getWorkingFile()
            kmerind_filename = self._store.getWorkingFile()
            stream.pdist_chunk(kmerSigs, kmer_filename, chunk_size=2*self._size, metric="euclidean")
            kmer_ranks = stream.argrank_chunk(kmer_filename, kmerind_filename, weight_fun=weight_fun, chunk_size=self._size)
            self._store.cleanupWorkingFiles()
            self._cacher.store("kmer", kmer_ranks)
        del kmer_ranks
    
    def makeRanks(self, covProfiles, kmerSigs, contigLengths, silent=False):
        self._calculateRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        return (self._cacher.get("cov"), self._cacher.get("kmer"))
    
    def makeRankStat(self, covProfiles, kmerSigs, contigLengths, silent=False, fun=lambda a: a):
        """Compute norms in {coverage rank space x kmer rank space}
        """
        self._calculateRanks(covProfiles, kmerSigs, contigLengths, silent=silent)
        #return self._cacher.get("cov")*self._cacher.get("kmer")
        
        dists_file = self._store.getWorkingFile()
        fun(self._cacher.get("cov")).tofile(dists_file)
        tmp_file = self._store.getWorkingFile()
        self._cacher.get("kmer").tofile(tmp_file)
        fold = lambda a, b: a+fun(b)
        stream.iapply_func_chunk(dists_file, tmp_file, fold, chunk_size=self._size)
        dists = np.fromfile(dists_file, dtype=np.double)
        self._store.cleanupWorkingFiles()
        return dists
        
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################  
class Cacher:
    """Abstract Class for caching profile feature distances.
    Subclass should provide `distances` and `fcluster` methods to the
    interface outlined below.
    """
    
    def cleanup(self, silent):
        pass
        
    def get(self, key):
        """
        Parameters
        ----------
        key : String
        
        Returns
        -------
        value : ndarray
        
        """
        pass
        
    def store(self, key, value):
        """
        Parameters
        ----------
        key : string
        value : ndarray
        
        """
        pass
     
     
class TempFileStore:
    """Create and clean up temp files"""
    
    def __init__(self):
        self._workingFiles = []
        
    def getWorkingFile(self):
        (_f, filename) = tempfile.mkstemp(prefix="groopm.working", dir=os.getcwd())
        self._workingFiles.append(filename)
        return filename
    
    def _cleanupOne(self, filename):
        try:
            os.remove(filename)
        except OSError:
            pass
        
    def cleanupWorkingFiles(self):
        try:
            while True:
                f = self._workingFiles.pop()
                self._cleanupOne(f)
        except IndexError:
            pass
        
        
class FileCacher(Cacher):
    """Cache using numpy to/fromfile"""
    
    def __init__(self, distStorePrefix, keys=['cov', 'kmer']):
        self._prefix = distStorePrefix
        self._stores = dict([(k, self._prefix+"."+k) for k in keys])
        self._owned = set()
        
    def _cleanupOne(self, filename):
        try:
            os.remove(filename)
        except OSError:
            pass
            
    def get(self, key):
        try:
            vals = np.fromfile(self._stores[key], dtype=np.double)
        except IOError:
            raise CacheUnavailableException()
        return vals
        
    def store(self, key, values):
        if not os.path.lexists(self._stores[key]):
            self._owned.add(key)
        np.asanyarray(values, dtype=np.double).tofile(self._stores[key])
        
    def cleanup(self, silent=False):
        for key in self._owned:
            if not silent:
                print("removing distance store {0}".format(self._stores[key]))
            self._cleanupOne(self._stores[key]) 
        
        
        
class TablesCacher(Cacher):
    """Cache using pytable"""

    def __init__(self, distStore, keys=["cov", "kmer"], descriptions=["Coverage distances", "Tetramer distances"]):
        self._distStoreFile = distStore
        try:
            with tables.open_file(self._distStoreFile, mode="a", title="Distance store") as h5file:
                pass
        except:
            print "Error creating database:", self._distStoreFile, sys.exc_info()[0]
            raise
        self._tables = dict(zip(keys, descriptions))
        
            
    def cleanup(self):
        try:
            os.remove(self._distStoreFile)
        except OSError:
            pass
    
    def get(self, key):
        try:
            with tables.open_file(self._distStoreFile, mode="r") as h5file:
                vals = h5file.get_node("/", key).read()
        except tables.exceptions.NoSuchNodeError:
            raise CacheUnavailableException()
        return vals
        
    def store(self, key, values):
        with tables.open_file(self._distStoreFile, mode="a") as h5file:
            h5file.create_array("/", key, values, self._tables[key])

        

def assert_num_obs(n, y):
    if n != sp_distance.num_obs_y(y):
        raise SavedDistancesInvalidNumberException("Saved distances for different number of observations")
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
class FlatClusterEngine:
    """Flat clustering pipeline.
    
    Subclass should provide `getScores` and `isNoiseCluster` methods with the
    described interfaces.
    """
    
    def unbinClusters(self, unbin, out_bins):
        out_bins[unbin] = 0
        (_, new_bids) = np.unique(out_bins[out_bins != 0], return_inverse=True)
        out_bins[out_bins != 0] = new_bids+1
    
    
    def makeClusters(self,
                     Z,
                     return_leaders=False,
                     return_noise=False,
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
            - Identify 'mergeable' clusters where at most one child cluster
              exceeds the minimum standard
            - Merge clusters if the combined cluster increases the global
              sum quality or is mergeable and does not decrease the sum
              quality
        
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
        
        scores = np.asarray(self.getScores(Z))
        # NOTE: Cases of nested clusters where the child and parent cluster heights
        # are equal are ambiguously encoded in hierarchical clusterings. 
        # This is accounted for in the following way:
        #   1. Find nodes that do not have strictly lower height to their parents.
        #   2. Set the quality scores of these clusters to a value that will ensure
        #      the sum of descendent scores are propagated.
        #   3. Merge these clusters only if the highest equal height ancestor is
        #      merged.
        flat_ids = hierarchy.flatten_nodes(Z)
        scores[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = np.min(scores)
        
        # NOTE: support is a measure of the degree to which a cluster quality
        # improves on the combined quality of the best clusters below (computed
        # by maxscorebelow), and represents the net benefit merging the cluster.
        # Positive values indicate that the cluster should be favoured compared
        # to the best combination of clusters below, zero values indicate no
        # preference, and negative values indicate that the best clusters below
        # should be prefered.
        support = scores[n:] - hierarchy.maxscoresbelow(Z, scores, operator.add)
        
        is_noise_cluster = np.asarray(self.isNoiseCluster(Z))
        #is_noise_cluster[n:] = is_noise_cluster[n+flat_ids]
        
        # NOTE: conservative bins are a minimal set of clusters that have
        # maximum combined quality. The returned conservative bins have below
        # standard bins dissolved.
        to_merge = support>0
        to_merge = to_merge[flat_ids]
        (conservative_bins, conservative_leaders) = hierarchy.fcluster_merge(Z, to_merge, return_nodes=True)
        conservative_bins += 1 # bin ids start at 1
        self.unbinClusters(is_noise_cluster[conservative_leaders], out_bins=conservative_bins)
        
        # NOTE: A seed cluster is any putative cluster that meets the minimum
        # standard, is not below a conservative cluster, and is not equal in height
        # to their parent cluster. The algorithm requires that seed clusters be
        # removed below conservative clusters so conservative clusters can absorb
        # non-seed clusters.
        is_seed_cluster = np.logical_not(is_noise_cluster)
        is_seed_cluster[hierarchy.descendents(Z, conservative_leaders)] = False
        is_seed_cluster[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = False # propagates descendent seed count to equal height parents
        
        # NOTE: In this step we find the largest possible clusters seed clusters
        # without merging any two seed clusters. 
        to_merge = hierarchy.maxscoresbelow(Z, is_seed_cluster.astype(int), fun=operator.add)<=1
        #to_merge = np.logical_or(support > 0,
        #                         np.logical_and(support > -self.support_tol, is_mergable))
        to_merge = to_merge[flat_ids]
        (bins, leaders) = hierarchy.fcluster_merge(Z, to_merge, return_nodes=True)
        bins += 1 # bin ids start at 1
        # NOTE: any new clusters that are below minimum standard are dissolved
        # here. 
        self.unbinClusters(is_noise_cluster[leaders], out_bins=bins)
                     
        if not (return_leaders or return_noise or return_support or return_coeffs or
                return_seeds or return_conservative_bins or return_conservative_leaders):
            return bins
            
        out = (bins,)
        if return_leaders:
            out += (leaders,)
        if return_noise:
            out += (is_noise_cluster,)
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
        """Compute cluster scores for nodes in a hierarchical clustering.
        
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
            `scores[i]` where `i >= n` is the score for the cluster consisting
            of the original observations below the node represented by the
            `i-n`th row of `Z`.
        """
        pass #subclass to override
        
    def isNoiseCluster(self, Z):
        """Indicate clusters that are e.g. too small,  not enough bp, to be 
        legit clusters by themselves.
        
        Parameters
        ----------
        Z : ndarray
            Linkage matrix encoding a hierarchical clustering.
        
        Returns
        -------
        l : ndarray
            1-D boolean array. `l[i]` is True for clusters that are
            'noise' and otherwise False. Here `i` is a singleton 
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
            
    def getScores(self, Z):
        return MarkerCheckCQE(self._profile).makeScores(Z)
        
    def isNoiseCluster(self, Z):
        Z = np.asarray(Z)
        n = sp_hierarchy.num_obs_linkage(Z)
        flat_ids = hierarchy.flatten_nodes(Z)
        
        # Quality clusters have total contig length at least minSize (if
        # defined) or a number of contigs at least minPts (if defined)
        doMinSize = self._minSize is not None
        doMinPts = self._minPts is not None
        if not doMinSize and not doMinPts:
            return np.zeros(2*n-1, dtype=bool)
        if doMinSize:
            weights = np.concatenate((self._profile.contigLengths, np.zeros(n-1)))
            weights[n:] = hierarchy.maxscoresbelow(Z, weights, fun=operator.add)
            weights[n:] = weights[flat_ids+n]
            is_noise = weights < self._minSize   
        if doMinPts:
            is_below_minPts = np.concatenate((np.full(n, 1 < self._minPts, dtype=bool), Z[flat_ids, 3] < self._minPts))
            if doMinSize:
                is_noise = np.logical_and(is_noise, is_below_minPts)
            else:
                is_noise = is_below_minPts
                
        return is_noise
        
              
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
            
        # Bottom-up traversal of hierarchy, concatenating the sets of leaves 
        # of children and computing the score for each node
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
            A dictionary with leaf ids as keys and lists of associated leaf node data as values.
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
    find groups of items that act as a particular item's category. Single copy
    markers can be used to distinguish when items should be in different bins.
    
    With these factors in mind, we have used the following adjusted BCubed 
    metrics:
        - For an item, other taxonomically similar items are considered to share
          the item's category.
        - For an item, a probability of being grouped with another item is 
          assigned as being 1 over the number of ways to pair two items from
          the first item's category that share the markers of the input pair. 
        - Precision for an item represents the average probability of being paired
          with another item in the same cluster.
        - Recall for an item represents the relative likelihood of being paired with
          an item in the same cluster to the expected number of pairings in the
          unclustered dataset.
          
    These metrics are combined in a naive linear manner using a mixing fraction
    alpha which can be combined additively when assessing multiple clusters.
    """
    
    def __init__(self, profile):
        self._d = 1
        self._alpha = 0.5
        self._mapping = profile.mapping
        
        # Taxonomic connectivity matrix: L[i,j] = 1 where mapping i and j are 
        # taxonomically 'similar enough', otherwise 0.
        self._L = sp_distance.squareform(self._mapping.classification.makeDistances()) < self._d
        
        # Marker connectivity matrix: M[i, j] = 1 where mapping i and j are
        # from the same marker, otherwise 0.
        markerNames = self._mapping.markerNames
        self._M = markerNames[:, None] == markerNames[None, :]
        
        # Compute the compatible marker group sizes
        gsizes = np.array([np.count_nonzero(row) for row in self._L])
        gweights = np.array([np.unique(markerNames[row], return_counts=True)[1].max() for row in self._L])
        #self._gscalefactors = gweights * 1. / gsizes
        self._gscalefactors = 1. / self._getProbs(self._L, self._M)
        self._gscalefactors_ = 1. / np.array([ (1. / np.maximum(self._M[i, row].sum(), self._M[np.ix_(row, row)].sum(axis=1))).sum() for (i, row) in enumerate(self._L)])
        assert np.all(self._gscalefactors == self._gscalefactors_)
    
    def _getProbs(self, L, M):
        return np.array([ (1. / np.maximum(M[i, row].sum(), M[np.ix_(row, row)].sum(axis=1))).sum() for (i, row) in enumerate(L)])
    
    def getLeafData(self):
        """Leaf data is a list of indices of mappings."""
        return dict([(i, data) for (i, data) in self._mapping.iterindices()])
        
    def getScore(self, indices):
        """Compute modified BCubed completeness and precision scores."""
        indices = np.asarray(indices)
        
        probs = self._getProbs(self._L[np.ix_(indices, indices)], self._M[np.ix_(indices, indices)])
        #probs = np.array([ (1. / np.maximum(self._M[i, indices[row]].sum(), self._M[np.ix_(indices[row], indices[row])].sum(axis=1))).sum() for (i, row) in enumerate(self._L[np.ix_(indices, indices)])])
        
        # weighted item precision
        prec = (probs * 1. / len(indices)).sum()
        #prec = np.sum([weights[i] * (weights[self._L[index, indices]].sum() + 1 - weights[i]) * 1. / len(indices) for (i, index) in enumerate(indices)])
        
        # weighted item completeness / recall
        recall = (probs * self._gscalefactors[indices]).sum()
        #recall = np.sum([weights[i] * (weights[self._L[index, indices]].sum() + 1 - weights[i]) * self._gscalefactors[index] for (i, index) in enumerate(indices)])
        
        f = self._alpha * recall + (1 - self._alpha) * prec
        return f

       
###############################################################################
###############################################################################
###############################################################################
###############################################################################
                
class _RecursiveTreePrinter:
    """Helper class for TreePrinter
    
    Prints the embedded tree generated by an array of leaf node indices.
    Labels are generated using the highest original tree nodes below the parent
    embedded node.
    """
    
    def __init__(self, Z, indices, leaf_labeller, node_labeller):
        self._Z = np.asarray(Z)
        self._n = sp_hierarchy.num_obs_linkage(self._Z)
        self._flat_ids = hierarchy.flatten_nodes(self._Z)
        self._embed_ids = hierarchy.embed_nodes(self._Z, indices)
        self._indices = indices
        self._leaf_labeller = leaf_labeller
        self._node_labeller = node_labeller

    def getLines(self, node_id=None):
        """
        Get a list of lines for each embedded tree node below a given node
        """
        n = self._n
        if node_id is None:
            node_id = self._embed_ids[-1]
            
        # NOTE: Cases of nested clusters where the child and parent cluster heights
        # are equal are ambiguously encoded in hierarchical clusterings. 
        # This is accounted for in the following way:
        #   1. Find nodes that do not have strictly lower height to their parents.
        #   2. Embed the highest equal height ancestor of an embedded node only if
        #      the node is the embedded id of the ancestor.
        
        # Find the highest embedded node below the current node
        embed_id = self._embed_ids[node_id-n] if node_id >= n else node_id
        
        # Find the highest equal height ancestor of the current node
        flat_id = self._flat_ids[node_id-n]+n if node_id >= n else node_id
        
        # Embedding when the highest embedded node below the ancestor is the same
        # guarantees that the ancestor is embedded at most once
        embed = node_id < n or self._embed_ids[flat_id-n] == embed_id
        #embed = node_id < n or self._flat_embed_ids[flat_id-n] == embed_id
        
        if embed:
            # compute the next lines using the children of the embedded descendent
            node_id = embed_id
            
        if node_id < n:
            if node_id==-1 or not np.any(node_id == self._indices): # not embedded
                return []
            else:   
                # embedded leaf 
                return [self._node_labeller(flat_id)+self._leaf_labeller(node_id)]
        else:
            left_child = int(self._Z[node_id-n,0])
            right_child = int(self._Z[node_id-n,1])
            child_lines = self.getLines(left_child)+self.getLines(right_child)
            if embed:
                # label using the flat id of the current node
                return [self._node_labeller(flat_id)]+['--'+l for l in child_lines]
            else:
                return child_lines
        

class TreePrinter:
    def printTree(self, indices, leaves_list=None):
        Z = self.getLinkage()
        Z = np.asarray(Z)
        rp = _RecursiveTreePrinter(Z,
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

        
class MarkerCheckTreePrinter(TreePrinter):
    def __init__(self, profile):
        self._profile = profile
        Z = hierarchy.linkage_from_reachability(self._profile.reachOrder, self._profile.reachDists)
        self._Z = Z
        self._n = sp_hierarchy.num_obs_linkage(self._Z)
        ce = MarkerCheckFCE(self._profile, minPts=20, minSize=1000000)
        self._scores = ce.getScores(self._Z)
        self._is_noise = ce.isNoiseCluster(self._Z)
        n = self._n
        weights = np.concatenate((self._profile.contigLengths, np.zeros(n-1)))
        weights[n:] = hierarchy.maxscoresbelow(Z, weights, fun=np.add)
        #flat_ids = hierarchy.flatten_nodes(Z)
        #weights[n:] = weights[flat_ids+n]
        self._weights = weights
        self._counts = np.concatenate((np.ones(n), Z[:, 3]))
        
    def getLinkage(self):
        return self._Z
        
    def getLeafLabel(self, node_id):
        return "'%s" % self._profile.contigNames[node_id]
        
    def getNodeLabel(self, node_id):
        return (":%.2f[%dbp,n=%d]" + ("N" if self._is_noise[node_id] else "")) % (self._scores[node_id], self._weights[node_id], self._counts[node_id])


###############################################################################
###############################################################################
###############################################################################
###############################################################################
