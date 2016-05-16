#!/usr/bin/env python
###############################################################################
#                                                                             #
#    plot.py                                                                  #
#                                                                             #
#    Data visualisation                                                       #
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
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"
__status__ = "Development"

###############################################################################
import os
import sys
import colorsys
import operator
import numpy as np
import numpy.linalg as np_linalg
import scipy.spatial.distance as sp_distance
import scipy.cluster.hierarchy as sp_hierarchy
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.cm as plt_cm
from mpl_toolkits.mplot3d import axes3d, Axes3D


# GroopM imports
from utils import makeSurePathExists
from profileManager import ProfileManager
from binManager import BinManager
from classification import ClassificationManager, ClassificationConsensusFinder
import distance
import hierarchy
from hierarchy import ClassificationLeavesLister, ClassificationCoherenceClusterTool

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
class BinPlotter:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, folder=None):
        self._pm = ProfileManager(dbFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)

    def loadProfile(self, timer):
        return self._pm.loadData(timer, loadBins=True, minLength=0, removeBins=True, bids=[0])

    def plot(self,
             timer,
             bids=None,
             origin="mediod",
             doWeight=True,
             colorMap="HSV",
             prefix="BIN"
            ):
        
        profile = self.loadProfile(timer)

        bm = BinManager(profile)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)

        fplot = BinDistancePlotter(profile, doWeight=doWeight, colourmap=colorMap)
        print "    %s" % timer.getTimeStamp()
        
        for bid in bids:
            fileName = "" if self._outDir is None else os.path.join(self._outDir, "%s_%d.png" % (prefix, bid))

            fplot.plot(fileName=fileName,
                       origin=origin,
                       bid=bid)
                       
            if fileName=="":
                break
        print "    %s" % timer.getTimeStamp()

        
class ReachabilityPlotter:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, markerFileName, folder=None):
        self._pm = ProfileManager(dbFileName, markerFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)
            
    def loadProfile(self, timer):
        return self._pm.loadData(timer, loadBins=True, loadMarkers=True, minLength=0,
                removeBins=True, bids=[0])
        
    def plot(self,
             timer,
             ks,
             pts,
             linear=False,
             doWeight=True,
             bids=None,
             prefix="REACH",
            ):
        
        profile = self.loadProfile(timer)

        bm = BinManager(profile)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)
            
        fplot = HierarchyReachabilityPlotter(profile, threshold=1, doWeight=doWeight)
        print "    %s" % timer.getTimeStamp()
        
        if ks is None and pts is None:
            pairs = [(None, None)]
        elif ks is None:
            pairs = [(p, None) for p in pts]
        elif pts is None:
            ks = [(None, k) for k in ks]
        else:
            pairs = [(p, k) for p in pts for k in ks]
        for (p, k) in pairs:
            if self._outDir is None:
                fileName = ""
            else:
                file_str = prefix
                if k is not None:
                    file_str += "_k=%g" % k
                if p is not None:
                    file_str += "_p=%d" % p
                file_str += ".png"
                fileName = os.path.join(self._outDir, file_str)
            fplot.plot(fileName=fileName,
                       bids=bids,
                       smooth=k,
                       minPts=p,
                       linear=linear)
                       
            if fileName == "":
                break
        print "    %s" % timer.getTimeStamp()
        
        
class TreePlotter:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, markerFileName, folder=None):
        self._pm = ProfileManager(dbFileName, markerFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)
            
    def loadProfile(self, timer):
        return self._pm.loadData(timer, loadBins=True, loadMarkers=True, minLength=0,
                removeBins=True, bids=[0])
        
    def plot(self,
             timer,
             doWeight=True,
             prefix="TREE"
            ):
        
        profile = self.loadProfile(timer)

        fplot = HierarchyRemovedPlotter(profile, threshold=1, doWeight=doWeight)
        print "    %s" % timer.getTimeStamp()
        
        fileName = "" if self._outDir is None else os.path.join(self._outDir, "%s.png" % prefix)
        fplot.plot(fileName=fileName)
        print "    %s" % timer.getTimeStamp()

        
# Basic plotting tools
class GenericPlotter:
    def plot(self, fileName=""):
        # plot contigs in coverage space
        fig = plt.figure()

        self.plotOnFig(fig)

        if(fileName != ""):
            try:
                fig.set_size_inches(15,15)
                plt.savefig(fileName,dpi=300)
            except:
                print "Error saving image:", fileName, sys.exc_info()[0]
                raise
        else:
            print "Plotting"
            try:
                plt.show()
            except:
                print "Error showing image", sys.exc_info()[0]
                raise

        plt.close(fig)
        del fig
        
    def plotOnFig(self, fig): pass


class Plotter2D(GenericPlotter):
    def plotOnFig(self, fig):
        ax = fig.add_subplot(111)
        self.plotOnAx(ax, fig)
            
    def plotOnAx(self, ax, fig): pass

    
class Plotter3D(GenericPlotter):
    def plotOnFig(self, fig):
        ax = fig.add_subplot(111, projection='3d')
        self.plotOnAx(ax, fig)
        
    def plotOnAx(self, ax, fig): pass
    
    
# Plot types
class FeatureAxisPlotter:
    def __init__(self, x, y,
                 colours,
                 sizes,
                 colourmap,
                 edgecolours,
                 z=None,
                 xlabel="", ylabel="", zlabel=""):
        """
        Parameters
        ------
        x: array_like, shape (n,)
        y: array_like, shape (n,)
        colours: color or sequence of color
        sizes: scalar or array_like, shape (n,)
        colourmap: Colormap
        edgecolours: color or sequence of color
        z: array_like, shape (n,), optional
        xlabel: string, optional
        ylabel: string, optional
        zlabel: string, optional
        """
        self.x = x
        self.y = y
        self.z = z
        self.sizes = sizes
        self.colours = colours
        self.colourmap = colourmap
        self.edgecolours = edgecolours
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
    
    def __call__(self, ax, fig):
        
        coords = (self.x, self.y)
        if self.z is not None:
            coords += (self.z,)
        sc = ax.scatter(*coords,
                        c=self.colours, s=self.sizes,
                        cmap=self.colourmap,
                        vmin=0., vmax=1., marker='.')                        
        sc.set_edgecolors(self.edgecolours)
        sc.set_edgecolors = sc.set_facecolors = lambda *args:None
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if len(coords) == 3:
            ax.set_zlabel(self.zlabel)

            
class FeaturePlotter(Plotter2D): 
    def __init__(self, *args, **kwargs):
        self.plotOnAx = FeatureAxisPlotter(*args, **kwargs)


class SurfacePlotter(Plotter3D):
    def __init__(self, *args, **kwargs):
        self.plotOnAx = FeatureAxisPlotter(*args, **kwargs)
            

class DendrogramAxisPlotter:
    def __init__(self, Z,
                 link_colour_func,
                 leaf_label_func,
                 xlabel="", ylabel=""):
        """
        Fields
        ------
        Z: linkage matrix, shape(n, 4)
        link_colour_func: callable
        leaf_label_func: callable
        xlabel: string
        ylabel: string
        """
        self.Z = Z
        self.link_colour_func = link_colour_func
        self.leaf_label_func = leaf_label_func
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def __call__(self, ax, fig):
        sp_hierarchy.dendrogram(self.Z, ax=ax, p=4000,
                                truncate_mode='lastp',
                                distance_sort='ascending',
                                color_threshold=0,
                                leaf_label_func=self.leaf_label_func,
                                link_color_func=self.link_colour_func)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        
    
class DendrogramPlotter(Plotter2D):
    def __init__(self, *args, **kwargs):
        self.plotOnAx = DendrogramAxisPlotter(*args, **kwargs)
        
        
class BarAxisPlotter:
    def __init__(self,
                 height,
                 colours,
                 xticks=[],
                 xticklabels=[],
                 xlabel="",
                 ylabel="",
                 text=[],
                 colourbar=None):
        self.y = height
        self.colours = colours
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticks = xticks
        self.xticklabels = xticklabels
        self.text = text
        self.colourbar = colourbar
        
    def __call__(self, ax, fig):
        y = self.y
        x = np.arange(len(y))
        bc = ax.bar(x, y, width=1,
               color=self.colours,
               linewidth=0)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels,
                           rotation="horizontal")
        for (x, y, text) in self.text:
            ax.text(x, y, text, ha='center', va='bottom')
                           
        if self.colourbar is not None:
            fig.colorbar(self.colourbar, ax=ax)
        
        
class BarPlotter(Plotter2D):
    def __init__(self, *args, **kwargs):
        self.plotOnAx = BarAxisPlotter(*args, **kwargs)
        

# Tree plotters
class HierarchyReachabilityPlotter:
    def __init__(self, profile, threshold, doWeight, colourmap="Blues"):
        self._profile = profile
        self._threshold = threshold
        self._colourmap = getColorMap(colourmap)
        #sorting_indices = np.flipud(self._profile.contigLengths.argsort())
        x = sp_distance.pdist(self._profile.covProfiles, metric="euclidean")
        y = sp_distance.pdist(self._profile.kmerSigs, metric="euclidean")
        if doWeight:
            w = sp_distance.pdist(self._profile.contigLengths[:, None], operator.mul)
        else:
            w = None
        rnorm = np_linalg.norm(distance.argrank((x, y), weights=w, axis=1), axis=0)
        self._dists = rnorm
        self._weights = w
        #self._order = sorting_indices
        self._mapping = self._profile.markers
        self._cm = ClassificationManager(self._mapping)
        self._cf = ClassificationConsensusFinder(self._mapping, self._threshold)
        
    def plot(self,
             bids,
             smooth,
             minPts,
             linear,
             label="count",
             highlight="markers",
             fileName=""):
                 
        if smooth is None:
            minWt = None
        elif linear:
            minWt = smooth * self._profile.contigLengths[self._order]
        else:
            minWt = np.full(self._profile.numContigs, smooth)
        dd = distance.density_distance(self._dists, weights=self._weights, minWt=minWt, minPts=minPts)
        (o, d) = distance.reachability_order(dd)
        
        x = d[o]
        #o = self._order[o]
        if label=="count":
            iloc = dict(zip(o, range(len(o))))
            (xticks, xticklabels) = zip(*[(iloc[i]+0.5, len(indices)) for (i, indices) in self._cm.iterindices()])
            xlabel = "count"
        elif label=="tag":
            iloc = dict(zip(o, range(len(o))))
            (xticks, xticklabels) = zip(*[(iloc[i]+0.5, self._cf.consensusTag(indices)) for (i, indices) in self._cm.iterindices()])
            xlabel = "lineage"
        else:
            raise ValueError("Invalid `label` argument parameter value: `%s`" % label)
        
        if highlight=="bins":
            # alternate red and black stretches for different bins
            binIds = self._profile.binIds[o]
            flag = np.concatenate(([False], binIds[1:] != binIds[:-1], [True]))
            iflag = np.cumsum(flag[:-1])
            colours = np.array(['k', 'r'], dtype='|S1')[iflag % 2]
            colours[binIds==0] = 'c'
            
            # label stretches with bin ids
            group_ends = np.flatnonzero(flag[1:])
            group_centers = np.concatenate(([group_ends[0]/2], (group_ends[1:]+group_ends[:-1]+1)/2))
            group_heights = np.concatenate(([x[:group_ends[0]+1].max()], [x[s:e+1].max() for (s, e) in zip(group_ends[:-1]+1, group_ends[1:])]))
            group_labels = binIds[group_ends].astype(str)
            k = np.in1d(binIds[group_ends], bids)
            text = zip(group_centers[k], group_heights[k], group_labels[k])
            smap = None
        elif highlight=="markers":
            # color leaves by first non-zero ancestor coherence score
            scores = np.zeros(self._profile.numContigs)
            Z = hierarchy.linkage_from_reachability(o, d)
            ll = ClassificationLeavesLister(Z, self._mapping)
            (_r, node_dict) = sp_hierarchy.to_tree(Z, rd=True)
            mnodes = ll.nodes
            for k in mnodes:
                ix = ll.leaves_list(k)
                score = self._cf.disagreement(ix)
                if score > 0:
                    row_indices = np.array(node_dict[k].pre_order(lambda x: x.get_id()))
                    scores[row_indices[scores[row_indices]==0]] = score
            
            scores = scores[o]
            norm = plt_colors.Normalize(vmin=0, vmax=np.max(scores))
            smap = plt_cm.ScalarMappable(norm=norm, cmap=self._colourmap)
            smap.set_array(scores)
            colours = smap.to_rgba(scores)
            text = []
        else:
            raise ValueError("Invalid `highlight` argument parameter value: `%s`" % highlight)
        
        
        hplot = BarPlotter(
            height=x,
            colours=colours,
            xlabel=xlabel,
            ylabel="dendist",
            xticks=xticks,
            xticklabels=xticklabels,
            text=text,
            colourbar=smap,
            )
        hplot.plot(fileName)
        

class HierarchyRemovedPlotter:
    def __init__(self, profile, threshold, doWeight):
        self._profile = profile
        self._threshold = threshold
        self._mapping = self._profile.markers
        self._cm = ClassificationManager(self._mapping)
        x = sp_distance.pdist(self._profile.covProfiles, metric="euclidean")
        y = sp_distance.pdist(self._profile.kmerSigs, metric="euclidean")
        if doWeight:
            w = sp_distance.pdist(self._profile.contigLengths[:, None], operator.mul)
        else:
            w = None
        rnorm = np_linalg.norm(distance.argrank((x, y), weights=w, axis=1), axis=0)
        ddist = distance.density_distance(rnorm)
        self._Z = sp_hierarchy.single(ddist)
        self._mcf = ClassificiationConsensusFinder(self._mapping, self._threshold)
        self._mll = ClassificationLeavesLister(self._Z, self._mapping)

        
    def plot(self,
             label="count",
             fileName=""):
        
        (L, M) = sp_hierarchy.leaders(self._Z, self._profile.binIds)
        L = np.concatenate((L[M!=0], np.flatnonzero(T==0)))
        rootancestors = hierarchy.ancestors(self._Z, L)
        rootancestors_set = set(rootancestors)
        if label=="tag":
            leaf_label_func=lambda k: '' if k in rootancestors_set else self.leaf_label_tag(k)
            xlabel="lineage"
        elif label=="count":
            leaf_label_func=self.leaf_label_count
            xlabel="count"
        elif label=="coeff":
            leaf_label_func=self.leaf_label_coeff
            xlabel="coeff"
        else:
            raise ValueError("Invalid `label` argument parameter value: `%s`" % label)
        
        hplot = DendrogramPlotter(
            self._Z,
            link_colour_func=lambda k: 'k' if k in rootancestors_set else 'r',
            leaf_label_func=leaf_label_func,
            xlabel=xlabel, ylabel="dendist"
        )
        hplot.plot(fileName)
        
    def leaf_label_coeff(self, k):
        indices = self._mll.indices(k)
        coeff = self._mcf.disagreement(indices)
        return '' if count <= 0 else str(coeff)
        
    def leaf_label_count(self, k):
        count = len(self._mll.indices(k))
        return '' if count == 0 else str(count)
        
    def leaf_label_tag(self, k):
        indices = self._mll.indices(k)
        return self._mcf.consensusTag(indices)

  
# Bin plotters
class BinDistancePlotter:
    def __init__(self, profile, doWeight, colourmap='HSV'):
        self._profile = profile
        self._colourmap = getColorMap(colourmap)
        x = sp_distance.pdist(self._profile.covProfiles, metric="euclidean")
        y = sp_distance.pdist(self._profile.kmerSigs, metric="euclidean")
        if doWeight:
            w = sp_distance.pdist(self._profile.contigLengths[:, None], operator.mul)
            scale_factor = 1./w.sum()
        else:
            w = None
            scale_factor = 1./len(x)
        self._x = distance.argrank(x, weights=w)*scale_factor
        self._y = distance.argrank(y, weights=w)*scale_factor
        self._w = w
        self._c = sp_distance.pdist(self._profile.contigGCs[:, None], lambda a, b: (a+b)/2)
        self._h = sp_distance.pdist(self._profile.binIds[:, None], lambda a, b: a!=0 and a==b).astype(bool)
        
    def plot(self,
             bid,
             origin,
             fileName=""):
        
        n = self._profile.numContigs
        bin_indices = BinManager(self._profile).getBinIndices(bid)
        if origin=="mediod":
            bin_condensed_indices = [distance.condensed_index(n, bi, bj) for (i, bi) in enumerate(bin_indices[:-1]) for bj in bin_indices[i+1:]]
            #bin_squareform_indices = distance.pcoords(bin_indices, n)
            origin = distance.mediod(np_linalg.norm((self._x[bin_condensed_indices], self._y[bin_condensed_indices]), axis=0))
        elif origin=="max_coverage":
            origin = np.argmax(self._profile.normCoverages[bin_indices])
        elif origin=="max_length":
            origin = np.argmax(self._profile.contigLengths[bin_indices])
        else:
            raise ValueError("Invalid `origin` argument parameter value: `%s`" % origin)
        
        bi = bin_indices[origin]
        condensed_indices = [distance.condensed_index(n, bi, i) for i in range(n) if i != bi]
        fplot = FeaturePlotter(self._x[condensed_indices],
                               self._y[condensed_indices],
                               colours=self._c[condensed_indices],
                               sizes=20,
                               edgecolours=np.where(self._h[condensed_indices], 'r', 'k'),
                               colourmap=self._colourmap,
                               xlabel="cov", ylabel="kmer")
        fplot.plot(fileName)
        

#------------------------------------------------------------------------------
# Helpers
    
def getColorMap(colorMapStr):
    if colorMapStr == 'HSV':
        S = 1.0
        V = 1.0
        return plt_colors.LinearSegmentedColormap.from_list('GC', [colorsys.hsv_to_rgb((1.0 + np.sin(np.pi * (val/1000.0) - np.pi/2))/2., S, V) for val in xrange(0, 1000)], N=1000)
    elif colorMapStr == 'Accent':
        return plt_cm.get_cmap('Accent')
    elif colorMapStr == 'Blues':
        return plt_cm.get_cmap('Blues')
    elif colorMapStr == 'Spectral':
        return plt_cm.get_cmap('spectral')
    elif colorMapStr == 'Grayscale':
        return plt_cm.get_cmap('gist_yarg')
    elif colorMapStr == 'Discrete':
        discrete_map = [(0,0,0)]
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))

        discrete_map.append((0,0,0))
        discrete_map.append((141/255.0,211/255.0,199/255.0))
        discrete_map.append((255/255.0,255/255.0,179/255.0))
        discrete_map.append((190/255.0,186/255.0,218/255.0))
        discrete_map.append((251/255.0,128/255.0,114/255.0))
        discrete_map.append((128/255.0,177/255.0,211/255.0))
        discrete_map.append((253/255.0,180/255.0,98/255.0))
        discrete_map.append((179/255.0,222/255.0,105/255.0))
        discrete_map.append((252/255.0,205/255.0,229/255.0))
        discrete_map.append((217/255.0,217/255.0,217/255.0))
        discrete_map.append((188/255.0,128/255.0,189/255.0))
        discrete_map.append((204/255.0,235/255.0,197/255.0))
        discrete_map.append((255/255.0,237/255.0,111/255.0))
        discrete_map.append((1,1,1))

        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        return plt_colors.LinearSegmentedColormap.from_list('GC_DISCRETE', discrete_map, N=20)

    elif colorMapStr == 'DiscretePaired':
        discrete_map = [(0,0,0)]
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))

        discrete_map.append((0,0,0))
        discrete_map.append((166/255.0,206/255.0,227/255.0))
        discrete_map.append((31/255.0,120/255.0,180/255.0))
        discrete_map.append((178/255.0,223/255.0,138/255.0))
        discrete_map.append((51/255.0,160/255.0,44/255.0))
        discrete_map.append((251/255.0,154/255.0,153/255.0))
        discrete_map.append((227/255.0,26/255.0,28/255.0))
        discrete_map.append((253/255.0,191/255.0,111/255.0))
        discrete_map.append((255/255.0,127/255.0,0/255.0))
        discrete_map.append((202/255.0,178/255.0,214/255.0))
        discrete_map.append((106/255.0,61/255.0,154/255.0))
        discrete_map.append((255/255.0,255/255.0,179/255.0))
        discrete_map.append((217/255.0,95/255.0,2/255.0))
        discrete_map.append((1,1,1))

        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        discrete_map.append((0,0,0))
        return plt_colors.LinearSegmentedColormap.from_list('GC_DISCRETE', discrete_map, N=20)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
