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
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.cm as plt_cm
from mpl_toolkits.mplot3d import axes3d, Axes3D


# GroopM imports
from utils import makeSurePathExists
from profileManager import ProfileManager
from binManager import BinManager
from classification import ClassificationManager
import distance
import hierarchy
from hierarchy import HierarchyCliqueFinder, ClassificationCoherenceClusterTool

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
             colorMap="HSV",
             prefix="BIN"
            ):
        
        profile = self.loadProfile(timer)

        bm = BinManager(profile)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)

        fplot = BinDistancePlotter(profile, colourmap=colorMap)
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
             prefix="REACH"
            ):
        
        profile = self.loadProfile(timer)

        fplot = HierarchyReachabilityPlotter(profile)
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
        self.plotOnAx(ax)
            
    def plotOnAx(self, ax): pass

    
class Plotter3D(GenericPlotter):
    def plotOnFig(self, fig):
        ax = fig.add_subplot(111, projection='3d')
        self.plotOnAx(ax)
        
    def plotOnAx(self, ax): pass
    
    
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
        self.edgecolours = colours
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
    
    def __call__(self, ax):
        
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
    
    def __call__(self, ax):
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
                 tick_labels=None,
                 xlabel="",
                 ylabel=""):
        self.y = height
        self.colours = colours
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tick_labels = tick_labels
        
    def __call__(self, ax):
        y = self.y
        x = np.arange(len(y))
        bc = ax.bar(x, y, width=1,
               color=self.colours)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xticks(x+0.5)
        ax.set_xticklabels(self.tick_labels)
        
        
class BarPlotter(Plotter2D):
    def __init__(self, *args, **kwargs):
        self.plotOnAx = BarAxisPlotter(*args, **kwargs)
        

# Tree plotters
class HierarchyReachabilityPlotter:
    def __init__(self, profile, threshold=1):
        self._profile = profile
        self._threshold = threshold
        x = sp_distance.pdist(self._profile.covProfiles, metric="euclidean")
        y = sp_distance.pdist(self._profile.kmerSigs, metric="euclidean")
        w = sp_distance.pdist(self._profile.contigLengths[:, None], operator.mul)
        rnorm = np_linalg.norm(distance.argrank((x, y), weights=w, axis=1), axis=0)
        self._heights = distance.density_distance(rnorm, w, 1e8)
        self._order = distance.reachability_order(self._heights)
        
    def plot(self,
             fileName=""):
        
        mapping = self._profile.markers
        cm = ClassificationManager(mapping)
        tick_labels = ['']*len(self._order)
        for (i, ix) in enumerate(mapping.rowIndices):
            tick_labels[ix] = "x"
        
        hplot = BarPlotter(
            height=self._heights[self._order],
            colours=np.where(self._profile.binIds[self._order] != 0, 'r', 'k'),
            xlabel="lineage",
            ylabel="dendist",
            tick_labels=tick_labels)
        hplot.plot(fileName)
        

class HierarchyRemovedPlotter:
    def __init__(self, profile, threshold=1):
        self._profile = profile
        self._threshold = threshold
        self._mapping = self._profile.markers
        self._cm = ClassificationManager(self._mapping)
        x = sp_distance.pdist(self._profile.covProfiles, metric="euclidean")
        y = sp_distance.pdist(self._profile.kmerSigs, metric="euclidean")
        w = sp_distance.pdist(self._profile.contigLengths[:, None], operator.mul)
        rnorm = np_linalg.norm(distance.argrank((x, y), weights=w, axis=1), axis=0)
        ddist = distance.density_distance(rnorm)
        self._Z = sp_hierarchy.single(ddist)
        self._mcf = HierarchyCliqueFinder(self._Z, self._threshold, self._mapping)

        
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
        indices = self._mcf.indices(k)
        coeff = 2*len(self._mcf.maxClique(indices)) - len(indices)
        return '' if count <= 0 else str(coeff)
        
    def leaf_label_count(self, k):
        count = len(self._mcf.indices(k))
        return '' if count == 0 else str(count)
        
    def leaf_label_tag(self, k):
        indices = self._mcf.indices(k)
        q = indices[self._mcf.maxClique(indices)]
        if len(q) == 0:
            return ""
        tags = [[t for t in self._cm.tags(i)] for i in q]
        tag = tags[np.argmax([len(t) for t in tags])]
        return tag[min(6 - self._threshold, len(tag)-1)]

  
# Bin plotters
class BinDistancePlotter:
    def __init__(self, profile, colourmap='HSV'):
        self._profile = profile
        self._colourmap = getColorMap(colourmap)
        x = sp_distance.pdist(self._profile.covProfiles, metric="euclidean")
        y = sp_distance.pdist(self._profile.kmerSigs, metric="euclidean")
        w = sp_distance.pdist(self._profile.contigLengths[:, None], operator.mul)
        scale_factor = 1./w.sum()
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
            bin_squareform_indices = distance.pcoords(bin_indices, n)
            origin = distance.mediod(np.linalg.norm((self._x[bin_squareform_indices], self._y[bin_squareform_indices]), axis=0))
        elif origin=="max_coverage":
            origin = np.argmax(self._profile.normCoverages[bin_indices])
        elif origin=="max_length":
            origin = np.argmax(self._profile.contigLengths[bin_indices])
        else:
            raise ValueError("Invalid `origin` argument parameter value: `%s`" % origin)
        
        indices = distance.ccoords(bin_indices[origin], np.arange(n), n)[0]
        fplot = FeaturePlotter(self._x[indices],
                               self._y[indices],
                               colours=self._c[indices],
                               sizes=20,
                               edgecolours=np.where(self._h[indices], 'r', 'k'),
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
