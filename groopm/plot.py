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
from utils import makeSurePathExists, split_contiguous, group_iterator
from profileManager import ProfileManager
from binManager import BinManager
import distance
from cluster import (ProfileDistanceEngine,
                     StreamingProfileDistanceEngine,
                     FileCacher,
                     MarkerCheckCQE,
                     MarkerCheckFCE
                    )
from classification import BinClassifier
import hierarchy

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class BinPlotManager:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, folder=None):
        self._pm = ProfileManager(dbFileName)
        self._dbFileName = dbFileName
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)

    def loadProfile(self, timer):
        return self._pm.loadData(timer,
                                 loadMarkers=False,
                                 loadBins=True,
                                 loadReachability=True)

    def plot(self,
             timer,
             bids=None,
             origin="mediod",
             colorMap="HSV",
             prefix="BIN",
             surface=False,
             savedDistsPrefix="",
             keepDists=False
            ):
            
        profile = self.loadProfile(timer)

        bm = BinManager(profile)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)

        if savedDistsPrefix=="":
            savedDistsPrefix = self._dbFileName+".dists"
        cacher = FileCacher(savedDistsPrefix)

        print "    Initialising plotter"
        
        fplotter = BinDistancePlotter4D if surface else BinDistancePlotter
        fplot = fplotter(profile, colourmap=colorMap, cacher=cacher)
        print "    %s" % timer.getTimeStamp()
        
        
        for bid in bids:
            fileName = "" if self._outDir is None else os.path.join(self._outDir, "%s_%d.png" % (prefix, bid))

            fplot.plot(fileName=fileName,
                       origin=origin,
                       bid=bid)
                       
            if fileName=="":
                break
        if self._outDir is not None:
            print "    %s" % timer.getTimeStamp()
        
        if not keepDists:
            try:
                cacher.cleanup()
            except:
                raise

        
class ReachabilityPlotManager:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, folder=None):
        self._pm = ProfileManager(dbFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)
            
    def loadProfile(self, timer):
        return self._pm.loadData(timer, loadBins=True, loadMarkers=True,
                loadReachability=True)

        return profile
        
    def plot(self,
             timer,
             bids=None,
             label="tag",
             filename="REACH.png",
            ):
        
        profile = self.loadProfile(timer)

        bm = BinManager(profile)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)
            
        print "    Initialising plotter"
        fplot = ProfileReachabilityPlotter(profile)
        print "    %s" % timer.getTimeStamp()
        
        if os.path.splitext(filename)[1] != '.png':
            filename+='.png'
            
        fileName = "" if self._outDir is None else os.path.join(self._outDir, filename)
        fplot.plot(fileName=fileName, bids=bids, label=label)
                   
        if self._outDir is not None:
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
                 colourbar=None,
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
        self.colourbar = colourbar
    
    def __call__(self, ax, fig):
        sp_hierarchy.dendrogram(self.Z, ax=ax, p=4000,
                                truncate_mode='lastp',
                                #distance_sort='ascending',
                                color_threshold=0,
                                leaf_label_func=self.leaf_label_func,
                                link_color_func=self.link_colour_func)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        
        if self.colourbar is not None:
            fig.colorbar(self.colourbar, ax=ax)
        
    
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
                 text_alignment="center",
                 text_rotation="horizontal",
                 colourbar=None):
        self.y = height
        self.colours = colours
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticks = xticks
        self.xticklabels = xticklabels
        self.text = text
        self.text_alignment = text_alignment
        self.text_rotation = text_rotation
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
        ax.tick_params(axis="x", length=2, direction="out", width=1, top='off')
        for (x, y, text) in self.text:
            ax.text(x, y, text, va="bottom",
                                ha=self.text_alignment,
                                rotation=self.text_rotation,
                                rotation_mode="anchor")
                           
        if self.colourbar is not None:
            fig.colorbar(self.colourbar, ax=ax)
        
        
class BarPlotter(Plotter2D):
    def __init__(self, *args, **kwargs):
        self.plotOnAx = BarAxisPlotter(*args, **kwargs)
        

# GroopM plotting tools
class ProfileReachabilityPlotter:
    """
    Bar plot of reachability distances between points in traversal order.
    """
    def __init__(self, profile):
        self._profile = profile
        self._bc = BinClassifier(self._profile.mapping)
        
    def plot(self,
             bids,
             label="tag",
             fileName=""):
                 
        n = 0
        for i in range(self._profile.numContigs):
            n += self._profile.contigLengths[i]*self._profile.contigLengths[i+1:].sum()
        h = self._profile.reachDists
        o = self._profile.reachOrder
        
        # ticks with empty labels for contigs with marker hits
        iloc = dict(zip(o, range(len(o))))
        (xticks, xticklabels) = zip(*[(iloc[i], "") for (i,_) in self._profile.mapping.iterindices() if i in iloc])
        xlabel = ""
            
        # colour contigs by bin
        binIds = self._profile.binIds[o]
        flag_first = np.concatenate(([True], binIds[1:] != binIds[:-1])) # first of equal value run
        first_indices = np.flatnonzero(flag_first)
        last_indices = np.concatenate((first_indices[1:], [len(o)]))
        #(first_indices, last_indices) = split_contiguous(binIds)
        is_bin = binIds[flag_first]!=0
        first_binned_indices = first_indices[is_bin]
        last_binned_indices = last_indices[is_bin]
        
        # alternate colouring of stretches for different bins
        # red and black for selected bins, greys for unselected, cyan for unbinned
        colour_ids = np.zeros(len(o), dtype=np.int)
        is_selected_bin = np.in1d(binIds[first_binned_indices], bids)
        is_unselected_bin = np.logical_not(is_selected_bin)
        for (i, (s, e)) in enumerate(zip(first_binned_indices[is_unselected_bin], last_binned_indices[is_unselected_bin])):
            colour_ids[s:e] = (i%2) + 1
        for (i, (s, e)) in enumerate(zip(first_binned_indices[is_selected_bin], last_binned_indices[is_selected_bin])):
            colour_ids[s:e] = (i%2) + 3
        colours = np.array(['c', '0.5', '0.7', 'k', 'r'])[colour_ids]
        
        # label stretches with bin ids
        group_centers = (first_binned_indices+last_binned_indices)*0.5
        group_heights = np.array([h[s:e].max() for (s, e) in zip(first_binned_indices, last_binned_indices)])
        if label=="bids":
            group_labels = binIds[first_binned_indices].astype(str)
            text_alignment = "center"
            text_rotation = "horizontal"
        elif label=="tag":
            mapping_bids = binIds[self._profile.mapping.rowIndices]
            group_labels = np.array([self._bc.consensusTag(np.flatnonzero(mapping_bids==bid)) for bid in binIds[first_binned_indices]])
            text_alignment = "right"
            text_rotation = -60
        else:
            raise ValueError("Parameter value for 'label' argument must be one of 'bid', 'tag'. Got '%s'." % label)
    
        text = zip(group_centers[is_selected_bin], group_heights[is_selected_bin], group_labels[is_selected_bin])
        
        
            
        hplot = BarPlotter(
            height=h[1:],
            colours=colours[1:],
            xlabel=xlabel,
            ylabel="reachability dist",
            xticks=xticks,
            xticklabels=xticklabels,
            text=text,
            text_alignment=text_alignment,
            text_rotation=text_rotation
            )
        hplot.plot(fileName)

  
# Bin plotters
class BinDistancePlotter:
    def __init__(self, profile, colourmap='HSV', cacher=None):
        self._profile = profile
        self._colourmap = getColorMap(colourmap)
        if cacher is None:
            de = ProfileDistanceEngine()
        else:
            de = StreamingProfileDistanceEngine(cacher=cacher, size=int(2**31-1))
        (self._x, self._y) = de.makeRanks(self._profile.covProfiles,
                                          self._profile.kmerSigs,
                                          self._profile.contigLengths
                                         )
        scale_factor = 2. / (self._profile.contigLengths.sum()**2-(self._profile.contigLengths**2).sum())
        self._x *= scale_factor
        self._y *= scale_factor
        
    def plot(self,
             bid,
             origin,
             fileName=""):
        
        n = self._profile.numContigs
        bin_indices = BinManager(self._profile).getBinIndices(bid)
        if origin=="mediod":
            (i, j) = distance.pairs(len(bin_indices))
            bin_condensed_indices = distance.condensed_index(n, bin_indices[i], bin_indices[j])
            x = self._x[bin_condensed_indices]
            y = self._y[bin_condensed_indices]
            origin = distance.mediod((x**2 + y**2)**(1./2))
        elif origin=="max_coverage":
            origin = np.argmax(self._profile.normCoverages[bin_indices])
        elif origin=="max_length":
            origin = np.argmax(self._profile.contigLengths[bin_indices])
        else:
            raise ValueError("Invalid `origin` argument parameter value: `%s`" % origin)
        
        bi = bin_indices[origin]
        not_bi = np.array([i for i in range(n) if i!=bi])
        condensed_indices = distance.condensed_index(n, bi, not_bi)
        x = np.zeros(n, dtype=float)
        x[not_bi] = self._x[condensed_indices]
        y = np.zeros(n, dtype=float)
        y[not_bi] = self._y[condensed_indices]
        c = self._profile.contigGCs
        h = np.logical_and(self._profile.binIds == self._profile.binIds[bi], self._profile.binIds[bi] != 0)
        fplot = FeaturePlotter(x,
                               y,
                               colours=c,
                               sizes=20,
                               edgecolours=np.where(h, 'r', 'k'),
                               colourmap=self._colourmap,
                               xlabel="cov", ylabel="kmer")
        fplot.plot(fileName)
        
        
class BinDistancePlotter4D:
    def __init__(self, profile, colourmap='HSV', cacher=None):
        self._profile = profile
        self._colourmap = getColorMap(colourmap)
        if cacher is None:
            de = ProfileDistanceEngine()
        else:
            de = StreamingProfileDistanceEngine(cacher=cacher, size=int(2**31-1))
        (self._x, self._y) = de.makeRanks(self._profile.covProfiles,
                                          self._profile.kmerSigs,
                                          self._profile.contigLengths
                                         )
        scale_factor = 2. / (self._profile.contigLengths.sum()**2-(self._profile.contigLengths**2).sum())
        self._x *= scale_factor
        self._y *= scale_factor
        
    def plot(self,
             bid,
             origin,
             fileName=""):
        
        n = self._profile.numContigs
        bin_indices = BinManager(self._profile).getBinIndices(bid)
        if origin=="mediod":
            (i, j) = distance.pairs(len(bin_indices))
            bin_condensed_indices = distance.condensed_index(n, bin_indices[i], bin_indices[j])
            x = self._x[bin_condensed_indices]
            y = self._y[bin_condensed_indices]
            origin = distance.mediod((x**2 + y**2)**(1./2))
        elif origin=="max_coverage":
            origin = np.argmax(self._profile.normCoverages[bin_indices])
        elif origin=="max_length":
            origin = np.argmax(self._profile.contigLengths[bin_indices])
        else:
            raise ValueError("Invalid `origin` argument parameter value: `%s`" % origin)
        
        bi = bin_indices[origin]
        not_bi = np.array([i for i in range(n) if i!=bi])
        condensed_indices = distance.condensed_index(n, bi, not_bi)
        x = np.zeros(n, dtype=float)
        x[not_bi] = self._x[condensed_indices]
        y = np.zeros(n, dtype=float)
        y[not_bi] = self._y[condensed_indices]
        z = self._profile.normCoverages.flatten()
        c = self._profile.contigGCs
        h = np.logical_and(self._profile.binIds == self._profile.binIds[bi], self._profile.binIds[bi] != 0)
        fplot = SurfacePlotter(x,
                               y,
                               z=z,
                               colours=c,
                               sizes=20,
                               edgecolours=np.where(h, 'r', 'k'),
                               colourmap=self._colourmap,
                               xlabel="cov", ylabel="kmer", zlabel="cov_norm")
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
    elif colorMapStr == 'Sequential':
        return plt_cm.get_cmap('copper')
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