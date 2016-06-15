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
import distance
from cluster import ClassificationClusterEngine, ProfileDistanceEngine, MarkerCheckEngine
from classification import ClassificationManager
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
                                 removeBins=True,
                                 bids=[0])

    def plot(self,
             timer,
             bids=None,
             origin="mediod",
             colorMap="HSV",
             prefix="BIN",
            ):
            
        profile = self.loadProfile(timer)

        bm = BinManager(profile)
        if bids is None or len(bids) == 0:
            bids = bm.getBids()
        else:
            bm.checkBids(bids)

        print "    Initialising plotter"
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
             prefix="REACH",
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
        
        fileName = "" if self._outDir is None else os.path.join(self._outDir, "%s.png" % prefix)
        fplot.plot(fileName=fileName, bids=bids)
                   
        print "    %s" % timer.getTimeStamp()
        
        
class TreePlotManager:
    """Plot and highlight contigs from a bin"""
    def __init__(self, dbFileName, folder=None):
        self._pm = ProfileManager(dbFileName)
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)
            
    def loadProfile(self, timer):
        return self._pm.loadData(timer, loadBins=True, loadMarkers=True,
                loadReachability=True, removeBins=True, bids=[0])
        
    def plot(self,
             timer,
             prefix="TREE"
            ):
        
        profile = self.loadProfile(timer)

        print "    Initialising plotter"
        fplot = TreePlotter(profile)
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
                 xticklabel_rotation="horizontal",
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
        self.xticklabel_rotation = xticklabel_rotation
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
                           rotation=self.xticklabel_rotation)
        for (x, y, text) in self.text:
            ax.text(x, y, text, ha='center', va='bottom')
                           
        if self.colourbar is not None:
            fig.colorbar(self.colourbar, ax=ax)
        
        
class BarPlotter(Plotter2D):
    def __init__(self, *args, **kwargs):
        self.plotOnAx = BarAxisPlotter(*args, **kwargs)
        

# Tree plotters
class ProfileReachabilityPlotter:
    def __init__(self, profile, colourmap="Sequential"):
        self._profile = profile
        self._cf = ClassificationManager(self._profile.mapping)
        self._colourmap = getColorMap(colourmap)
        
    def plot(self,
             bids,
             label="count",
             highlight="merge",
             fileName=""):
        
        h = self._profile.reachDists
        o = self._profile.reachOrder
        if label=="count":
            iloc = dict(zip(o, range(len(o))))
            (xticks, xticklabels) = zip(*[(iloc[i]+0.5, len(indices)) for (i, indices) in self._profile.mapping.iterindices() if i in iloc])
            xlabel = "count"
            xticklabel_rotation = "horizontal"
        elif label=="tag":
            iloc = dict(zip(o, range(len(o))))
            (xticks, xticklabels) = zip(*[(iloc[i]+0.5, self._cf.consensusTag(indices)) for (i, indices) in self._profile.mapping.iterindices() if i in iloc])
            xlabel = "lineage"
            xticklabel_rotation = "vertical"
        else:
            raise ValueError("Parameter value for 'label' argument must be one of 'count', 'tag'. Got '%s'." % label)
        
        if highlight=="bins":
            # alternate red and black stretches for different bins
            binIds = self._profile.binIds[o]
            binned_indices = np.flatnonzero(binIds > 0)
            flag = np.concatenate(([False], binIds[binned_indices[1:]] != binIds[binned_indices[:-1]], [True]))
            iflag = np.cumsum(flag[:-1])
            colours = np.full(len(o), 'c', dtype="|S1")
            colours[binned_indices] = np.array(['k', 'r'], dtype='|S1')[iflag % 2]
            
            # label stretches with bin ids
            last_indices = np.flatnonzero(flag[1:])
            first_indices = np.concatenate(([0], last_indices[:-1]+1))
            last_binned_indices = binned_indices[last_indices]
            first_binned_indices = binned_indices[first_indices]
            group_centers = (first_binned_indices+last_binned_indices+1)*0.5
            group_heights = np.array([h[s:e+1].max() for (s, e) in zip(first_binned_indices, last_binned_indices)])
            group_labels = binIds[first_binned_indices].astype(str)
            k = np.in1d(binIds[first_binned_indices], bids)
            text = zip(group_centers[k], group_heights[k], group_labels[k])
            smap = None
        elif highlight in ["ratios", "merge"]:
            Z = hierarchy.linkage_from_reachability(o, h)
            n = Z.shape[0]+1
            flat_ids = hierarchy.flatten_nodes(Z)
            ratios = hierarchy.reachability_ratios(Z, o, h)
            splits = hierarchy.reachability_splits(h)
            coeffs = np.ones(n, dtype=float)
            coeffs[splits[:-1]] = ratios[flat_ids]
            minbelowratios = -hierarchy.maxscoresbelow(Z, np.concatenate((-coeffs, -np.ones(n-1, dtype=float))), fun=max)   
            if highlight=="minratios":
                coeffs[splits[:-1]] = minbelowratios
            if highlight in ["ratios", "minratios"]:
                smap = plt_cm.ScalarMappable(cmap=self._colourmap)
                smap.set_array(coeffs)
                colours = smap.to_rgba(coeffs)
            elif highlight=="merge":
                flat_coeffs = MarkerCheckEngine(self._profile).makeScores(Z)
                flat_coeffs[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = 0
                scores = hierarchy.support(Z, flat_coeffs, operator.add)
                is_supported = scores > 0
                is_unsupported = scores < 0
                below_supported = hierarchy.descendents(Z, np.flatnonzero(is_supported)+n, inclusive=True)
                below_supported = below_supported[below_supported >= n] - n
                is_unsupported[below_supported] = False
                print minbelowratios[below_supported].min(), minbelowratios[below_supported].max()
                supported_threshold = minbelowratios[below_supported].min()
                print minbelowratios[is_unsupported].min(), minbelowratios[is_unsupported].max()
                unsupported_threshold = minbelowratios[is_unsupported].max() # don't unbin higher ratios
                ix = np.where(coeffs==unsupported_threshold, 1, np.where(coeffs==supported_threshold, 2, 0))
                colours = np.array(["k", "b", "r"], dtype="|S1")[ix]
                smap = None
            text = []
        elif highlight in ["support", "coeffs", "nzcoeffs"]:
            # color leaves by maximum ancestor coherence score
            Z = hierarchy.linkage_from_reachability(o, h)
            n = Z.shape[0]+1
            flat_ids = hierarchy.flatten_nodes(Z)
            flat_coeffs = MarkerCheckEngine(self._profile).makeScores(Z)
            flat_coeffs[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = 0
            if highlight=="support":
                scores = hierarchy.support(Z, flat_coeffs, operator.add)
                scores = np.where(scores < 0, 0, np.where(scores > 0, 2, 1))
            elif highlight=="nzcoeffs":
                scores = flat_coeffs[n:] > 0
            elif highlight=="coeffs":
                scores = flat_coeffs[n:]
            splits = hierarchy.reachability_splits(h)
            coeffs = np.zeros(n, dtype=float)
            coeffs[splits[:-1]] = scores[flat_ids]
            if highlight=="support":
                colours = np.array(['k', 'b', 'r'], dtype='|S1')[coeffs.astype(int)]
                smap = None
            elif highlight=="nzcoeffs":
                colours = np.array(['k', 'r'], dtype="|S1")[coeffs.astype(int)]
                smap = None
            elif highlight=="coeffs":
                smap = plt_cm.ScalarMappable(cmap=self._colourmap)
                smap.set_array(coeffs)
                colours = smap.to_rgba(coeffs)
            text = []
        else:
            raise ValueError("Parameter value for 'highlight' argument must be one of 'bins', 'support', 'coeffs', 'nzcoeffs', 'ratios'. Got '%s'." % highlight)
        
        
        hplot = BarPlotter(
            height=h,
            colours=colours,
            xlabel=xlabel,
            ylabel="dendist",
            xticks=xticks,
            xticklabels=xticklabels,
            xticklabel_rotation=xticklabel_rotation,
            text=text,
            colourbar=smap,
            )
        hplot.plot(fileName)
        

class TreePlotter:
    def __init__(self, profile, colourmap="Sequential"):
        self._profile = profile
        self._cf = ClassificationManager(self._profile.mapping)
        self._ce = MarkerCheckEngine(self._profile)
        self._colourmap = getColorMap(colourmap)
        self._Z = hierarchy.linkage_from_reachability(self._profile.reachOrder, self._profile.reachDists)
        (_r, self._node_dict) = sp_hierarchy.to_tree(self._Z, rd=True)
        
    def plot(self,
             label="count",
             colour="support",
             fileName=""):
        
        n = self._Z.shape[0]+1
        coeffs = self._ce.makeScores(self._Z)
        T = hierarchy.fcluster_coeffs(self._Z, coeffs, merge="sum")
        (nodes, bids) = sp_hierarchy.leaders(self._Z, T.astype('i'))
        rootancestors = hierarchy.ancestors(self._Z, nodes)
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
            raise ValueError("Parameter value for argument 'label' must be one of 'tag', 'count', 'coeff'. Got '%s'" % label)
            
        if colour=="clusters":
            colour_set = dict([(k, 'r') for k in range(2*n-1) if k not in rootancestors_set])
        elif colour in ["nzcoeffs", "support"]:
            flat_ids = hierarchy.flatten_nodes(self._Z)
            flat_coeffs = coeffs.copy()
            flat_coeffs[n+np.flatnonzero(flat_ids!=np.arange(n-1))] = 0
            support = hierarchy.support(self._Z, flat_coeffs, np.add)
            support = np.concatenate((flat_coeffs[:n], support[flat_ids]))
            if True:
                to_merge = support >= 0
            else:
                to_merge = support > 0
            
            #lowest_ratio = np.max(reach_ratios[np.flatnonzero(is_between[n:])])
            if colour=="support":
                colour_set = dict([(k, 'r' if support[k] > 0 else 'b') for k in np.flatnonzero(support>=0)])
            elif colour=="nzcoeffs":
                colour_set = dict([(k, 'r') for k in np.flatnonzero(flat_coeffs!=0)])
        else:
            raise ValueError("Parameter value for argument 'colour' must be one of 'clusters', 'nzcoeffs', 'support'. Got '%s'" % colour)
           
        
        hplot = DendrogramPlotter(
            self._Z,
            link_colour_func=lambda k: colour_set[k] if k in colour_set else 'k',
            leaf_label_func=leaf_label_func,
            xlabel=xlabel, ylabel="dendist"
        )
        hplot.plot(fileName)
        
    def indices(self, k):
        leaves = self._node_dict[k].pre_order(lambda x: x.get_id())
        return np.flatnonzero(np.in1d(self._profile.mapping.rowIndices, leaves))
        
    def leaf_label_coeff(self, k):
        coeff = self._ce.getScore(self.indices(k))
        return '' if count <= 0 else str(coeff)
        
    def leaf_label_count(self, k):
        count = len(self.indices(k))
        return '' if count == 0 else str(count)
        
    def leaf_label_tag(self, k):
        indices = self.indices(k)
        return self._cf.consensusTag(indices)

  
# Bin plotters
class BinDistancePlotter:
    def __init__(self, profile, colourmap='HSV'):
        self._profile = profile
        self._colourmap = getColorMap(colourmap)
        (self._x, self._y, self._w) = ProfileDistanceEngine().makeDistances(self._profile.covProfiles,
                                                                            self._profile.kmerSigs,
                                                                            self._profile.contigLengths
                                                                            )

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
            w = self._w[bin_condensed_indices]
            origin = distance.mediod(np_linalg.norm((x, y), axis=0) * w)
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
        c = sp_distance.cdist(self._profile.contigGCs[[bi], None], self._profile.contigGCs[:, None], lambda a, b: (a+b)/2)[0]
        h = sp_distance.cdist(self._profile.binIds[[bi], None], self._profile.binIds[:, None], lambda a, b: a!=0 and a==b)[0].astype(bool)
        fplot = FeaturePlotter(x,
                               y,
                               colours=c,
                               sizes=20,
                               edgecolours=np.where(h, 'r', 'k'),
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
