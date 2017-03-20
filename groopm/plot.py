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
import matplotlib.colorbar as plt_colorbar
import matplotlib.lines as plt_lines
import matplotlib.markers as plt_markers
from mpl_toolkits.mplot3d import axes3d, Axes3D


# GroopM imports
from utils import makeSurePathExists, split_contiguous, group_iterator
from groopmExceptions import BinNotFoundException
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
from data3 import ClassificationEngine
from extract import BinReader

np.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ExplorePlotManager:
    """Plot and highlight contigs from bins near a contig"""
    def __init__(self, dbFileName, folder=None):
        self._pm = ProfileManager(dbFileName)
        self._dbFileName = dbFileName
        self._outDir = os.getcwd() if folder == "" else folder
        # make the dir if need be
        if self._outDir is not None:
            makeSurePathExists(self._outDir)

    def loadProfile(self, timer):
        return self._pm.loadData(timer,
                                 loadMarkers=True,
                                 loadBins=True,
                                 loadReachability=True)

    def plot(self,
             timer,
             bids=None,
             origin="mediod",
             colorMap="HSV",
             prefix="BIN",
             surface=False,
             rawDistances=False,
             groupfile="",
             separator=",",
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
        fplot = ContigExplorerPlotter(profile,
                                      colourmap=colorMap,
                                      cacher=cacher,
                                      surface=surface,
                                      rawDistances=rawDistances
                                     )
        print "    %s" % timer.getTimeStamp()
        
        group_list = None
        if groupfile!="":
            group_list = GroupAssignmentParser().parse(groupfile, separator, profile.contigNames)
            
        first_plot = True
        queue = []
        for i in range(len(bids)-1,-1,-1):
            queue.append(bids[i])
            
        while len(queue) > 0:
            if self._outDir is not None:
                bid = queue.pop()
                fileName = os.path.join(self._outDir, "%s_%d.png" % (prefix, bid))
            else:
                if not first_plot:
                    cbid = self.promptOnPlot(queue[-1], bm)
                    if cbid is None:
                        break
                    if cbid!=queue[-1]:
                        queue.append(cbid)
                bid = queue.pop()
                fileName = ""
                
            first_plot = False
            cid = fplot.getBinRepresentative(bid, mode=origin)
            is_in_bin = profile.binIds==bid
            highlight_markers = np.unique(profile.mapping.markerNames[is_in_bin[profile.mapping.rowIndices]])
            highlight_groups = [] if group_list is None else np.unique(group_list[is_in_bin])
            
            fplot.plot(fileName=fileName,
                       contig=cid,
                       bid=bid,
                       highlight_markers=highlight_markers,
                       highlight_groups=highlight_groups,
                       group_list=group_list)
                    
        if self._outDir is not None:
            print "    %s" % timer.getTimeStamp()
        
        if not keepDists:
            print("nuking stored distances")
            try:
                cacher.cleanup()
            except:
                raise
                
    def promptOnPlot(self, bid, bm, minimal=False):
        """Check that the user wants to continue interactive plotting"""
        input_not_ok = True
        while(input_not_ok):
            if(minimal):
                option = raw_input(" Enter bin id, or q to quit, or enter to continue:")
            else:
                option = raw_input(""" The next plot is {0}
 Enter bin id, or q to quit, or enter to continue:""".format(bid))
            try:
                bid = int(option)
                bm.checkBids([bid])
                print "****************************************************************"
                return bid
            except ValueError:
                if option.upper() in ["Q", ""]:
                    print "****************************************************************"
                    if(option.upper() == "Q"):
                        print("Operation cancelled")
                        return None
                    else:
                        return bid
                else:
                    print("Error, unrecognised choice '{0}'".format(option))
                    minimal=True
            except BinNotFoundException:
                print("Error, no bin with id '{0}'".format(bid))
                minimal=True
            

        
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
             groupfile="",
             separator=""
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
            
        group_list = None
        if groupfile!="":
            group_list = GroupAssignmentParser().parse(groupfile, separator, profile.contigNames)
            
        fileName = "" if self._outDir is None else os.path.join(self._outDir, filename)
        is_in_bin = np.in1d(profile.binIds, bids)
        highlight_markers = np.unique(profile.mapping.markerNames[is_in_bin[profile.mapping.rowIndices]])
        highlight_groups = [] if group_list is None else np.unique(group_list[is_in_bin])
        
        fplot.plot(fileName=fileName,
                   bids=bids,
                   highlight_markers=highlight_markers,
                   highlight_groups=highlight_groups,
                   label="bid")
                   
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
                 edgecolours,
                 markers,
                 legend_data=None,
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
        self.edgecolours = edgecolours
        self.markers = markers
        self.legend_data = legend_data
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
    
    def __call__(self, ax, fig):
        
        coords = (self.x, self.y)
        if self.z is not None:
            coords += (self.z,)
        
        marker_sets = [(slice(None), '.')] if self.markers is None else self.markers
        for (ix, marker) in marker_sets:
            sc = ax.scatter(*[x[ix] for x in coords],
                            c=self.colours[ix],
                            s=self.sizes[ix],
                            marker=marker)                        
            sc.set_edgecolors(self.edgecolours[ix])
            sc.set_edgecolors = sc.set_facecolors = lambda *args:None
        
        if self.legend_data is not None and len(self.legend_data)>0:
            (labels, data) = zip(*self.legend_data)
            proxies = [plt_lines.Line2D([0], [0], linestyle="none", 
                                        markersize=15, **dat) for dat in data]
            ax.legend(proxies, labels, numpoints=1)
        
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
                 vlines=[],
                 legend_data=None,
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
        self.vlines = vlines
        self.legend_data = legend_data
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
        for (x, colour, linestyle) in self.vlines:
            ax.axvline(x, linestyle=linestyle, c=colour)
        
        if self.legend_data is not None and len(self.legend_data)>0:
            (labels, color) = zip(*self.legend_data)
            proxies = [plt_patches.Rectangle((0,0), 0, 0, fill=True,
                                             color=clr) for clr in color]
            ax.legend(proxies, labels, numpoints=1)
                           
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
             highlight_groups=[],
             highlight_markers=[],
             highlight_taxstring=[],
             group_list=None,
             label="tag",
             fileName=""):
                 
        n = 0
        for i in range(self._profile.numContigs):
            n += self._profile.contigLengths[i]*self._profile.contigLengths[i+1:].sum()
        h = self._profile.reachDists
        o = self._profile.reachOrder
        
        # ticks with empty labels for contigs with marker hits
        he = ProfileHighlightEngine(self._profile)
        (tick_groups, tick_labels) = he.getHighlighted(markers=highlight_markers, highlight_per_marker=False)
        indices = np.flatnonzero(tick_groups[:,0])
        if len(indices)>0:
            iloc = dict(zip(o, range(len(o))))
            (xticks, xticklabels) = zip(*[(iloc[i], "") for i in np.flatnonzero(tick_groups[:,0]) if i in iloc])
        else:
            xticks = []
            xticklabels = []
        xlabel = "contigs in traversal order"
        
        # colouring based on group membership
        colourmap = getColorMap('Highlight2')
        (group_ids, group_labels) = he.getHighlighted(groups=highlight_groups, group_list=group_list)
        colours = colourmap(group_ids[:,0])
        legend_data = [(format_label(label), colourmap(i)) for (label, i) in zip(group_labels, range(1, len(group_labels)+1))]
        
            
        # find bin contigs
        obids = self._profile.binIds[o]
        (first_binned_indices, last_binned_indices) = split_contiguous(obids, filter_groups=[0])
        
        
        # alternate colouring of lines for different bins
        # red and black for selected bins, greys for unselected
        is_selected_bin = np.in1d(obids[first_binned_indices], bids)
        is_unselected_bin = np.logical_not(is_selected_bin)
        linecolour_ids = []
        linestyle_ids = []
        line_positions = []
        for (i, (s, e)) in enumerate(zip(first_binned_indices[is_unselected_bin], last_binned_indices[is_unselected_bin])):
            linecolour_ids.extend([(i%2)+1, (i%2)+1])
            linestyle_ids.extend([0, 1])
            line_positions.extend([s+1, e-1])
        for (i, (s, e)) in enumerate(zip(first_binned_indices[is_selected_bin], last_binned_indices[is_selected_bin])):
            linecolour_ids.extend([(i%2)+3, (i%2)+3])
            linestyle_ids.extend([0, 1])
            line_positions.extend([s+1, e-1])
        linecolours = np.array(['c', '0.5', '0.7', 'k', 'r'])[linecolour_ids]
        linestyles = np.array([':', '-'])[linestyle_ids]
        vlines = zip(line_positions, linecolours, linestyles)
        
        # label stretches with bin ids
        group_centers = (first_binned_indices+1+last_binned_indices)*0.5
        #group_heights = np.array([h[s:e].max() for (s, e) in zip(first_binned_indices, last_binned_indices)])
        if label=="bids":
            group_labels = obids[first_binned_indices].astype(str)
            text_alignment = "center"
            text_rotation = "horizontal"
        elif label=="tag":
            mapping_bids = self._profile.binIds[self._profile.mapping.rowIndices]
            group_labels = np.array(["?" if tag=="" else tag for tag in (self._bc.consensusTag(np.flatnonzero(mapping_bids==bid)) for bid in obids[first_binned_indices])])
            text_alignment = "right"
            text_rotation = -60
        else:
            raise ValueError("Parameter value for 'label' argument must be one of 'bid', 'tag'. Got '%s'." % label)
    
        text = zip(group_centers[is_selected_bin], [h.max()]*np.count_nonzero(is_selected_bin), group_labels[is_selected_bin])
        
        hplot = BarPlotter(
            height=h[1:],
            colours=colours[1:],
            xlabel=xlabel,
            ylabel="reachability dist",
            xticks=xticks,
            xticklabels=xticklabels,
            text=text,
            text_alignment=text_alignment,
            text_rotation=text_rotation, 
            vlines=vlines,
            legend_data=legend_data,
            )
        hplot.plot(fileName)

  
class ContigExplorerPlotter:
    def __init__(self, profile, colourmap='HSV', cacher=None, rawDistances=False, surface=False, fun=lambda a: a):
        self._profile = profile
        self._colourmap = getColorMap(colourmap)
        self._surface = surface
        self._rawDistances = rawDistances
        self._fun = fun
        
        covProfiles = self._profile.covProfiles
        kmerSigs = self._profile.kmerSigs * (self._profile.contigLengths[:, None] - 3) + 1
        kmerSigs = distance.logratio(kmerSigs, axis=1, mode="centered")
        if self._rawDistances:
            def getCoords(i,j):
                x = np.log(sp_distance.cdist(covProfiles[[i]], covProfiles[[j]], metric="euclidean"))
                y = sp_distance.cdist(kmerSigs[[i]], kmerSigs[[j]], metric="euclidean")
                return (x, y)
            self._getCoords = getCoords
            self._xlabel = "log(d_cov)"
            self._ylabel = "d_clr(kmer)"
        else:
            if cacher is None:
                de = ProfileDistanceEngine()
            else:
                de = StreamingProfileDistanceEngine(cacher=cacher, size=int(2**31-1))
                
            (x, y) = de.makeRanks(covProfiles,
                                  kmerSigs,
                                  self._profile.contigLengths
                                 )
            scale_factor = 200. / (self._profile.contigLengths.sum()**2-(self._profile.contigLengths**2).sum())
            x *= scale_factor
            y *= scale_factor
            n = self._profile.numContigs
            
            def getCoords(i, j):
                condensed_indices = distance.condensed_index(n, i, j)
                return (x[condensed_indices], y[condensed_indices])
            self._getCoords = getCoords
            self._xlabel = "percentile(d_cov)"
            self._ylabel = "percentile(d_clr(kmer))"
            
    def getBinRepresentative(self, bid, mode="max_length"):  
        indices = np.flatnonzero(self._profile.binIds == bid)
        if mode=="mediod":
            if self._rawDistances:
                raise ValueError("`mode` argument parameter value `mediod` is not appropriate for ContigExplorerPlotter with `rawDistances` flag set.")
            (i, j) = distance.pairs(len(indices))
            (x, y) = self._getCoords(indices[i], indices[j])
            choice = distance.mediod(self._fun(x) + self._fun(y))
        elif mode=="max_coverage":
            choice = np.argmax(self._profile.normCoverages[indices])
        elif mode=="max_length":
            choice = np.argmax(self._profile.contigLengths[indices])
        elif mode!="mediod":
            raise ValueError("Invalid `mode` argument parameter value: `%s`" % mode)
        
        return self._profile.contigNames[indices[choice]]
        
    
    def plot(self,
             contig,
             bid,
             highlight_groups=[],
             highlight_markers=[],
             highlight_taxstrings=[],
             group_list=None,
             fileName=""):
        
        n = self._profile.numContigs
        try:
            origin = np.flatnonzero(self._profile.contigNames==contig)[0]
        except IndexError:
            raise ContigNotFoundException("ERROR: No contig found in database with id {0}.".format(contig))
        
        # hard error if highlight bids don't exist
        bm = BinManager(self._profile)
        bin_indices = bm.getBinIndices([bid])
        
        # load distances
        others = np.array([i for i in range(n) if i!=origin])
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)
        (x[others], y[others]) = self._getCoords(origin, others)
        
        s = 20*(2**np.log10(self._profile.contigLengths / np.min(self._profile.contigLengths)))
        
        # colorize
        he = ProfileHighlightEngine(self._profile)
        (edge_groups, edge_labels) = he.getHighlighted(bids=[bid])
        (marker_groups, marker_labels) = he.getHighlighted(groups=highlight_groups,
                                                           group_list=group_list)
        (colour_groups, colour_labels) = he.getHighlighted(markers=highlight_markers,
                                                           highlight_per_marker=False)
        legend_data = []
        edgecolourmap = getColorMap('Highlight1')
        edgecolours = edgecolourmap(edge_groups[:,0])
        legend_data.extend([(format_label(label), dict(markeredgecolor=edgecolourmap(i), c="w", marker=".")) for (i, label) in enumerate(edge_labels, 1)])
        
        sm = plt_cm.ScalarMappable(plt_colors.Normalize(vmin=0., vmax=1.), self._colourmap)
        c = sm.to_rgba(self._profile.contigGCs)
        is_coloured = colour_groups[:,0]>0
        colourmap = getColorMap('Highlight2')
        c[is_coloured] = colourmap(colour_groups[is_coloured,0])
        is_coloured_plain_edge = np.logical_and(is_coloured, edge_groups[:,0]==0)
        edgecolours[is_coloured_plain_edge] = c[is_coloured_plain_edge]
        legend_data.extend([(format_label(l), dict(c=colourmap(i), marker=".")) for (i, l) in enumerate(colour_labels, 1)])
        
        marker_list = ['o', '^', 's', 'v', 'D']
        markers = [(marker_groups[:,0]==i, marker_list[i % len(marker_list)]) for i in range(len(marker_labels)+1)]
        legend_data.extend([(format_label(l), dict(marker=marker_list[i % len(marker_list)], c="w")) for (i, l) in enumerate(marker_labels, 1)])
        
        # apply visual transformation
        xlabel = self._xlabel
        ylabel = self._ylabel
        if not self._rawDistances:
            x = np.sqrt(self._fun(x))
            y = np.sqrt(self._fun(y))
            xlabel = "sqrt({0})".format(xlabel)
            ylabel = "sqrt({0})".format(ylabel)
        
        if self._surface:
            z = self.profile.normCoverages.flatten()
            fplot = SurfacePlotter(x,
                                   y,
                                   z=z,
                                   colours=c,
                                   sizes=s,
                                   edgecolours=edgecolours,
                                   legend_data=legend_data,
                                   xlabel=xlabel, ylabel=ylabel, zlabel="cov_norm")            
        else:
            fplot = FeaturePlotter(x,
                                   y,
                                   colours=c,
                                   sizes=s,
                                   edgecolours=edgecolours,
                                   markers=markers,
                                   legend_data=legend_data,
                                   xlabel=xlabel, ylabel=ylabel)
        fplot.plot(fileName)


class GroupManager:
    def __init__(self, n, mask=None):
        self._n = n
        self._mask = mask
        self._labels = []
        self._group_members = []
        
    def addGroup(self, indices, label):
        is_member = np.zeros(self._n, dtype=bool)
        is_member[indices] = True if self._mask is None else self._mask[indices]
        if np.any(is_member):
            self._group_members.append(is_member)
            self._labels.append(label)
            
    def getGroups(self):
        if len(self._group_members)==0:
            return (np.zeros((self._n, 1), dtype=int), np.array([]))
            
        ngroups = len(self._group_members)
        group_ids = np.transpose(self._group_members).astype(int) * np.arange(1,ngroups+1)[None,:]
        group_ids[group_ids==0] = ngroups+1
        sorted_group_ids = np.sort(group_ids, axis=1)
        sorted_group_ids[sorted_group_ids==ngroups+1] = 0
        
        return (sorted_group_ids, self._labels)
         
    def getGroupIntersections(self):
        if len(self._group_members)==0:
            return (np.zeros(self._n, dtype=int), np.array([]))
        flipped_group_members = [m for m in self._group_members]
        flipped_group_members.reverse()
        order = np.lexsort(flipped_group_members)
        sorted_group_intersections = np.fliplr(np.array(flipped_group_members).T[order])
        # flag first unique group intersections
        flag_first = np.concatenate(([np.any(sorted_group_intersections[0])],
                                     np.any(sorted_group_intersections[1:]!=sorted_group_intersections[:-1], axis=1)))
        group_intersection_ids = np.empty(self._n, dtype=int)
        group_intersection_ids[order] = np.cumsum(flag_first)
        labels = np.array(self._labels)
        group_intersection_labels = np.array(["/".join(labels[row]) for row in sorted_group_intersections[flag_first]])
        
        
        # reverse priority of group intersections
        nzids = group_intersection_ids!=0
        group_intersection_ids *= -1
        group_intersection_ids[nzids] += np.count_nonzero(flag_first)+1
        group_intersection_labels = np.flipud(group_intersection_labels)
        return (group_intersection_ids[:, None], group_intersection_labels)
        
        
class ProfileHighlightEngine:
    def __init__(self, profile):
        self._profile = profile
       
    def getHighlighted(self,
                       bids=[],
                       markers=[],
                       taxstrings=[],
                       groups=[],
                       group_list=None,
                       mask=None,
                       highlight_per_bid=True,
                       highlight_per_marker=True,
                       highlight_per_group=True,
                       highlight_per_taxstring=True,
                       highlight_intersections=False):
        
        # verify highlight inputs
        if group_list is not None:
            groups = np.asarray(groups)
            missing_groups = np.in1d(groups, group_list, invert=True)
            if np.any(missing_groups):
                print ("WARNING: No contig(s) assigned to group(s) {0}.".format(",".join(groups[missing_groups])))
        markers = np.asarray(markers)
        missing_markers = np.in1d(markers, self._profile.mapping.markerNames, invert=True)
        if np.any(missing_markers):
            print ("WARNING: No hits in database to marker(s) {0}.".format(",".join(markers[missing_markers])))
        
            
        # highlight groups and labels
        n = self._profile.numContigs
        gm = GroupManager(n, mask=mask)    
         
        if groups is not None and len(groups) > 0:
            if group_list is None or len(group_list) != n:
                raise ValueError("ERROR: Expected parameter `group_list` to be an array of length {0}.".format(n))
            
            if highlight_per_group:
                for group in groups:
                    if group=="":
                        continue
                    
                    gm.addGroup(np.flatnonzero(group_list==group), group)
            else:
                negroups = groups[groups!=""]
                gm.addGroup(np.flatnonzero(np.in1d(group_list, negroups)), "groups")
             
        if highlight_per_bid:
            
            for bid in bids:
                if bid == 0:
                    continue
                
                gm.addGroup(np.flatnonzero(self._profile.binIds==bid), "bid {0}".format(bid))
        else:
            nzbids = bids[bids!=0]
            gm.addGroup(np.flatnonzero(np.in1d(self._profile.binIds, nzbids)), "bids")
        
        if highlight_per_marker:
            for marker in markers:
                
                select_mappings = self._profile.mapping.markerNames==marker
                gm.addGroup(self._profile.mapping.rowIndices[select_mappings], "scg {0}".format(marker))
        else:
            select_mappings = np.in1d(self._profile.mapping.markerNames, markers)
            gm.addGroup(self._profile.mapping.rowIndices[select_mappings], "scgs")
        
        if highlight_per_taxstring:
            for taxstring in taxstrings:
                
                select_mappings = self._profile.mapping.classification.getPrefixed(taxstring)
                gm.addGroup(self._profile.mapping.rowIndices[select_mappings], "\"{0}\"".format(taxstring))
        else:
            select_mappings = np.zeros(n, dtype=bool)
            for taxstring in taxstrings:
                select_mappings = np.logical_or(select_mappings, self._profile.mapping.classification.getPrefixed(taxstring))
            gm.addGroup(self._profile.mapping.rowIndices[select_mappings], "taxons")
        
        return gm.getGroupIntersections() if highlight_intersections else gm.getGroups()
        

#------------------------------------------------------------------------------
# Helpers
        
def get_bin_tree(bids, d):
    bids = np.asarray(bids)
    d = np.asarray(d)
    
    # find bin contigs
    (first_binned_indices, last_binned_indices) = split_contiguous(bids, filter_groups=[0])
    
    split_obs = np.concat(([0], [np.arange(s, e)[np.argmax(d[s:e])] for (s, e) in zip(last_binned_indices[:-1], first_binned_indices[1:])]))
    
    Z = hierarchy.linkage_from_reachability(np.arange(len(split_obs)), d[split_obs])
    
    return (Z, split_obs)

class GroupAssignmentParser:
    def parse(self, filename, separator, cids):
        br = BinReader()
        try:
            with open(groupfile, "r") as f:
                try:
                    (con_names, con_groups) = br.parse(f, separator)
                    contig_groups = dict(zip(con_names, con_groups))
                except:
                    print "Error parsing group assignments"
                    raise
        except:
            print "Could not parse group assignment file:",groupfile, sys.exc_info()[0]
            raise
            
        return np.array([contig_groups.get(cid, "") for cid in cids])
        

def format_label(label):
    return "{0}...{1}".format(label[:8], label[-18:]) if len(label)>30 else label
        
    
def getColorMap(colorMapStr):
    if colorMapStr == 'HSV':
        S = 1.0
        V = 1.0
        return plt_colors.LinearSegmentedColormap.from_list('GC', [colorsys.hsv_to_rgb((1.0 + np.sin(np.pi * (val/1000.0) - np.pi/2))/2., S, V) for val in xrange(0, 1000)], N=1000)
    elif colorMapStr == 'Highlight1':
        return plt_colors.ListedColormap(["k", "r", "b", "g", "orange", "darkturquoise", "m"])
    elif colorMapStr == 'Highlight2':
        return plt_colors.ListedColormap(['cyan', 'dimgrey', 'orangered', 'indigo', 'goldenrod'])
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