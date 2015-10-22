#!/usr/bin/env python
###############################################################################
#                                                                             #
#    hybridMeasure.py                                                         #
#                                                                             #
#    Compute coverage / kmer hybrid distance measure                          #
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

import numpy
import scipy.spatial.distance as distance

import sys
import matplotlib.pyplot as plt

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class HybridMeasure:
    """Computes the following metric pair:
        cov  = euclidean distance in log coverage space,
        kmer = euclidean distance in kmer sig space. """
    def __init__(self, PM):
        self._PM = PM

    def getDistances(self, a_members, b_members=None):
        """Get distances between two sets of points for the metrics"""

        if b_members is None:
            cov = distance.squareform(distance.pdist(numpy.log10(self._PM.covProfiles[a_members]+1), metric="euclidean"))
            kmer = distance.squareform(distance.pdist(self._PM.kmerSigs[a_members], metric="euclidean"))
        else:
            cov = distance.cdist(numpy.log10(self._PM.covProfiles[a_members]+1), numpy.log10(self._PM.covProfiles[b_members]+1), metric="euclidean")
            kmer = distance.cdist(self._PM.kmerSigs[a_members], self._PM.kmerSigs[b_members], metric="euclidean")

        return numpy.array([cov, kmer])

    def getMediod(self, members):
        """Get member index that minimises the sum rank euclidean distance to other members.

        The sum rank euclidean distance is the sum of the euclidean distances of distance ranks for the metrics"""

        # for each member, sum of distances to other members
        scores = [numpy.sum(d, axis=1) for d in self.getDistances(members)]
        ranks = argrank(scores, axis=1)

        # combine using euclidean distance between ranks
        combined = numpy.linalg.norm(ranks, axis=0)
        index = numpy.argmin(combined)

        return index

    def associateWith(self, a_members, b_members):
        """Associate b points with closest a point"""

        distances = self.getDistances(self, a_members, b_members)
        (_dims, a_num, b_num) = distances.shape

        # rank distances to a points
        ranks = argrank(distances, axis=1)

        # combine using euclidean distance between ranks
        combined = numpy.linalg.norm(ranks, axis=0)
        b_to_a = numpy.argmin(combined, axis=0)

        return b_to_a

    def getDimNames(self):
        """Labels for distances returned by get_distances"""
        return ("log coverage euclidean", "kmer euclidean")


###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
#Ranking

def rankWithTies(array):
    """Return sorted of array indices with tied values averaged"""
    ranks = numpy.asarray(numpy.argsort(numpy.argsort(array)), dtype=float)
    for val in set(array):
        g = array == val
        ranks[g] = numpy.mean(ranks[g])
    return ranks


def argrank(array, axis=0):
    """Return the positions of elements of a when sorted along the specified axis"""
    return numpy.apply_along_axis(rankWithTies, axis, array)

#------------------------------------------------------------------------------
#Plotting

class HybridMeasurePlotter:
    """Plot contigs in hybrid measure space"""
    COLOURS = 'rbgcmyk'

    def __init__(self, PM):
        self._PM = PM
        self._HM = HybridMeasure(self._PM)

    def getOrigin(self, members, mode="mediod"):
        if mode=="mediod":
            index = self._HM.getMediod(members)
        elif mode=="max_coverage":
            index = numpy.argmax(self._PM.normCoverages[members])
        elif mode=="max_length":
            index = numpy.argmax(self._PM.contigLengths[members])
        else:
            raise ValueError("Invalid mode: %s" % mode)
        return members[index]

    def plot(self, origin, plotRanks=False, keep=None,
             highlight=None, divide=None, plotContigLengths=False, fileName=""):
        """Plot contigs in measure space"""
        fig = plt.figure()

        ax = fig.add_subplot(111)
        self.plotOnAx(ax, origin, plotRanks=plotRanks, keep=keep,
                      highlight=highlight, plotContigLengths=plotContigLengths)

        if divide is not None:
            for (clr, coords) in zip(COLOURS, divide):
                fmt = '-'+clr
                for (x_point, y_point) in zip(*coords):
                    ax.plot([x_point, x_point], [0, y_point], fmt)
                    ax.plot([0, x_point], [y_point, y_point], fmt)

        if(fileName != ""):
            try:
                fig.set_size_inches(6,6)
                plt.savefig(fileName,dpi=300)
            except:
                print "Error saving image:", fileName, sys.exc_info()[0]
                raise
        else:
            print "Plotting contig features"
            try:
                plt.show()
            except:
                print "Error showing image", sys.exc_info()[0]
                raise

        plt.close(fig)
        del fig

    def plotSurface(self, origin, f, label, plotRanks=False, keep=None,
            highlight=None, plotContigLengths=False, elev=None, azim=None,
            fileName="")
        """Plot a surface computed from coordinates in measure space"""
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        self.plotOnAx(ax, origin, z=f, z_label=label, plotRanks=plotRanks,
            keep=keep, highlight=highlight, plotContigLengths=plotContigLengths,
            elev=elev, azim=azim)

        if(fileName != ""):
            try:
                fig.set_size_inches(6,6)
                plt.savefig(fileName,dpi=300)
            except:
                print "Error saving image:", fileName, sys.exc_info()[0]
                raise
        else:
            print "Plotting contig features"
            try:
                plt.show()
            except:
                print "Error showing image", sys.exc_info()[0]
                raise

        plt.close(fig)
        del fig

    def plotOnAx(self, ax, origin, z=None, z_label=None, keep=None, extents=None,
            highlight=None, plotRanks=False, plotContigLengths=False, elev=None,
            azim=None):

        # display values
        distances = self._HM.getDistances([origin])[:, 0, :]
        data = argrank(distances, axis=1) if ranks else distances
        (x, y) = (data[0], data[1])
        disp_vals = (x, y, z(x, y)) if z is not None else (x, y)

        # display labels
        labels = self._HM.getDimNames()
        disp_cols = self._PM.contigGCs

        if highlight is not None:
            edgecolors=numpy.full_like(disp_cols, 'k', dtype=str)
            for (clr, hl) in zip(COLOURS, highlight):
                edgecolors[hl] = clr
            if keep is not None:
                edgecolors = edgecolors[keep]
        else:
            edgecolors = 'k'

        if plotContigLengths:
            disp_lens = numpy.sqrt(self._PM.contigLengths)
            if keep is not None:
                disp_lens = disp_lens[keep]
        else:
            disp_lens=30

        if keep is not None:
            disp_vals = [v[keep] for v in disp_vals]
            disp_cols = disp_cols[keep]

        sc = ax.scatter(*disp_vals,
                        c=disp_cols, s=disp_lens,
                        cmap=self._PM.colorMapGC,
                        vmin=0.0, vmax=1.0,
                        marker='.')
        sc.set_edgecolors(edgecolors)
        sc.set_edgecolors = sc.set_facecolors = lambda *args:None # disable depth transparency effect

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if z_label is not None and z is not None:
            ax.set_zlabel(z_label)

        if extents is not None:
            ax.set_xlim([extents[0], extents[1]])
            ax.set_ylim([extents[2], extents[3]])
            if z is not None:
                ax.set_zlim([extents[4], extents[5]])

        if z is not None:
            ax.view_init(elev=elev, azim=azim)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
