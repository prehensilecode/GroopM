#!/usr/bin/env python
###############################################################################
#                                                                             #
#    corre.py                                                                 #
#                                                                             #
#    Rank correlation significance testing                                    #
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
import scipy.stats as stats

numpy.seterr(all='raise')

###############################################################################
###############################################################################
###############################################################################
###############################################################################

#------------------------------------------------------------------------------
#Rank correlation testing

def getInsidePNull(ranks):
    """For each data point return the probability in uncorrelated data of having at least as high inner point count"""
    ranks = numpy.asarray(ranks, dtype=float)
    (_num_dims, num_points) = ranks.shape
    counts = getInsideCount(ranks)

    # For a point with ranks r, the probability of another point having a
    # lower rank in all dimensions is `prod(r / r_max)` where r_max is the
    # maximum rank, equal to the total number of points.
    r_max = num_points - 1
    p_inside = numpy.prod(ranks / r_max, axis=0)

    # Statistical test counts
    return binomOneTailedTest(counts, r_max, p_inside)

def getOutsidePNull(ranks, inners):
    """For each data point return the probability in uncorrelated data of having at least as high inner point count"""
    ranks = numpy.asarray(ranks, dtype=float)
    (_num_dims, num_points) = ranks.shape
    outer_points = getBoundingPoints(ranks[:, inners], ranks)
    counts = getOutsideCount(ranks, ranks[:, inners], outer_points=outer_points)

    # For a pair of points r, s the probability of a point with all ranks
    # at least as low as r and higher s is
    # `prod(r / r_max) - prod(s / r_max)` where r_max is the highest rank,
    # equal to the total number of points
    r_max = num_points - 1
    p_outside = numpy.prod(outer_points / r_max, axis=0) - numpy.prod(ranks[:, inners] / r_max, axis=0)

    # Statistical test counts
    return binomOneTailedTest(counts, r_max, p_outside)

def binomOneTailedTest(counts, ns, ps):
    """Test counts against one-tailed binomial distribution"""
    return numpy.array([stats.binom.sf(c-1, n, p) for (c, n, p) in numpy.broadcast(counts, ns, ps)])


#------------------------------------------------------------------------------
#Point counting

def getInsideCount(data, points=None):
    """For each data point return the number of data points that have lower value in all dimensions"""
    data = numpy.asarray(data)
    if points is None:
        points = data
    points = numpy.asarray(points)
    (_num_dims, num_points) = points.shape
    counts = numpy.empty(num_points, dtype=int)
    for i in range(num_points):
        is_inside = numpy.all([d <= c for (c, d) in zip(points[:, i], data)], axis=0)
        counts[i] = numpy.count_nonzero(is_inside) - 1 # discount origin

    return counts

def getOutsideCount(data, inner_points, outer_points=None):
    """Return the number of data points that have values in a region between each data point and an internal point."""

    if outer_points is None:
        outer_points = getBoundingPoints(inner_points, data)

    outer_counts = getInsideCount(data, outer_points)
    inner_counts = getInsideCount(data, inner_points)

    return outer_counts - inner_counts

def getBoundingPoints(a_points, b_points):
    """Return a bounding region that contains both of a pair of points."""
    (num_a_dims, num_a_points) = a_points.shape
    (_num_b_dims, num_b_points) = b_points.shape

    swap = num_b_points > num_a_points
    (first, second, num_first, num_second) = (b_points, a_points, num_b_points, num_a_points) if num_b_points > num_a_points else (a_points, b_points, num_a_points, num_b_points)

    bounds = numpy.empty_like(first)
    for (i, j) in numpy.broadcast(range(num_first), range(num_second)):
        bounds[:, i] = numpy.maximum(first[:, i], second[:, j])

    return bounds


###############################################################################
###############################################################################
###############################################################################
###############################################################################
