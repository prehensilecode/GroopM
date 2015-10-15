###############################################################################
#                                                                             #
#    This library is free software; you can redistribute it and/or            #
#    modify it under the terms of the GNU Lesser General Public               #
#    License as published by the Free Software Foundation; either             #
#    version 3.0 of the License, or (at your option) any later version.       #
#                                                                             #
#    This library is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU        #
#    Lesser General Public License for more details.                          #
#                                                                             #
#    You should have received a copy of the GNU Lesser General Public         #
#    License along with this library.                                         #
#                                                                             #
###############################################################################

__author__ = "Tim Lamberton"
__copyright__ = "Copyright 2015"
__credits__ = ["Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "tim.lamberton@gmail.com"

###############################################################################

# system imports
from tools import assert_equal_arrays, assert_almost_equal_arrays
import numpy
from groopm import cluster

###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
#Point counting

def test_get_count():
    # Four point configuration
    points = numpy.array([[1, 0, 3, 2], [3, 0, 2, 1]])
    assert_equal_arrays(cluster.get_inside_count(points),
                        [1, 0, 2, 1],
                        "`get_inside_count` counts number of points inside each point bounds")

    assert_equal_arrays(cluster.get_inside_count(points, points=numpy.array([[2], [3]])),
                        [2],
                        "`get_inside_count(...,points)` returns counts of points inside `points` bounds")

    assert_equal_arrays(cluster.get_bounding_points(points, points[:, :1]),
                        [[1, 1, 3, 2], [3, 3, 3, 3]],
                        "`get_bounding_points` returns a minimum bounding point for each of pairs of passed points")

    assert_equal_arrays(cluster.get_bounding_points(points[:, :1], points),
                        [[1, 1, 3, 2], [3, 3, 3, 3]],
                        "`get_bounding_points` returns same bounds when arguments are swapped")

    assert_equal_arrays(cluster.get_outside_count(points, inner_points=points[:, :1]),
                        [0, 0, 2, 1],
                        "`get_outside_count(...,points)` returns counts of points between each point and corresponding `points` bounds")

#------------------------------------------------------------------------------
#Rank correlation testing

def test_binom_one_tailed_test():

    outcomes = (3, 2)
    p0 = float(outcomes[0]) / sum(outcomes)
    pnot0 = 1 - p0
    p0_n_of_four = [pnot0*pnot0*pnot0*pnot0, # 0 of 4
                    4*p0*pnot0*pnot0*pnot0,  # 1 of 4
                    6*p0*p0*pnot0*pnot0,     # 2 of 4
                    4*p0*p0*p0*pnot0,        # 3 of 4
                    p0*p0*p0*p0]             # 4 of 4

    assert_almost_equal_arrays(cluster.binom_one_tailed_test([2, 3], 4, p0),
                               [sum(p0_n_of_four[2:]), sum(p0_n_of_four[3:])],
                               "`binom_one_tailed_test` returns binomial probabilities for an array of trial successes")

    # inisde
    points = numpy.array([[1, 0, 3, 2], [3, 0, 2, 1]])
    p_in = numpy.array([3, 0, 6, 2], dtype=float) / 9
    p_out = 1 - p_in
    p_in_n_of_three = numpy.array([p_out*p_out*p_out,  # 0 of 3
                                   3*p_in*p_out*p_out, # 1 of 3
                                   3*p_in*p_in*p_out,  # 2 of 3
                                   p_in*p_in*p_in])    # 3 of 3

    counts = numpy.array([1, 0, 2, 1])
    assert_almost_equal_arrays(cluster.binom_one_tailed_test(counts, 3, p_in),
                               [sum(p_in_n_of_three[c:, i]) for (i, c) in enumerate(counts)],
                               "`binom_one_tailed_test` returns binomial probabilities for an array of trial probabilties")

    assert_almost_equal_arrays(cluster.get_inside_p_null(points),
                               cluster.binom_one_tailed_test(counts, 3, p_in),
                               "`get_inside_p_null` returns binomial test probabilities of correlations of a set of ranks")

    # outside
    points = numpy.array([[1, 0, 3, 2], [3, 0, 2, 1]])
    bouding_points = numpy.array([[1, 1, 3, 2], [3, 3, 3, 3]])
    p_between_0 = numpy.array([0, 0, 6, 3], dtype=float) / 9
    counts = numpy.array([0, 0, 2, 1])
    assert_almost_equal_arrays(cluster.get_outside_p_null(points, [0]),
                               cluster.binom_one_tailed_test(counts, 3, p_between_0),
                               "`get_outside_p_null` returns binomial test probabilities of correlations of a rank subrange")



###############################################################################
###############################################################################
###############################################################################
###############################################################################
