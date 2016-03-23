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
import numpy as np
import itertools
from groopm.corre import (getInsideCount,
                          getBoundingPoints,
                          getOutsideCount,
                          binomOneTailedTest,
                          getInsidePNull,
                          getOutsidePNull,
                          ContainmentFinder)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def test_get_count():
    # Four point configuration
    points = np.array([[1, 0, 3, 2], [3, 0, 2, 1]])
    assert_equal_arrays(getInsideCount(points),
                        [1, 0, 2, 1],
                        "count number of points inside each point bounds")

    assert_equal_arrays(getInsideCount(points, points=np.array([[2], [3]])),
                        [2],
                        "return count of points inside `points` bounds")

    assert_equal_arrays(getBoundingPoints(points, points[:, :1]),
                        [[1, 1, 3, 2], [3, 3, 3, 3]],
                        "return a minimum bounding point for each of pairs of passed points")

    assert_equal_arrays(getBoundingPoints(points[:, :1], points),
                        [[1, 1, 3, 2], [3, 3, 3, 3]],
                        "return same bounds when arguments are swapped")

    assert_equal_arrays(getOutsideCount(points, inner_points=points[:, :1]),
                        [0, 0, 2, 1],
                        "return counts of points between each point and corresponding `points` bounds")


def test_binom_one_tailed_test():

    outcomes = (3, 2)
    p0 = float(outcomes[0]) / sum(outcomes)
    pnot0 = 1 - p0
    p0_n_of_four = [pnot0*pnot0*pnot0*pnot0, # 0 of 4
                    4*p0*pnot0*pnot0*pnot0,  # 1 of 4
                    6*p0*p0*pnot0*pnot0,     # 2 of 4
                    4*p0*p0*p0*pnot0,        # 3 of 4
                    p0*p0*p0*p0]             # 4 of 4

    assert_almost_equal_arrays(binomOneTailedTest([2, 3], 4, p0),
                               [sum(p0_n_of_four[2:]), sum(p0_n_of_four[3:])],
                               "return binomial probabilities for an array of trial successes")

    # inisde
    points = np.array([[1, 0, 3, 2], [3, 0, 2, 1]])
    p_in = np.array([3, 0, 6, 2], dtype=float) / 9
    p_out = 1 - p_in
    p_in_n_of_three = np.array([p_out*p_out*p_out,  # 0 of 3
                                   3*p_in*p_out*p_out, # 1 of 3
                                   3*p_in*p_in*p_out,  # 2 of 3
                                   p_in*p_in*p_in])    # 3 of 3

    counts = np.array([1, 0, 2, 1])
    assert_almost_equal_arrays(binomOneTailedTest(counts, 3, p_in),
                               [sum(p_in_n_of_three[c:, i]) for (i, c) in enumerate(counts)],
                               "return binomial probabilities for an array of trial probabilties")

    assert_almost_equal_arrays(getInsidePNull(points),
                               binomOneTailedTest(counts, 3, p_in),
                               "return binomial test probabilities of correlations of a set of ranks")

    # outside
    points = np.array([[1, 0, 3, 2], [3, 0, 2, 1]])
    bouding_points = np.array([[1, 1, 3, 2], [3, 3, 3, 3]])
    p_between_0 = np.array([0, 0, 6, 3], dtype=float) / 9
    counts = np.array([0, 0, 2, 1])
    assert_almost_equal_arrays(getOutsidePNull(points, [0]),
                               binomOneTailedTest(counts, 3, p_between_0),
                               "return binomial test probabilities of correlations of a rank subrange")

                               
def test_containment_finder():

    # 2D
    data = np.array([[3, 2, 1, 4],
                        [4, 1, 2, 3]])
    edges = [[None,0],
             [None,3],
             [0,1],
             [0,2],
             [3,1],
             [3,2]] # computed by hand
    links = ContainmentFinder(data).run()
    pairs = sorted(list(itertools.chain(*[[[i,j] for j in links[i]] for i in links])))
    assert_equal_arrays(pairs,
                        edges,
                        "computes containment pairs in 2 dimensions")

    # 3D
    data = np.array([[3,1,2,0],
                        [1,1,2,0],
                        [1,1,2,0]])
    edges = [[None,0],
             [None,2],
             [0,3],
             [1,3],
             [2,1]] # computed by hand
    links = ContainmentFinder(data).run()
    pairs = sorted(list(itertools.chain(*[[[i,j] for j in links[i]] for i in links])))
    assert_equal_arrays(pairs,
                        edges,
                        "computes containment pairs in 3 dimensions")


###############################################################################
###############################################################################
###############################################################################
###############################################################################
