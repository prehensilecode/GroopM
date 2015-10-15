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
from groopm import hybridMeasure

###############################################################################
###############################################################################
###############################################################################
###############################################################################

###############################################################################
#Utility functions
###############################################################################

#------------------------------------------------------------------------------
#Ranking

def test_argrank():
    assert_equal_arrays(hybridMeasure.argrank([5, 3, 4, 8]),
                       [2, 0, 1, 3],
                       "`argrank` returns integer rank of values in one-dimensional array")

    assert_equal_arrays(hybridMeasure.argrank([5, 3, 8, 8]),
                        [1, 0, 2.5, 2.5],
                        "`argrank` returns mean of tied ranks")

    arr2d = numpy.array([[1, 10, 5, 2], [1, 4, 6, 2], [5, 5, 3, 10]])
    ranks2d = numpy.array([[0, 3, 2, 1], [0, 2, 3, 1], [1.5, 1.5, 0, 3]])
    assert_equal_arrays(hybridMeasure.argrank(arr2d, axis=1),
                        ranks2d,
                        "`argrank(..,axis=1)` returns ranks along rows of 2-d array")
    assert_equal_arrays(hybridMeasure.argrank(arr2d.T, axis=0),
                        ranks2d.T,
                        "`argrank(..,axis=0)` returns ranks along columns of 2-d array")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
