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
from nose.tools import assert_true
from tools import assert_equal_arrays, assert_almost_equal_arrays
import numpy as np
import scipy.spatial.distance as sp_distance
import random
from groopm.distance import (mediod,
                             argrank,
                             pcoords,
                             ccoords)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def test_mediod():
    points = np.array([[0, 0, 1], 
                       [1, 1, 1], # mediod?
                       [1, 1, 2],
                       [1, 2, 1]])
    distances = sp_distance.pdist(points, metric="cityblock")
    assert_true(mediod(distances) == 1,
                "`mediod` returns index of mediod")

                
def test_argrank():
    assert_equal_arrays(argrank([5, 3, 4, 8]),
                        [2, 0, 1, 3],
                        "`argrank` returns integer rank of values in one-dimensional array")

    assert_equal_arrays(argrank([5, 3, 8, 8]),
                        [1, 0, 2.5, 2.5],
                        "`argrank` returns mean of tied ranks")

    arr2d = np.array([[1, 10, 5, 2], [1, 4, 6, 2], [5, 5, 3, 10]])
    ranks2d = np.array([[0, 3, 2, 1], [0, 2, 3, 1], [1.5, 1.5, 0, 3]])
    assert_equal_arrays(argrank(arr2d, axis=1),
                        ranks2d,
                        "`argrank(..,axis=1)` returns ranks along rows of 2D array")
    assert_equal_arrays(argrank(arr2d.T, axis=0),
                        ranks2d.T,
                        "`argrank(..,axis=0)` returns ranks along columns of 2D array")
                        
    assert_equal_arrays(argrank([5, 3, 4, 8], weights=[2, 2, 1, 3]),
                        argrank([5, 3, 4, 8, 5, 3, 8, 8])[:4],
                        "`argrank` returns weighted ranks when weights parameter is passed")
    
    assert_equal_arrays(argrank([[1, 10, 5, 2],
                                 [1,  4, 6, 2]
                                ], weights=[5, 1, 1, 6], axis=1),
                        argrank([[1, 10, 5, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                                 [1,  4, 6, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2]
                                ], axis=1)[:, :4],
                        "`argrank` broadcasts weights vector along rows of 2D array")
                        
    assert_equal_arrays(argrank([[1, 10, 5, 2],
                                 [1,  4, 6, 2]
                                ], weights=[2, 5], axis=0),
                        argrank([[1, 10, 5, 2],
                                 [1,  4, 6, 2],
                                 [1, 10, 5, 2],
                                 [1,  4, 6, 2],
                                 [1,  4, 6, 2],
                                 [1,  4, 6, 2],
                                 [1,  4, 6, 2]
                                ], axis=0)[:2],
                        "`argrank` broadcasts weights vector along columns of 2D array")
                        

def test_squareform_index():
    
    n = random.randint(3, 10)
    indices = np.arange(n * (n - 1) // 2)
    
    assert_equal_arrays(pcoords(np.arange(n), n),
                        indices,
                        "`pcoords` computes linear index of condensed distance matrix")
                        
    yi = ccoords(np.arange(n), np.arange(n), n)
    assert_equal_arrays(np.diag(yi),
                        -1,
                        "`ccoords` returns -1 for diagonal elements")
    assert_equal_arrays(yi[np.triu_indices(n, k=1)],
                        indices,
                        "`ccoords` computes linear index correctly when row < col")
    assert_equal_arrays(yi.T[np.triu_indices(n, k=1)],
                        indices,
                        "`ccoords` computes linear index correctly when row > col")
                        


###############################################################################
###############################################################################
###############################################################################
###############################################################################
