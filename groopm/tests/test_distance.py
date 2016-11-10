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
import numpy as np
import scipy.spatial.distance as sp_distance
import random

# local imports
from tools import equal_arrays, almost_equal_arrays
from groopm.distance import (mediod,
                             core_distance,
                             reachability_order,
                             _fractional_rank,
                             _ifractional_rank,
                             argrank,
                             pairs,
                             condensed_index,
                             squareform_coords)

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
    v1 = [5, 3, 4, 8]
    o1 = [2, 0, 1, 3]
    assert_true(equal_arrays(_fractional_rank(v1),
                             o1),
                "returns integer rank of values in one-dimensional array")
    x1 = np.array(v1)
    _ifractional_rank(x1)
    assert_true(equal_arrays(x1, o1),
                "loads array ranks into input array")

    v2 = [5, 3, 8, 8]
    o2 = [1, 0, 2.5, 2.5]
    assert_true(equal_arrays(_fractional_rank(v2),
                             o2),
                "returns mean of tied ranks")
    x2 = np.array(v2, dtype=np.double)
    _ifractional_rank(x2)
    assert_true(equal_arrays(x2, o2),
                "loads array ranks with tied means into input array")

    v3 = np.array([[1, 10, 5, 2], [1, 4, 6, 2], [5, 5, 3, 10]])
    o3 = np.array([[0, 3, 2, 1], [0, 2, 3, 1], [1.5, 1.5, 0, 3]])
    assert_true(equal_arrays(argrank(v3, axis=1),
                             o3),
                "with `axis=1` passed returns ranks along rows of 2D array")
    assert_true(equal_arrays(argrank(v3.T, axis=0),
                             o3.T),
                "with `axis=0` passed returns ranks along columns of 2D array")
    
    v4 = np.asarray([5, 3, 4, 8])
    w4 = np.asarray([2, 2, 1, 3])
    v4_dup = np.asarray([5, 3, 4, 8, 5, 3, 8, 8])
    assert_true(equal_arrays(_fractional_rank(v4, weight_fun=lambda i: w4[i]),
                             _fractional_rank(v4_dup)[:4]),
                "returns weighted ranks when weights parameter is passed")
    x4 = np.array(v4, dtype=np.double)
    _ifractional_rank(x4, weight_fun=lambda i: w4[i])
    assert_true(equal_arrays(x4,
                             _fractional_rank(v4_dup)[:4]),
                "returns idential ranks to non-mutating methods")
    
    v5 = np.array([[1, 10, 5, 2],
                   [1,  4, 6, 2]])
    w5 = np.asarray([5, 1, 1, 6])
    v5_dup = np.array([[1, 10, 5, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                       [1,  4, 6, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2]])
    assert_true(equal_arrays(argrank(v5, weight_fun=lambda i: w5[i], axis=1),
                             argrank(v5_dup, axis=1)[:, :4]),
                "broadcasts weights vector along rows of 2D array when ranking columns")
    
    v6 = np.array([[1, 10, 5, 2],
                   [1,  4, 6, 2]])
    w6 = np.asarray([2, 5])
    v6_dup = np.array([[1, 10, 5, 2],
                       [1,  4, 6, 2],
                       [1, 10, 5, 2],
                       [1,  4, 6, 2],
                       [1,  4, 6, 2],
                       [1,  4, 6, 2],
                       [1,  4, 6, 2]])
    assert_true(equal_arrays(argrank(v6, weight_fun=lambda i: w6[i], axis=0),
                             argrank(v6_dup, axis=0)[:2]),
                "broadcasts weights vector along columns of 2D array when ranking rows")

        
def test_core_distance():
    """
    Y encodes distances for pairs:
    (0, 1) =  2.2
    (0, 2) =  7.2
    (0, 3) = 10.4
    (0, 4) =  6.7
    (1, 2) = 12.8
    (1, 3) =  8.6
    (1, 4) =  8.9
    (2, 3) = 12.7
    (2, 4) =  8.6
    (3, 4) =  2.2
    
    closest to furtherest distances
    0 = 2.2, 6.7,  7.2, 10.4
    1 = 2.2, 8.6,  8.9, 12.8
    2 = 7.2, 8.6, 12.7, 12.8
    3 = 2.2, 8.6, 10.4, 12.7
    4 = 2.2, 6.7,  8.6,  8.9
    """
    Y = np.array([2.2, 7.2, 10.4, 6.7, 12.8, 8.6, 8.9, 12.7, 8.6, 2.2])
    n = sp_distance.num_obs_y(Y)
    
    assert_true(equal_arrays(core_distance(Y, minPts=1),
                             [2.2, 2.2, 7.2, 2.2, 2.2]),
                "returns nearest neighbour distance with minPts=1")
    assert_true(equal_arrays(core_distance(Y, weight_fun=lambda _i, _j: 1, minWt=[1]*n),
                             [2.2, 2.2, 7.2, 2.2, 2.2]),
                "returns nearest neighbour distance with unit weights and minWts")
                        
    assert_true(equal_arrays(core_distance(Y, minPts=2),
                             [6.7, 8.6, 8.6, 8.6, 6.7]),
                "returns 2-nearest neighbour distance with minPts=2")
         
    assert_true(equal_arrays(core_distance(Y, minPts=4),
                             [10.4, 12.8, 12.8, 12.7, 8.9]),
                "returns distance to 4-nearest neighbour with minPts=4")    
    assert_true(equal_arrays(core_distance(Y, weight_fun=lambda _i, _j: 1, minWt=[4]*n),
                             [10.4, 12.8, 12.8, 12.7, 8.9]),
                "returns distance to 4-nearest neighbour distance with unit "
                "weights and minWts=4")   
                        
    
    """
    Y encodes weighted distances for pairs:
    (0, 1) =  17.7
    (0, 2) =  70.0
    (0, 3) =  97.1
    (0, 4) =  50.8
    (1, 2) = 121.6
    (1, 3) =  79.4
    (1, 4) =  82.1
    (2, 3) = 120.9
    (2, 4) =  77.3
    (3, 4) =  14.4
    
    w encodes pairwise weights:
    (0, 1) =  4
    (0, 2) =  8
    (0, 3) =  6
    (0, 4) = 10
    (1, 2) =  6
    (1, 3) =  6
    (1, 4) = 10
    (2, 3) = 12
    (2, 4) = 20
    (3, 4) = 15
    
    cumulative weights
    0 =  4, 14, 22, 28
    1 =  4, 10, 20, 26
    2 =  8, 28, 36, 42
    3 = 15, 21, 27, 39
    4 = 15, 25, 45, 55
    
    closest to furtherest distances
    0 = 17.7, 50.8,  70.0,  97.1
    1 = 17.7, 79.4,  82.1, 121.6
    2 = 70.0, 77.3, 120.9, 121.6
    3 = 14.4, 79.4,  97.1, 120.9
    4 = 14.4, 50.8,  77.3,  82.1  
    """ 
    Y = np.array([17.7, 70., 97.1, 50.8, 121.6, 79.4, 82.1, 120.9, 77.3, 14.4])
    w = np.array([   4,   8,    6,   10,     6,    6,   10,    12,   20,   15])
    n = sp_distance.num_obs_y(Y)
    
    assert_true(equal_arrays(core_distance(Y, weight_fun=lambda i, j: w[condensed_index(n, i, j)], minWt=[20]*n),
                             [70.0, 82.1, 77.3, 79.4, 50.8]) and
                equal_arrays(core_distance(Y, weight_fun=lambda i, j: w[condensed_index(n, i, j)], minWt=[30]*n),
                             [97.1, 121.6, 120.9, 120.9, 77.3]),
                "computes weighted core distances at various limits")
                
                
def test_reachability_order():
    #
    
    """
    Y encodes weighted distances for pairs:
    (0, 1) =  17.7
    (0, 2) =  70.0
    (0, 3) =  97.1
    (0, 4) =  50.8
    (1, 2) = 121.6
    (1, 3) =  79.4
    (1, 4) =  82.1
    (2, 3) = 120.9
    (2, 4) =  77.3
    (3, 4) =  14.4
    
    closest to furtherest distances
    0 = 17.7, 50.8,  70.0,  97.1
    1 = 17.7, 79.4,  82.1, 121.6
    2 = 70.0, 77.3, 120.9, 121.6
    3 = 14.4, 79.4,  97.1, 120.9
    4 = 14.4, 50.8,  77.3,  82.1 
    """
    Y = np.array([17.7, 70., 97.1, 50.8, 121.6, 79.4, 82.1, 120.9, 77.3, 14.4])
    (o, d) = reachability_order(Y)
    assert_true(equal_arrays(o, [0, 1, 4, 3, 2]),
                "returns reachability traversal order")
    assert_true(equal_arrays(d, [0, 17.7, 50.8, 14.4, 70.0]),
                "returns reachability distances when traversing points")
    
    """
    closest to furtherest pairs / distances with core_dists
    0 = 70.0 (1), 70.0 (2), 70.0 (4),  97.1 (3)
    1 = 82.1 (0), 82.1 (3), 82.1 (4),  121.6 (2)
    2 = 77.3 (0), 77.3 (4), 120.9 (3), 121.6 (1)
    3 = 79.4 (1), 79.4 (4), 97.1 (0),  120.9 (2)
    4 = 50.8 (0), 50.8 (3), 77.3 (2),  82.1 (1)
    """
    core_dists = np.array([70.0, 82.1, 77.3, 79.4, 50.8])
    (o, d) = reachability_order(Y, core_dists)
    assert_true(equal_arrays(o, [0, 1, 2, 4, 3]),
                "returns reachability traversal order with core distances")
    assert_true(equal_arrays(d, [70.0, 70.0, 70.0, 70.0, 50.8]),
                "returns reachability distances computed using core distances")
                        
    

def test_condensed_index():
    n = random.randint(3, 10)
    m = n * (n - 1) // 2
    (ri, ci) = pairs(n)
    assert_true(equal_arrays(ri, [i for i in range(n-1) for _ in range(i+1, n)]) and
                equal_arrays(ci, [j for i in range(n-1) for j in range(i+1, n)]),
                "generate pairs of square coordinates")
                
    condensed_indices = condensed_index(n, ri, ci)
    assert_true(equal_arrays(condensed_indices,
                             np.arange(m)),
                "compute linear index of condensed distance matrix")
                        
    assert_true(equal_arrays(condensed_indices,
                             condensed_index(n, ci, ri)),
                "computes linear index correctly when row < col")
                        

def test_squareform_coords():
    n = random.randint(3, 10)
    m = n * (n - 1) // 2
    (ri, ci) = pairs(n)
    
    (squareform_i, squareform_j) = squareform_coords(n, range(m))
    print squareform_i, ri
    print squareform_j, ci
    assert_true(equal_arrays(squareform_i, ri) and equal_arrays(squareform_j, ci),
                "compute row and column indices for elements of condensed matrix")
    assert_true(np.all(squareform_i <= squareform_j),
                "returns upper triangular coordinates")
                        

###############################################################################
###############################################################################
###############################################################################
###############################################################################
