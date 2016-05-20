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

from nose.tools import assert_true
import numpy as np
import numpy.random as np_random

# local imports
from tools import assert_equal_arrays, assert_isomorphic
import groopm.distance as distance
from groopm.hierarchy import (coeffs_linkage,
                              fcluster_coeffs,
                              flatten_nodes,
                              linkage_from_reachability,
                              ancestors,
                             )

###############################################################################
###############################################################################
###############################################################################
###############################################################################              
def test_coeffs_linkage():
    """
    Z describes tree:
        0-------+
        2---+   |-6
        1   |-5-+
        |-4-+
        3
    """
    Z = np.array([[1., 3., 1., 2.],
                  [2., 4., 1., 3.],
                  [0., 5., 2., 4.]])
                   
    """Assign Z leaf data:
        0:[1]------+
        2:[]---+   |-6
        1:[]   |-5-+
        |----4-+
        3:[]    
    """
    (coeffs, num_points) = coeffs_linkage(Z, {0:[1]}, max)
    assert_equal_arrays(coeffs,
                        [1, 0, 0, 0, 0, 0, 1],
                        "`coeffs_linkage` computes coefficients in the case of "
                        "a single non-zero coeff")
    assert_equal_arrays(num_points,
                        [1, 0, 0, 0, 0, 0, 1],
                        "`coeffs_linkage` counts number of data points in the "
                        "case of a single non-zero coeff")
                                
    """Assign leaf data:
        0:[]--------+
        2:[1]---+   |-6
        1:[1]   |-5-+
        |-----4-+
        3:[]    
    """
    (coeffs, num_points) = coeffs_linkage(Z, {1:[1], 2:[1]}, sum)
    assert_equal_arrays(coeffs,
                        [0, 1, 1, 0, 1, 2, 2],
                        "`coeffs_linkage` computes coefficients in case of a "
                        "non-zero leaf and larger valued internal coeff")
    assert_equal_arrays(num_points,
                        [0, 1, 1, 0, 1, 2, 2],
                        "`coeffs_linkage` computes number of data points in "
                        "case of a non-zero leaf and larger valued internal coeff")
                                              
    """Assign leaf data:
        0:[]--------+
        2:[1]---+   |-6
        1:[2,2] |-5-+
        |----4--+
        3:[1,0]    
    """
    (coeffs, num_points) = coeffs_linkage(Z, {1: [2, 2], 2: [1], 3:[1, 0]}, lambda x: max(x) - min(x))
    assert_equal_arrays(coeffs,
                        [0, 0, 0, 1, 2, 2, 2],
                        "`coeffs_linkage` returns max coefficients when a higher leaf is lower valued")
    assert_equal_arrays(num_points,
                        [0, 2, 1, 2, 4, 5, 5],
                        "`coeffs_linkage` returns number of data points where leaves have mulitple points")
    
                        
    
def test_fcluster_coeffs():
    """Z describes tree:
        0-------+
        2---+   |-6
        1   |-5-+
        |-4-+
        3
    """
    Z = np.array([[1., 3., 1., 2.],
                  [2., 4., 1., 3.],
                  [0., 5., 2., 4.]])
                  
    """Assign coefficients/num_points:
        0:1/1-----------+
        2:0/3---+       |-6:0/5
        1:1/1   |-5:0/4-+
        |-4:1/1-+
        3:0/0    
    """
    (T, c, M) = fcluster_coeffs(Z,
                                [1, 1, 0, 0, 1, 0, 0],
                                [1, 1, 3, 0, 1, 4, 5],
                                return_coeffs=True,
                                return_nodes=True)
    assert_equal_arrays(M,
                        [0, 2, 4],
                        "`fcluster_coeffs` returns cluster roots for skewed tree")
    assert_equal_arrays(c,
                        [1, 1, 0, 1],
                        "`fcluster_coeffs` returns leaf cluster coefficients for skewed tree")
    assert_isomorphic(T,
                      [1, 2, 3, 2],
                      "`fcluster_coeffs` computes flat cluster indices for skewed tree")
                                
    """Assign coefficients/num_points:
        0:0/2-----------+
        2:1/2---+       |-6:0/8
        1:1/2   |-5:2/6-+
        |-4:0/4-+
        3:0/2    
    """
    (T, M) = fcluster_coeffs(Z,
                                [0, 1, 1, 0, 0, 2, 0],
                                [2, 2, 2, 2, 4, 6, 8],
                                return_nodes=True)
    assert_equal_arrays(M,
                        [0, 5],
                        "`fcluster_coeffs` returns cluster roots for skewed "
                        "tree with large valued internal coefficient")
    assert_isomorphic(T,
                      [1, 2, 2, 2],
                      "`fcluster_coeffs` computes flat cluster indices for "
                      "skewed tree with large valued internal coefficient")  
                                              
    """Assign coefficients:
        0:0/3-----------+
        2:0/3---+       |-6:0/10
        1:0/2   |-5:1/7-+
        |-4:2/4-+
        3:0/2    
    """
    (T, M) = fcluster_coeffs(Z,
                             [0, 0, 0, 0, 2, 1, 0],
                             [3, 2, 3, 2, 4, 7, 10],
                             return_nodes=True)
    assert_equal_arrays(M,
                        [0, 2, 4],
                        "`fcluster_coeffs` returns cluster roots for skewed "
                        "tree with lower valued internal coefficient")
    assert_isomorphic(T,
                      [1, 2, 3, 2],
                      "`fcluster_coeffs` returns flat cluster indices for "
                      "skewed tree with lower valued internal coefficient")
                      
                      
    """Z describes tree:
        0
        |---7---+
        1       |
                |-8
        2---+   |
        3   |-6-+
        |-5-+
        4
    """
    Z = np.array([[3., 4., 1., 2.],
                  [2., 5., 1., 3.],
                  [0., 1., 3., 2.],
                  [6., 7., 4., 5.]])
                  
    """Assign coeffs/num_points:
        0:0/0
        |-----7:1/1-----+
        1:1/1           |
                        |-8:1/3
        2:0/0---+       |
        3:2/2   |-6:2/2-+
        |-5:2/2-+
        4:0/0
    """
    (T, c, M) = fcluster_coeffs(Z,
                                [0, 1, 0, 2, 0, 2, 2, 1, 1],
                                [0, 1, 0, 2, 0, 2, 2, 1, 3],
                                return_coeffs=True,
                                return_nodes=True)
    assert_equal_arrays(M,
                        [6, 7],
                        "`fcluster_coeffs` computes cluster roots for balanced tree")
    assert_equal_arrays(c,
                        [1, 1, 2, 2, 2],
                        "`fcluster_coeffs` computes leaf cluster coefficient scores for balanced tree")
    assert_isomorphic(T,
                      [1, 1, 2, 2, 2],
                      "`fcluster_coeffs` computes flat cluster indices for balanced tree")
                        
    """Assign coeffs/num_points:
        0:1/1
        |-----7:0/2-----+
        1:1/1           |
                        |-8:0/8
        2:2/2---+       |
        3:2/2   |-6:2/6-+
        |-5:0/4-+
        4:2/2
    """                    
    (T, M) = fcluster_coeffs(Z,
                                [1, 1, 2, 2, 2, 0, 2, 0, 0],
                                [1, 1, 2, 2, 2, 4, 6, 2, 8],
                                return_nodes=True)
    assert_equal_arrays(M,
                        [0, 1, 6],
                        "`filter_descendents` returns cluster roots for "
                        "balanced tree with singleton and non-singleton clusters")
    assert_isomorphic(T,
                      [1, 2, 3, 3, 3],
                      "`fcluster_coeffs` computes flat cluster indices for "
                      "balanced tree with singleton and non-singleton clusters")

    
def test_ancestors():
    """Z describes tree
        0
        |---7---+
        1       |
                |
        2---+   |-8
            |   |
        3   |-6-+
        |-5-+
        4
    """
    Z = np.array([[3., 4., 1., 2.],
                  [2., 5., 1., 3.],
                  [0., 1., 3., 2.],
                  [6., 7., 4., 5.]])
                  
    assert_equal_arrays(ancestors(Z, range(5)),
                        [5, 6, 7, 8],
                        "`ancestors` returns ancestors of all leaf clusters")
    
    assert_equal_arrays(ancestors(Z, [1]),
                        [7, 8],
                        "`ancestors` returns ancestors of a single leaf cluster")
    
    assert_equal_arrays(ancestors(Z, [5, 6, 8]),
                        [6, 8],
                        "`ancestors` returns union of ancestors for a path of nodes")
                        
    assert_equal_arrays(ancestors(Z, [5, 6, 8], inclusive=True),
                        [5, 6, 8],
                        "`ancestors` returns union of path nodes including nodes"
                        " themselves when `inclusive` flag is set")
             
                     
def test_flatten_nodes():
    """Z describes tree:
        0-------+
        2---+   |-6
        1   |-5-+
        |-4-+
        3
    """
    Z = np.array([[1., 3., 1., 2.],
                  [2., 4., 1., 3.],
                  [0., 5., 2., 4.]])
    
    assert_equal_arrays(flatten_nodes(Z),
                        [1, 1, 2],
                        "`flat_nodes` assigns nodes the indices of direct parent of equal height")
                      
    """Z describes tree:
        5
        |-9-+
        6   |-10-+
        2---+    |
                 |
        1        |-12
        |-8-+    |
        0   |    |
            |-11-+
        3   |
        |-7-+
        4
    """
    Z = np.array([[ 3.,  4., 1., 2.],
                  [ 0.,  1., 2., 2.],
                  [ 5.,  6., 3., 2.],
                  [ 2.,  9., 3., 3.],
                  [ 7.,  8., 3., 4.],
                  [10., 11., 3., 7.]])
    
    assert_equal_arrays(flatten_nodes(Z),
                        [0, 1, 5, 5, 5, 5],
                        "`flat_nodes` assigns nodes the indices of parents and grandparents of equal height")
                     
                     
def test_linkage_from_reachability():
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
    
    Corresponding tree is:
        0
        |-6-+
        1   |
            |-7-+
        3   |   |
        |-5-+   |-8
        4       |
        2-------+
    """
    
    Y = np.array([17.7, 70., 97.1, 50.8, 121.6, 79.4, 82.1, 120.9, 77.3, 14.4])
    Z = np.array([[3., 4., 14.4, 2.],
                  [0., 1., 17.7, 2.],
                  [5., 6., 50.8, 4.],
                  [2., 7., 70.0, 5.]])
    
    (o, d) = distance.reachability_order(Y)
    assert_equal_arrays(linkage_from_reachability(o, d),
                        Z,
                        "`linkage_from_reachability` returns linkage "
                        "corresponding to reachability ordering")
                        
    
    """
    Y encodes weighted distances for pairs:
    (0, 1) =  2
    (0, 2) =  9
    (0, 3) =  3
    (0, 4) =  5
    (0, 5) = 18
    (0, 6) =  7
    (1, 2) = 13
    (1, 3) =  4
    (1, 4) =  4
    (1, 5) =  4
    (1, 6) =  3
    (2, 3) =  9
    (2, 4) =  8
    (2, 5) =  3
    (2, 6) =  5
    (3, 4) =  1
    (3, 5) = 10
    (3, 6) =  9
    (4, 5) = 12
    (4, 6) = 11
    (5, 6) =  3
    
    Corresponding tree:
        5
        |-9-+
        6   |-10-+
        2---+    |
                 |
        1        |-12
        |-8-+    |
        0   |    |
            |-11-+
        3   |
        |-7-+
        4
    """
    Y = np.array([2., 9., 3., 5., 18., 7., 13., 4., 4., 4., 3., 9., 8., 3., 5., 1., 10., 9., 12., 11., 3.])
    Z = np.array([[ 3.,  4., 1., 2.],
                  [ 0.,  1., 2., 2.],
                  [ 5.,  6., 3., 2.],
                  [ 2.,  9., 3., 3.],
                  [ 7.,  8., 3., 4.],
                  [10., 11., 3., 7.]])
                  
    (o, d) = distance.reachability_order(Y)
    assert_equal_arrays(linkage_from_reachability(o, d)[:, 2],
                        Z[:, 2],
                        "Ignoring ties, `linkage_from_reachability` returns "
                        "linkage corresponding to reachability ordering for a "
                        "moderately complex hierarchy")
    
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
