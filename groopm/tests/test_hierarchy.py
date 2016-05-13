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
from groopm.hierarchy import (height,
                              maxcoeffs,
                              filter_descendents,
                              ancestors,
                              maxcoeff_roots,
                              cluster_remove,
                              linkage_from_reachability,
                             )

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def test_height():
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
                  
    assert_equal_arrays(height(Z),
                        [2, 2, 2, 1, 0, 1],
                        "`height` returns condensed matrix of lowest common ancestor indices")
                
                
def test_maxcoeffs():
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
                  
    """Assign coefficients:
        1-------+
        0---+   |-0
        0   |-0-+
        |-0-+
        0    
    """
    assert_equal_arrays(maxcoeffs(Z, [1, 0, 0, 0, 0, 0, 0]),
                        [1, 0, 0, 0, 0, 0, 1],
                        "`maxcoeffs` returns max coefficient of descendents in the case of a single "
                        "non-zero coeff")
                                
    """Assign coefficients:
        0-------+
        1---+   |-0
        1   |-2-+
        |-0-+
        0    
    """
    assert_equal_arrays(maxcoeffs(Z, [0, 1, 1, 0, 0, 2, 0]),
                        [0, 1, 1, 0, 1, 2, 2],
                        "`maxcoeffs` returns max coefficients in case of a non-zero leaf and larger "
                        "valued internal coeff")
                                              
    """Assign coefficients:
        0-------+
        0---+   |-0
        0   |-1-+
        |-2-+
        0    
    """
    assert_equal_arrays(maxcoeffs(Z, [0, 0, 0, 0, 2, 1, 0]),
                        [0, 0, 0, 0, 2, 2, 2],
                        "`maxcoeffs` returns max coefficients when a higher leaf is lower valued")
                        
    
def test_filter_descendents():
    """Z describes tree:
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
                  
    assert_equal_arrays(filter_descendents(Z, [5, 6, 7]),
                        [6, 7],
                        "`filter_descendents` removes nodes that are descendents")
                        
    assert_equal_arrays(filter_descendents(Z, [0, 1, 2, 3, 4, 5, 6]),
                        [0, 1, 6],
                        "`filter_descendents` returns internal and leaf nodes that are not descendents")
    
                        
def test_maxcoeff_roots():
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
                  
    """Assign coefficients:
        1-------+
        0---+   |-0
        0   |-0-+
        |-0-+
        0    
    """
    print maxcoeff_roots(Z, [1, 0, 0, 0, 0, 0, 0])
    assert_equal_arrays(maxcoeff_roots(Z, [1, 0, 0, 0, 0, 0, 0]),
                        [0, 5],
                        "`maxcoeff_roots` returns ancestors of a single "
                        "non-zero coeff")
                                
    """Assign coefficients:
        0-------+
        1---+   |-0
        1   |-2-+
        |-0-+
        0    
    """
    assert_equal_arrays(maxcoeff_roots(Z, [0, 1, 1, 0, 0, 2, 0]),
                        [0, 5],
                        "`maxcoeff_roots` returns ancestors of larger "
                        "valued internal coeff")
                                              
    """Assign coefficients:
        0-------+
        0---+   |-0
        0   |-1-+
        |-2-+
        0    
    """
    assert_equal_arrays(maxcoeff_roots(Z, [0, 0, 0, 0, 2, 1, 0]),
                        [0, 2, 4],
                        "`maxcoeff_roots` returns ancestors of larger "
                        "valued internal coeff when a higher leaf is lower valued")
    
    
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
             

def test_cluster_remove():
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
                                     
    assert_isomorphic(cluster_remove(Z, [5, 6]),
                      [0, 1, 2, 1],
                      "`cluster_remove` computes flat cluster indices after removing "
                      "internal indices")
                
    assert_isomorphic(cluster_remove(Z, [0, 6]),
                      [0, 1, 1, 1],
                      "`cluster_remove` computes flat cluster indices after removing "
                     "a leaf and internal indices")  


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
                        "`linkage_from_reachability` returns linkage corresponding to reachability ordering")
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
