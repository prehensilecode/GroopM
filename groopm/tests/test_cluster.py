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
                              maxcoeffs,
                              fcluster_merge,
                              flatten_nodes,
                              fcluster_coeffs,
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
    assert_equal_arrays(coeffs_linkage(Z, {0:[1]}, max),
                        [1, 0, 0, 0, 0, 0, 1],
                        "`coeffs_linkage` computes coefficients in the case of "
                        "a single non-zero coeff")
                                
    """Assign leaf data:
        0:[]--------+
        2:[1]---+   |-6
        1:[1]   |-5-+
        |-----4-+
        3:[]    
    """
    assert_equal_arrays(coeffs_linkage(Z, {1:[1], 2:[1]}, sum),
                        [0, 1, 1, 0, 1, 2, 2],
                        "`coeffs_linkage` computes coefficients in case of a "
                        "non-zero leaf and larger valued internal coeff")
                                              
    """Assign leaf data:
        0:[]--------+
        2:[1]---+   |-6
        1:[2,2] |-5-+
        |----4--+
        3:[1,0]    
    """
    assert_equal_arrays(coeffs_linkage(Z, {1: [2, 2], 2: [1], 3:[1, 0]}, lambda x: max(x) - min(x)),
                        [0, 0, 0, 1, 2, 2, 2],
                        "`coeffs_linkage` returns max coefficients when a higher leaf is lower valued")
    
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
