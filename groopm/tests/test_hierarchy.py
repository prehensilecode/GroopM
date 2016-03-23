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

from nose.tools import assert_equals
from tools import assert_equal_arrays, assert_almost_equal_arrays
import numpy as np
import scipy.cluster.hierarchy as sp_hierarchy
from groopm.hierarchy import (height,
                              maxcoeffs,
                              filter_descendents)

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
                        [2., 2., 2., 1., 0., 1.],
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
                        [0, 0, 1],
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
                        [1, 2, 2],
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
                        [2, 2, 2],
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
                  
    assert_equal_arrays(filter_descendents(Z, [0, 1, 2]),
                        [1, 2],
                        "`filter_descendents` removes nodes that are descendents")
    
    
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
