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
import scipy.cluster.hierarchy as sp_hierarchy

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def assert_equal_arrays(a, b, message):
    return assert_true(np.all(np.asarray(a) == np.asarray(b)), message)

    
def assert_almost_equal_arrays(a, b, message):
    return assert_true(np.all(np.around(a, 6) == np.around(b, 6)), message)
    
    
def assert_isomorphic(T1, T2, message):
    return assert_true(sp_hierarchy.is_isomorphic(T1, T2) and sp_hierarchy.is_isomorphic(T2, T1), message)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
