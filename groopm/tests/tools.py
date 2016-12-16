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

def equal_arrays(a, b):
    return np.all(np.asarray(a) == np.asarray(b))

    
def almost_equal_arrays(a, b):
    return np.all(np.around(a, 6) == np.around(b, 6))
    
    
def is_isomorphic(T1, T2):
    return sp_hierarchy.is_isomorphic(T1, T2) and sp_hierarchy.is_isomorphic(T2, T1)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
