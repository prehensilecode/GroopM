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
from tools import (equal_arrays, almost_equal_arrays)
from groopm.stream_ext import (merge,)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
   
def test_merge():
    left = np_random.rand(np_random.random_integers(5, 10))
    left.sort()
    right = np_random.rand(np_random.random_integers(5, 10))
    right.sort()
    values = np.concatenate((left, right))
    indices = values.argsort()
    sorted_values = values[indices]
    
    n = np_random.random_integers(5, values.size)
    merged = np.zeros(n, dtype=values.dtype)
    merged_indices = np.zeros(n, dtype=indices.dtype)
    merge(left.size,
          left,
          np.arange(left.size),
          right.size,
          right,
          np.arange(left.size, values.size),
          n,
          merged,
          merged_indices,
          )
    assert_true(equal_arrays(sorted_values[:n], merged),
                "sorts values in output array")
    assert_true(equal_arrays(indices[:n], merged_indices),
                "writes sorting indices into output indices array")
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
