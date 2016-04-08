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
from tools import assert_equal_arrays
from groopm.utils import (group_iterator,)

###############################################################################
###############################################################################
###############################################################################
###############################################################################               
               
def test_group_iterator():
    
    grouping = ["A", "B", "A", "A", "B"]
    pairs = [("A", [0, 2, 3]), ("B", [1, 4])]
    assert_true(all([t==p for (t, p) in zip(group_iterator(grouping),  pairs)]),
                "`group_iterator` returns grouping variable names and indices pairs")

                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
