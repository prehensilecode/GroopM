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
import scipy.spatial.distance as sp_distance
import numpy.random as np_random
import os

# local imports
from tools import equal_arrays
from groopm.stream import pdist_chunk

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
    
class TestStream:
    
    @classmethod
    def setup_class(self):
        
        self.workingDir = os.path.join(os.path.split(__file__)[0], "test_stream")
        os.mkdir(self.workingDir)
        self.storageFile = os.path.join(self.workingDir, "test_stream.store")
        
    @classmethod
    def teardown_class(self):
        try:
            os.remove(self.storageFile)
        except OSError:
            pass
        os.rmdir(self.workingDir)
        
    def testPdistChunk(self):
        #
        features = np_random.rand(100, 50)
        dists = sp_distance.pdist(features, metric="euclidean")
        pdist_chunk(features, self.storageFile, chunk_size=30, metric="euclidean")
        assert_true(equal_arrays(np.memmap(self.storageFile, dtype=np.double),
                                 dists),
                    "computes same distances as unchunked function")
    
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
