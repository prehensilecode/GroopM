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
import shutil
import tempfile
import operator

# local imports
from tools import (equal_arrays, almost_equal_arrays)
from groopm.distance import argrank
from groopm.stream import (pdist_chunk,
                           argsort_chunk_mergesort,
                           argrank_chunk,
                           iapply_func_chunk
                          )

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
    
class TestStream:
    
    @classmethod
    def setup_class(self):
        
        self.workingDir = tempfile.mkdtemp(prefix="test_stream", dir=os.path.join(os.path.split(__file__)[0]))
        self.pdistFile = os.path.join(self.workingDir, "test_stream.pdist.store")
        self.argsortInfile = os.path.join(self.workingDir, "test_stream.argsort.in.store")
        self.argsortOutfile = os.path.join(self.workingDir, "test_stream.argsort.out.store")
        self.argrankDistsFile = os.path.join(self.workingDir, "test_stream.argrank.dists.store")
        self.argrankIndicesFile = os.path.join(self.workingDir, "test_stream.argrank.indices.store")
        self.iapplyFuncInfile = os.path.join(self.workingDir, "test_stream.iapply_func.in.store")
        self.iapplyFuncOutfile = os.path.join(self.workingDir, "test_stream.iapply_func.out.store")
    
    def _remove_one(self, filename):
        try:
            os.remove(filename)
        except OSError:
            pass
    
    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.workingDir)
    
    def testPdistChunk(self):
        #
        filename = self.pdistFile
        
        def _test_one_small():
            f1 = np_random.rand(20, 50)
            d1 = sp_distance.pdist(f1, metric="euclidean")
            pdist_chunk(f1, filename, chunk_size=30, metric="euclidean")
            assert_true(equal_arrays(np.fromfile(filename, dtype=np.double),
                                     d1),
                        "computes same distances as unchunked function")
            os.remove(filename)
        
        for _ in range(50):
            _test_one_small()
        
        # high mem
        def _test_one_big():
            f2 = np_random.rand(2**10, 50)
            d2 = sp_distance.pdist(f2, metric="euclidean")
            pdist_chunk(f2, filename, chunk_size=int(1e5), metric="euclidean")
            assert_true(equal_arrays(np.fromfile(filename, dtype=np.double),
                                     d2),
                        "computes same distances as unchunked function for a large-ish dataset")
                        
            os.remove(filename)
        
        for _ in range(5):
            _test_one_big()
    
    def testArgsortChunkMergesort(self):
        #
        infile = self.argsortInfile
        outfile = self.argsortOutfile
        
        def _test_one_small():
            d1 = np_random.rand(190).astype(np.double)
            d1.tofile(infile)
            i1 = d1.argsort()
            argsort_chunk_mergesort(infile, outfile, chunk_size=30)
            assert_true(equal_arrays(np.fromfile(outfile, dtype=np.int), i1),
                        "sorted indices are stored in output file")
            assert_true(equal_arrays(np.fromfile(infile, dtype=np.double), d1[i1]),
                        "input file values are in sorted order")
            os.remove(infile)
            os.remove(outfile)
            
        for _ in range(50):
            _test_one_small()
        
        # high mem
        def _test_one_big():
            d2 = np_random.rand(2**9*(2**10-1)).astype(np.double)
            d2.tofile(infile)
            argsort_chunk_mergesort(infile, outfile, chunk_size=int(1e5))
            arr = np.fromfile(infile, dtype=np.double)
            assert_true(np.all(arr[1:]>=arr[:-1]), "large-ish input file values are sorted")
            inds = np.fromfile(outfile, dtype=np.int)
            assert_true(np.all(d2[inds[1:]]>=d2[inds[:-1]]), "output file contains sorting indices for large-ish input")
            os.remove(infile)
            os.remove(outfile)
        
        for _ in range(5):
            _test_one_big()
        
    def testArgrankChunk(self):
        #
        dist_file = self.argrankDistsFile
        indices_file = self.argrankIndicesFile
        
        def _test_one_small():
            d1 = np_random.rand(190).astype(np.double)
            d1.tofile(dist_file)
            x1 = argrank_chunk(dist_file, indices_file, chunk_size=40)
            assert_true(equal_arrays(x1, argrank(d1, axis=None)),
                        "returns equal ranks to non-chunked function")
            
            d1.tofile(dist_file)
            w2 = np_random.rand(190).astype(np.double)
            x2 = argrank_chunk(dist_file, indices_file, weight_fun=lambda i: w2[i], chunk_size=40)
            assert_true(almost_equal_arrays(x2, argrank(d1, weight_fun=lambda i: w2[i], axis=None)),
                        "correctly weights ranks when passed a weight function")
            os.remove(dist_file)
            os.remove(indices_file)
            
        for _ in range(50):
            _test_one_small()
        
        # high mem
        def _test_one_big():
            d2 = np.arange(2**9*(2**10-1))
            np_random.shuffle(d2)
            d2.tofile(dist_file)
            #i2.tofile(indices_file)
            x3 = argrank_chunk(dist_file, indices_file, chunk_size=int(1e5))
            assert_true(equal_arrays(x3, d2+1), "computes ranks of a large-ish permutation array")
            os.remove(dist_file)
            os.remove(indices_file)
        
        for _ in range(5):
            _test_one_big()
    
    def testIapplyFuncChunk(self):
        #
        infilename = self.iapplyFuncInfile
        outfilename = self.iapplyFuncOutfile
        
        def _test_one_small():
            a = np_random.rand(200)
            b = np_random.rand(200)
            b.tofile(infilename)
            a.tofile(outfilename)
            iapply_func_chunk(outfilename, infilename, operator.add, chunk_size=50)
            assert_true(equal_arrays(a+b, np.fromfile(outfilename, dtype=a.dtype)), "applies add operation in place using disk-stored array")
            os.remove(infilename)
            os.remove(outfilename)
            
        for _ in range(50):
            _test_one_small()
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
