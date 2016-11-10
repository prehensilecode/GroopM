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
from tools import (equal_arrays, almost_equal_arrays)
from groopm.distance import argrank
from groopm.stream import (pdist_chunk,
                           argsort_chunk_mergesort,
                           argrank_chunk
                          )

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
    
class TestStream:
    
    @classmethod
    def setup_class(self):
        
        self.workingDir = os.path.join(os.path.split(__file__)[0], "test_stream")
        os.mkdir(self.workingDir)
        self.pdistFile = os.path.join(self.workingDir, "test_stream.pdist.store")
        self.argsortInfile = os.path.join(self.workingDir, "test_stream.argsort.in.store")
        self.argsortOutfile = os.path.join(self.workingDir, "test_stream.argsort.out.store")
        self.argrankDistsFile = os.path.join(self.workingDir, "test_stream.argrank.dists.store")
        self.argrankIndicesFile = os.path.join(self.workingDir, "test_stream.argrank.indices.store")
        
    @classmethod
    def teardown_class(self):
        try:
            os.remove(self.pdistFile)
        except OSError:
            pass
        try:
            os.remove(self.argsortInfile)
            os.remove(self.argsortOutfile)
            os.remove(self.argsortInfile+".2")
            os.remove(self.argsortOutfile+".2")
        except OSError:
            pass
        try:
            os.remove(self.argrankDistsFile)
            os.remove(self.argrankIndicesFile)
        except OSError:
            pass
        os.rmdir(self.workingDir)
        
    def testPdistChunk(self):
        #
        filename = self.pdistFile
        f1 = np_random.rand(100, 50)
        d1 = sp_distance.pdist(f1, metric="euclidean")
        pdist_chunk(f1, filename, chunk_size=30, metric="euclidean")
        assert_true(equal_arrays(np.fromfile(filename, dtype=np.double),
                                 d1),
                    "computes same distances as unchunked function")
        os.remove(filename)
        
        # high mem
        f2 = np_random.rand(2**10, 50)
        d2 = sp_distance.pdist(f2, metric="euclidean")
        pdist_chunk(f2, filename, chunk_size=1e5, metric="euclidean")
        assert_true(equal_arrays(np.fromfile(filename, dtype=np.double),
                                 d2),
                    "computes same distances as unchunked function for a large-ish dataset")
    
    def testArgsortChunkMergesort(self):
        #
        infile = self.argsortInfile
        outfile = self.argsortOutfile
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
        
        # high mem
        d2 = np_random.rand(2**9*(2**10-1)).astype(np.double)
        d2.tofile(infile)
        argsort_chunk_mergesort(infile, self.argsortOutfile, chunk_size=1e5)
        arr = np.fromfile(infile, dtype=np.double)
        assert_true(np.all(arr[1:]>=arr[:-1]), "large-ish input file values are sorted")
        inds = np.fromfile(outfile, dtype=np.int)
        assert_true(np.all(d2[inds[1:]]>=d2[inds[:-1]]), "output file contains sorting indices for large-ish input")
        
    def testArgrankChunk(self):
        #
        dist_file = self.argrankDistsFile
        indices_file = self.argrankIndicesFile
        d1 = np_random.rand(190).astype(np.double)
        i1 = d1.argsort()
        d1[i1].tofile(dist_file)
        i1.tofile(indices_file)
        (x1, s1) = argrank_chunk(indices_file, dist_file, chunk_size=40)
        assert_true(equal_arrays(x1, argrank(d1, axis=None)),
                    "returns equal ranks to non-chunked function")
        assert_true(s1==190, "returns number of ranks")
        
        w2 = np_random.rand(190).astype(np.double)
        (x2, s2) = argrank_chunk(indices_file, dist_file, weight_fun=lambda i: w2[i], chunk_size=40)
        assert_true(almost_equal_arrays(x2, argrank(d1, weight_fun=lambda i: w2[i], axis=None)),
                    "correctly weights ranks when passed a weight function")
        assert_true(np.round(s2,6)==np.round(w2.sum(),6), "returns sum of weights")
        
        os.remove(dist_file)
        os.remove(indices_file)
        
        # high mem
        numbers = np.arange(2**9*(2**10-1))
        perm = numbers.copy()
        np_random.shuffle(perm)
        d2 = perm
        i2 = np.empty(len(numbers), dtype=np.int)
        i2[perm] = numbers
        d2.tofile(dist_file)
        i2.tofile(indices_file)
        
        (x3, s3) = argrank_chunk(indices_file, dist_file, chunk_size=1e5)
        assert_true(equal_arrays(x3, perm), "computes ranks of a large-ish permutation array")
        assert_true(s3==len(numbers), "returns rank count for large-ish array")
        
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
