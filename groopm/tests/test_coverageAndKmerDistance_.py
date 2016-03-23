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

# system imports
from nose.tools import assert_equals
from tools import assert_equal_arrays, assert_almost_equal_arrays, DummyProfileManager
import numpy
from groopm.coverageAndKmerDistance import CoverageAndKmerDistanceTool

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def test_coverage_and_kmer_distance_tool():

    pm = DummyProfileManager(indices=numpy.array([0, 1, 2, 3]),
                             covProfiles=numpy.array([[1, 3],
                                                      [2, 3],
                                                      [1, 2],
                                                      [2, 2]]),
                             kmerSigs=numpy.array([[0,   0,   0.5, 0.5],
                                                   [1,   0,   0,   0  ],
                                                   [0,   0.5, 0,   0.5],
                                                   [0.5, 0,   0.5, 0  ]]),
                             contigGCs=numpy.array([]),
                             contigNames=numpy.array([]),
                             contigLengths=numpy.array([]),
                             binIds=numpy.array([]),
                             stoitNames=numpy.array([]))
    assert_equals(pm.numContigs, 4, "has five contigs")

    # contig distances, computed by hand
    cov_dist_matrix = numpy.sqrt([[0, 1, 1, 2],
                                  [1, 0, 2, 1],
                                  [1, 2, 0, 1],
                                  [2, 1, 1, 0]])
    kmer_dist_matrix = numpy.sqrt([[0,   1.5, 0.5, 0.5],
                                   [1.5, 0,   1.5, 0.5],
                                   [0.5, 1.5, 0,   1  ],
                                   [0.5, 0.5, 1,   0  ]])

    ct = CoverageAndKmerDistanceTool(pm)
    assert_equal_arrays(ct.getDistances([0, 1, 3]),
                        numpy.array([cov_dist_matrix[numpy.ix_([0, 1, 3], [0, 1, 3])],
                                     kmer_dist_matrix[numpy.ix_([0, 1, 3], [0, 1, 3])]]),
                        "returns squareform distance array of selected contigs")
    assert_equal_arrays(ct.getDistances([0, 1, 3], [2]),
                        numpy.array([cov_dist_matrix[numpy.ix_([0, 1, 3], [2])],
                                     kmer_dist_matrix[numpy.ix_([0, 1, 3], [2])]]),
                        "returns distance array between two groups of contigs")

    # contig mediod
    assert_equal_arrays(ct.getMediod([1, 2, 3]),
                        2, # computed by hand
                        "computes index of group mediod")

    # closest points
    assert_equal_arrays(ct.associateWith([0, 3], [1, 2]),
                        [1, 0], #=> 1 closest to 3, 2 closest to 0, computed by hand
                        "computes index of closest contig")


###############################################################################
###############################################################################
###############################################################################
###############################################################################
