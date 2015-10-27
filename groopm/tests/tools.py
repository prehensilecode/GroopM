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
from nose.tools import assert_true
import numpy


def assert_equal_arrays(a, b, message):
    return assert_true(numpy.all(numpy.asarray(a) == numpy.asarray(b)), message)

def assert_almost_equal_arrays(a, b, message):
    return assert_true(numpy.all(numpy.around(a, 6) == numpy.around(b, 6)), message)

#-----------------------------------------------------------------------------
#Mocks

class MockProfileManager:
    """Mock profile data:

    index          1        2         4
    raw coverage   10,1     2,3       0,5
    raw kmers      3,2,2,3  4,5,0,1   1,6,1,2
    contigNames    "c1"     "c2"      "c4"
    binIds         0        0         0

    numContigs = 3
    stoitNames = ["s1", "s2"]
    numStoits = 2

    """
    def __init__(self):
        self.indices = numpy.array([1, 2, 4])

        # kmers
        self.kmerSigs = numpy.array([[3, 2, 2, 3], [4, 5, 0, 1], [1, 6, 1, 2]])
        self.contigLengths = numpy.sum(self.kmerSigs, axis=1)
        for i in range(4):
            self.kmerSigs[i] /= self.contigLengths[i]

        # coverage
        self.covProfiles = numpy.array([[10, 1], [2, 3], [0, 5]])
        for j in range(2):
            self.covProfiles[:, j] /= sum(self.covProfiles[:, j]) # normalise by total stoit reads
        for i in range(4):
            self.covProfiles[i] /= self.contigLengths[i] # normalise by contig lengths

        self.normCoverages = numpy.linalg.norm(self.covProfiles, axis=1)
        self.contigGCs = numpy.sum(self.kmerSigs[:, 2:]) / self.contigLengths
        self.binIds = [0, 0, 0]


###############################################################################
###############################################################################
###############################################################################
###############################################################################
