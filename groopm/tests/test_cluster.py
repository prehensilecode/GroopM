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

# local imports
from tools import equal_arrays, is_isomorphic
from groopm.cluster import ClusterQualityEngine, FlatClusterEngine

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class QualityEngineTester(ClusterQualityEngine):
    def __init__(self, leaf_data, score_fn):
        self.getLeafData = lambda: leaf_data
        self.getScore = lambda data: score_fn(data)
    
def test_ClusterQualityEngine():
    #
    """ The tree used for the following tests is represented below:
        0---+
        2-+ |-6
        1 |-5
        |-4
        3
    """
    Z = np.array([[1., 3., 1., 2.],
                  [2., 4., 1., 3.],
                  [0., 5., 2., 4.]])
                   
    """Test that a Quality Engine using max with a single leaf data point
    will propagate score to ancestors. The following data is assigned to leaves:
        0:[1]--+
        2:[]-+ |-6
        1:[] |-5
        |-4--+
        3:[]    
    """
    qe1 = QualityEngineTester({0:[1]}, max)
    assert_true(equal_arrays(qe1.makeScores(Z),
                             [1, 0, 0, 0, 0, 0, 1]),
                "computes max of a single leaf data point for all ancestors")
                                
    """Test that a Quality Engine using len and a pair of leaves with data will
    combined leaf data and propagate to ancestors. The following data is assinged to
    leaves:
        0:[]------+
        2:["a"]-+ |-6
        1:["x"] |-5
        |-4-----+
        3:[]    
    """
    qe2 = QualityEngineTester({1:["a"], 2:["x"]}, len)
    assert_true(equal_arrays(qe2.makeScores(Z),
                             [0, 1, 1, 0, 1, 2, 2]),
                "computes length of leaf data in case of pair of leaf data points")
                                              
    """Test that a Quality Engine using a range measure will compute the range
    of data from the full set of leaves. Leaf data assigned as follows:
        0:[]------+
        2:[1]---+ |-6
        1:[2,2] |-5
        |-4-----+
        3:[1,0]    
    """
    qe3 = QualityEngineTester({1: [2, 2], 2: [1], 3:[1, 0]}, lambda x: max(x) - min(x))
    assert_true(equal_arrays(qe3.makeScores(Z),
                             [0, 0, 0, 1, 2, 2, 2]),
                "computes non-trivial function of combined leaf data points")
             
             
class ClusterEngineTester(FlatClusterEngine):
    def __init__(self, scores, qualities):
        self.getScores = lambda _: scores
        self.isNoiseCluster = lambda _: qualities
    
def test_ClusterEngineTester(): 
    #
    
    """Test that the cluster engine with uniform zero scores will merge low
    quality clusters. The tree for this test is represented below.
    
    [1:1]
     |-[3:0]-+
    [0:1]    |-[4:0]
    [2:0]----+
    """
    Z1 = np.array([[0., 1., 1., 2.],
                   [2., 3., 2., 3.]])
    q1 = [True, True, False, False, False]
    ce1 = ClusterEngineTester([0]*len(q1), q1)
    assert_true(is_isomorphic(ce1.makeClusters(Z1),
                              [1, 1, 2]),
                "ClusterEngine with uniform zero scores merges clusters "
                "designated low quality")
    
    """Clustering with nested equal height branches.
          ==Equal==
    [1:1]-=+      =                [1:1]-+
          =|-[3:0]=                      |
    [0:0]-=+  |   = => Treat as => [0:0]-+-[4:0]
          =  [4:0]=                      |
    [2:0]-=---+   =                [2:0]-+
          =========
    """
    Z2 = np.array([[0., 1., 1., 2.],
                   [2., 3., 1., 3.]])
    q2 = [True, False, False, False, False]
    ce2 = ClusterEngineTester([0]*len(q2), q2)
    assert_true(is_isomorphic(ce2.makeClusters(Z2),
                              [1, 2, 3]),
                "ClusterEngine with uniform zero scores won't partially merge "
                "nested clusters when parent cluster won't merge")
    
    """Test that the cluster engine with uniform quality clusters will
    merge child clusters to improve cluster score total. The tree for
    this test is represented below.
    
    [1:1]
     |-[3:2]-+
    [0:0]    |-[4:1]
    [2:1]----+
    """
    Z3 = np.array([[0., 1., 1., 2.],
                   [2., 3., 2., 2.]])
    v3 = [0, 1, 1, 2, 1]
    ce3 = ClusterEngineTester(v3, [False]*len(v3))
    assert_true(is_isomorphic(ce3.makeClusters(Z3),
                              [1, 1, 2]),
                "ClusterEngine with uniform quality clusters merges "
                "child clusters if merged cluster improves combined "
                "score")
    
    """Clustering with nested equal height branches.
          ==Equal==
    [1:1]-=--+    =              [1:1]-+
          = [3:2] =                    |
    [0:0]-=--+    = >=Treat as=> [0:0]-+-[4:1]
          = [4:1] =                    |
    [2:1]-=--+    =              [2:1]-+
          =========
    """
    Z4a = np.array([[0., 1., 1., 2.],
                    [2., 3., 1., 3.]])
    Z4b = np.array([[0., 2., 1., 2.],
                    [1., 3., 1., 3.]]) # equivalent
    Z4c = np.array([[1., 2., 1., 2.],
                    [0., 3., 1., 3.]]) # equivalent
    v4 = [0, 1, 1, 2, 1]
    ce4 = ClusterEngineTester(v4, [False]*len(v4))
    assert_true(is_isomorphic(ce4.makeClusters(Z4a),
                              [1, 2, 3]) and
                is_isomorphic(ce4.makeClusters(Z4b),
                              [1, 2, 3]) and
                is_isomorphic(ce4.makeClusters(Z4c),
                              [1, 2, 3]),
                "ClusterEngine with uniform quality clusters won't partially "
                "merge nested clusters when parent cluster won't merge")
    
    """Test that the cluster engine with uniform quality clusters will
    not merge clusters if combined best descendent cluster score is not
    improved. The tree for this test is represented below.
    
    [1:1]
     |-[3:0]-+
    [0:0]    |-[4:2]
    [2:1]----+
    """
    Z5 = np.array([[0., 1., 1., 2.],
                   [2., 3., 2., 2.]])
    v5 = [0, 1, 1, 0, 2]
    ce5 = ClusterEngineTester(v5, [False]*len(v5))
    assert_true(is_isomorphic(ce5.makeClusters(Z5),
                              [1, 2, 3]),
                "ClusterEngine with uniform quality clusters doesn't merge "
                "descendent clusters if merged cluster combined  score is "
                "not improved")
    
    
    """Test that the cluster engine correctly propagates scores for equal
    height nested branches, so that a parent cluster will merge all nested
    clusters if it improves the combined cluster score of non-nested
    clusters below it. The tree for this test is represented below:
        
    [2:0]----------+                    [2:0]---------+
          ==Equal==|                                  |
    [1:1]-=--+    =|-[8:1]              [1:1]-+       |-[8:1]
          = [5:1] =|                          |       |
    [0:0]-=--+    =|       >=Treat as=> [0:0]-+       |
          = [7:3]-=+                          |-[7:3]-+
    [3:0]-=--+    =                     [3:0]-+
          = [6:4] =                           |
    [4:1]-=--+    =                     [4:1]-+
          =========
    """
    Z6a = np.array([[ 0.,  1., 1., 2.],
                    [ 3.,  4., 1., 2.],
                    [ 5.,  6., 1., 4.],
                    [ 2.,  7., 3., 5.]])
    Z6b = np.array([[ 0.,  1., 1., 2.],
                    [ 3.,  5., 1., 2.],
                    [ 4.,  6., 1., 4.],
                    [ 2.,  7., 3., 5.]]) # equivalent
    v6 = [0, 1, 0, 0, 1, 1, 4, 3, 1]
    ce6 = ClusterEngineTester(v6, [False]*len(v6))
    assert_true(is_isomorphic(ce6.makeClusters(Z6a),
                              [1, 1, 2, 1, 1]) and
                is_isomorphic(ce6.makeClusters(Z6b),
                              [1, 1, 2, 1, 1]),
                "merges nested clusters of equal height when parent cluster "
                "would be merged with non-nested descendents")
        
    
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
