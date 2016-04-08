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
from groopm.utils import (greedy_clique_by_elimination,
                         )

###############################################################################
###############################################################################
###############################################################################
###############################################################################               
               
               
def test_greedy_clique_by_elimination():
    C = np.array([[True , True , False],
                  [True , True , False],
                  [False, False, True ]]) # 0, 1 in clique
    node_perm = np_random.permutation(3)
    C_perm = C[np.ix_(node_perm, node_perm)]
    indices_perm = np.empty(3, dtype=int)
    indices_perm[node_perm] = np.arange(3)
    
    assert_equal_arrays(greedy_clique_by_elimination(C_perm),
                        np.sort(indices_perm[:2]),
                        "`greedy_clique_by_elimination` returns indices of clique")
    
    # two cliques with n-1 connecting edges
    C = np.array([[True , True , True , False, True , True ],
                  [True , True , True , True , False, True ],
                  [True , True , True , True , True , False],
                  [False, True , True , True , True , True ],
                  [True , False, True , True , True , True ],
                  [True , True , False, True , True , True ]]) # 0, 1, 2 and 3, 4, 5 cliques
    node_perm = np_random.permutation(6)
    C_perm = C[np.ix_(node_perm, node_perm)]
    indices_perm = np.empty(6, dtype=int)
    indices_perm[node_perm] = np.arange(6)
    
    assert_true(len(greedy_clique_by_elimination(C_perm)) == 3,
                "`greedy_clique_by_elimination` computes correct clique size for two highly connected equal sized cliques")
                
    # two cliques with universally connected link node
    C = np.array([[True , True , True , True , False, False],
                  [True , True , True , True , False, False],
                  [True , True , True , True , False, False],
                  [True , True , True , True , True , True ],
                  [False, False, False, True , True , True ],
                  [False, False, False, True , True , True ]]) #0, 1, 2, 3 and 3, 4, 5 cliques
    node_perm = np_random.permutation(6)
    C_perm = C[np.ix_(node_perm, node_perm)]
    indices_perm = np.empty(6, dtype=int)
    indices_perm[node_perm] = np.arange(6)
    
    assert_equal_arrays(greedy_clique_by_elimination(C_perm),
                        np.sort(indices_perm[:4]),
                        "`greedy_clique_by_elimination` computes the larger of two overlapping cliques")

                
    # fully disconnected array throws error
                        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
