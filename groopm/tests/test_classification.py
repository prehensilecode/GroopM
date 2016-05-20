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
from tools import assert_equal_arrays, assert_almost_equal_arrays
import numpy as np
import numpy.random as np_random
from groopm.classification import (parse_taxstring,
                                   Classification,
                                   greedy_clique_by_elimination,
                                  )

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def test_parse_taxstring():
    assert_equal_arrays(parse_taxstring("d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus"),
                        ["Archaea", "Euryarchaeota", "Methanococci", "Methanococcales", "Methanococcaceae", "Methanococcus"],
                        "`parse_taxstring` returns array of parsed taxonomic ranks")
    
    assert_equal_arrays(parse_taxstring("d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales"),
                        ["Bacteria", "Proteobacteria", "Betaproteobacteria", "Burkholderiales"],
                        "`parse_taxstring` returns array of parsed taxonomic ranks defined to order level")
    

def test_Classification():
    
    classification = Classification([
        "d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus",
        "d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus",
        "d__Archaea; p__Euryarchaeota; c__Thermococci; o__Thermococcales; f__Thermococcaceae; g__Pyrococcus",
        "d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales; f__Burkholderiaceae; g__Burkholderia",
        "d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales",
        "d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Nitrosomonadales; f__Nitrosomonadaceae; g__Nitrosomonas",
        ])
    
    assert_equal_arrays(classification.tags(0),
                        ["d__Archaea", "p__Euryarchaeota", "c__Methanococci", "o__Methanococcales", "f__Methanococcaceae", "g__Methanococcus"],
                        "`Classifcation.tags` returns array of tagged taxonomic levels")
    
    """
    Pairwise coherence levels:
    (Methanococcus, Methanococcus2): 0 (=Species)
    (Methanococcus, Pyrococcus): 5 (=Phylum)
    (Methanococcus, Burkholderia): 7 (=Root)
    (Methanococcus, Burkholderiales): 7
    (Methanococcus, Nitrosomonas): 7
    (Methanococcus2, Pyrococcus): 5
    (Methanococcus2, Burkholderia): 7
    (Methanococcus2, Burkholderiales): 7
    (Methanococcus2, Nitrosomonas): 7
    (Pyrococcus, Burkholderia): 7
    (Pyrococcus, Burkholderiales): 7
    (Pyrococcus, Nitrosomonas): 7
    (Burkholderia, Burkholderiales): 0 (=Species)
    (Burkholderia, Nitrosomonas): 4 (=Class)
    (Burkholderiales, Nitrosomonas): 4
    """
    print classification._table[:2]
    print classification._taxons[classification._table[:2]]
    print classification.distances()
    assert_equal_arrays(classification.distances(),
                        [0, 5, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 0, 4, 4],
                        "`Classification.distances` computes pairwise distance between classifications")
    
                                    
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
                

###############################################################################
###############################################################################
###############################################################################
###############################################################################
