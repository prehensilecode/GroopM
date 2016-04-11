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
import random
from groopm.classification import (_Classification,
                                   _classification_pdist,
                                  )

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def test_Classification():
    methanococcus = _Classification("d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus")
    ranks = ["Archaea", "Euryarchaeota", "Methanococci", "Methanococcales", "Methanococcaceae", "Methanococcus"]
    assert_equal_arrays(methanococcus.ranks,
                        ranks,
                        "`_Classification.ranks` returns array of parsed taxonomic ranks")
                        
    assert_equal_arrays([methanococcus.taxon(field) for field in ["domain", "phylum", "class", "order", "family", "genus"]],
                         ranks,
                         "`_Classifcation.taxon` reports specific taxonomic levels")
                         
    methanococcus2 = _Classification("d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus")
    assert_true(methanococcus.distance(methanococcus2)==0, "`_Classification.distance` for identical classifications is 0.")
    
    pyrococcus = _Classification("d__Archaea; p__Euryarchaeota; c__Thermococci; o__Thermococcales; f__Thermococcaceae; g__Pyrococcus")
    assert_true(methanococcus.distance(pyrococcus)==5, "`_Classification.distance` computes taxonomic rank of divergence for two genus level classifications")
    
    burkholderia = _Classification("d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales; f__Burkholderiaceae; g__Burkholderia")
    burkholderiales = _Classification("d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales")
    assert_true(burkholderia.distance(burkholderiales)==0, "`_Classification.distance` returns 0 if no divergence is found for any ranks")
    
    nitrosomonas = _Classification("d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Nitrosomonadales; f__Nitrosomonadaceae; g__Nitrosomonas")
    assert_true(burkholderiales.distance(nitrosomonas)==4, "`_Classification.distance` computes taxonomic rank of divergence for classifications to order and genus level")
    
    assert_equal_arrays(_classification_pdist([methanococcus, pyrococcus, burkholderia, burkholderiales, nitrosomonas]),
                        [5, 7, 7, 7, 7, 7, 7, 0, 4, 4],
                        "`_classification_pdist` computes pairwise distance between classifications")
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
