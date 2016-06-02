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
from tools import assert_equal_arrays
import numpy as np
from groopm.data3 import ClassificationEngine

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class TestClassificatonEngine:
    _ce = ClassificationEngine()
    
    def test_parse_taxstring(self):
        assert_equal_arrays(self._ce.parse_taxstring("d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus"),
                            ["Archaea", "Euryarchaeota", "Methanococci", "Methanococcales", "Methanococcaceae", "Methanococcus"],
                            "`parse_taxstring` returns array of parsed taxonomic ranks")
        
        assert_equal_arrays(self._ce.parse_taxstring("d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales"),
                            ["Bacteria", "Proteobacteria", "Betaproteobacteria", "Burkholderiales"],
                            "`parse_taxstring` returns array of parsed taxonomic ranks defined to order level")
    

    def test_parse(self):
        (table, taxons) = self._ce.parse([
            "d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus",
            "d__Archaea; p__Euryarchaeota; c__Methanococci; o__Methanococcales; f__Methanococcaceae; g__Methanococcus",
            "d__Archaea; p__Euryarchaeota; c__Thermococci; o__Thermococcales; f__Thermococcaceae; g__Pyrococcus",
            "d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales; f__Burkholderiaceae; g__Burkholderia",
            "d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Burkholderiales",
            "d__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Nitrosomonadales; f__Nitrosomonadaceae; g__Nitrosomonas",
            ])
        
        assert_equal_arrays(taxons[table[0]],
                            ["Archaea", "Euryarchaeota", "Methanococci", "Methanococcales", "Methanococcaceae", "Methanococcus", ""],
                            "`parse` returns table of taxon tag indices and array of tags")
        
        #Pairwise distances:
        #(Methanococcus, Methanococcus2): 0 (=Species)
        assert_true(self._ce.getDistance(table[0], table[1])==0,
                    "`getDistance` returns 0 distance between two equal classifications")
        #(Methanococcus, Pyrococcus): 5 (=Phylum)
        assert_true(self._ce.getDistance(table[0], table[2])==5,
                    "`getDistance` returns 5 distance between classifications equal up to phylum level")
        #(Methanococcus, Burkholderia): 7 (=Root)
        assert_true(self._ce.getDistance(table[0], table[3])==7,
                    "`getDistance` returns 7 distance between classifications equal only at root level")
        #(Burkholderia, Burkholderiales): 0 (=Species)
        assert_true(self._ce.getDistance(table[3], table[4])==0,
                    "`getDistance` returns 0 distance between classifications equal at all defined levels")
        #(Burkholderiales, Nitrosomonas): 4 (=Class)
        assert_true(self._ce.getDistance(table[4], table[5])==4,
                    "`getDistance` returns 4 distance between classifications equal up to class level")
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
