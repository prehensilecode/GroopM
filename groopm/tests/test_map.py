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
__copyright__ = "Copyright 2016"
__credits__ = ["Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "tim.lamberton@gmail.com"

###############################################################################

from groopm.map import SingleMMapper, GraftMMapper, GRAFTM_PACKAGE_DIR
from groopm.utils import FastaReader
import os
import subprocess
import shutil

class TestMapper:
    
    @classmethod
    def setup_class(self):
        
        self.dataDir = os.path.join(os.path.split(__file__)[0], "map_data")
        self.workingDir = os.path.join(self.dataDir, "test_map")
        os.mkdir(self.workingDir)
        self.contigsFile = os.path.join(self.dataDir, 'contigs.fa')
        self.graftmPackageNames = ['DNGNGWU00001', 'DNGNGWU00002', 'DNGNGWU00003', 'DNGNGWU00007', 'DNGNGWU00009']
        self.graftmPackages = [os.path.join(GRAFTM_PACKAGE_DIR, name+'.gpkg') for name in self.graftmPackageNames]
        self._cid2Indices = None #cache contig index mapping
        
    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.workingDir)

    def getContigNames(self):
        if self._cid2Indices is None:
            reader = FastaReader()
            con_names = []
            with open(self.contigsFile, 'r') as f:
                for (cid,_seq) in reader.readFasta(f):
                    con_names.append(cid)
            self._cid2Indices = dict(zip(range(len(con_names)), sorted(con_names)))
            
        return self._cid2Indices
        
    def testSingleMMapper(self):
        cid2Indices = self.getContigNames()
        mapper = SingleMMapper(self.workingDir, silent=True)
        (con_indices, markers, taxstrings) = mapper.getMappings(self.contigsFile, cid2Indices)

    def testGraftMMapper(self):
        cid2Indices = self.getContigNames()
        mapper = GraftMMapper(self.workingDir, self.graftmPackages, silent=True)
        (con_indices, markers, taxstrings) = mapper.getMappings(self.contigsFile, cid2Indices)

