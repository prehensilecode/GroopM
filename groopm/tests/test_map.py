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

from groopm.map import SingleMMapper, GraftMMapper
import os
import subprocess
import shutil

class TestMapper:
    
    @classmethod
    def setup_class(self):
        self.workingDir = './groopm/tests/test_map'
        self.contigsFile = '/srv/projects/paperpalooza/assemblies/5_all_in_after3nextseq19/5_all_in_after3nextseq19.fa'
        self.graftmPackageDir = '/srv/home/uqtlambe/code/groopm/graftm_packages'
        self.graftmPackageNames = ['DNGNGWU00001', 'DNGNGWU00002', 'DNGNGWU00003', 'DNGNGWU00007', 'DNGNGWU00009']
        self.graftmPackages = dict([(name, os.path.join(self.graftmPackageDir, name+'.gpkg')) for name in self.graftmPackageNames])
        self.singlemMapper = SingleMMapper(self.workingDir)
        self.graftmMapper = GraftMMapper(self.workingDir, self.graftmPackages)
        
    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.workingDir)
        os.mkdir(self.workingDir)

    def testSingleMMapper(self):
        (contigs, markers, taxstrings) = self.singlemMapper.getMappings(self.contigsFile)

    def testGraftMMapper(self):
        (contigs, markers, taxstrings) = self.graftmMapper.getMappings(self.contigsFile)

