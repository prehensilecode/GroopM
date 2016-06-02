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
import numpy
import subprocess
import os
import sys
from groopm.data3 import __current_GMDB_version__

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class TestDBUpgrade:
    
    @classmethod
    def setup_class(self):
        """Setup class variables before any tests."""
        
        self.dataDir = os.path.join(os.path.split(__file__)[0], "db_upgrade_data")
        
        # following files already exist
        self.dbVersions = [0, 5]
        self.dbFiles = dict(zip(self.dbVersions,
                                [os.path.join(self.dataDir, "v%d.gm" % ver) for ver in self.dbVersions]))
        self.dbUpgradeScript = os.path.join(self.dataDir, "run_db_upgrade.py")
        self.fasta = os.path.join(self.dataDir, "test.fa")
        self.markerFile = os.path.join(self.dataDir, "lineages.txt")
                                
        # generated copies
        self.dbCopies = dict(zip(self.dbVersions, [os.path.join(self.dataDir, "v%dto%d.gm" % (ver, __current_GMDB_version__)) for ver in self.dbVersions]))
        

        
    @classmethod
    def teardown_class(self):
        for ver in self.dbVersions:
            self.rmTestFile(self.dbCopies[ver])
            
    @classmethod
    def rmTestFile(self, path):
        if os.path.exists(path):
            os.remove(path)
        else:
            sys.stderr.write("No file: %s\n" % path)
            
    def generate_db_copy(self, ver):
        cmd = "cp %s %s" % (self.dbFiles[ver], self.dbCopies[ver])
        subprocess.check_call(cmd, shell=True)
        return self.dbCopies[ver]
        
    def test_db_upgrade_v0(self):
        db_copy = self.generate_db_copy(0)
        cmd = "PYTHONPATH=%s:$PYTHONPATH %s %s" % (os.getcwd(), self.dbUpgradeScript, db_copy)
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stdoutdata, _stderrdata) = proc.communicate("\n".join([self.fasta, self.markerFile]))
        if proc.returncode != 0:
            print stdoutdata
            raise AssertionError("Upgrade script returned error code: %d" % proc.returncode)
        
    def test_db_upgrade_v5(self):
        db_copy = self.generate_db_copy(5)
        cmd = "PYTHONPATH=%s:$PYTHONPATH %s %s" % (os.getcwd(), self.dbUpgradeScript, db_copy)
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stdoutdata, _stderrdata) = proc.communicate(self.markerFile)
        if proc.returncode != 0:
            print stdoutdata
            raise AssertionError("Upgrade script returned error code: %d" % proc.returncode)
        
            
        


###############################################################################
###############################################################################
###############################################################################
###############################################################################
