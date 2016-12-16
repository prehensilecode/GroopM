#!/usr/bin/env python
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
import sys

# groopm imports
from groopm.data3 import DataManager
from groopm.groopmTimekeeper import TimeKeeper

###############################################################################
###############################################################################
###############################################################################
###############################################################################

if __name__ == '__main__':
    try:
        dbFileName = sys.argv[1]
    except IndexError:
        print "USAGE: %s DATABASE" % sys.argv[0]
        sys.exit(1)
    timer = TimeKeeper()
    DataManager().checkAndUpgradeDB(dbFileName, timer, silent=False)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
