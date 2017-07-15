#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import subprocess
import unittest

from MDmisc.egit import git_version
from MDmisc.eprint import eprint

################################################################################

def _exe( cmd, wd ):
    return subprocess.Popen( cmd, cwd = wd, stdout = subprocess.PIPE, stderr = subprocess.PIPE ).communicate()

################################################################################

try:
    version = git_version( describe = [ "--match", "v*" ] )[ "commit_describe" ]

except:
    version = "dev"
    
finally:
    os.chdir( os.path.split( os.path.abspath( __file__ ) )[ 0 ] )
    
    verstring = "__version__ = '%s'" % version
    
    with open( "TPS/version.py", "w+" ) as fp:
        fp.write( verstring )
        
    with open( "doc/version.py", "w+" ) as fp:
        fp.write( verstring )

################################################################################

wd = os.path.abspath( "./doc" )

cmd = [ 'make', 'html' ]

stdout, stderr = _exe( cmd, wd )

print( stdout )
eprint( stderr )
