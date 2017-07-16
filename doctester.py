#!/usr/bin/python
# -*- coding: UTF-8 -*-

import doctest
import unittest

import TPS.__init__
from TPS import TPS_generate

################################################################################
# 
#    Tests
# 
################################################################################

def TPStests():
    tests = unittest.TestSuite()
    
    ############################################################################
    # 
    #    Test for empty NIST object
    # 
    ############################################################################
    
    src = [ [ 3.6929, 10.3819 ], [ 6.5827, 8.8386 ], [ 6.7756, 12.0866 ], [ 4.8189, 11.2047 ], [ 5.6969, 10.0748 ] ]
    dst = [ [ 3.9724, 6.5354 ], [ 6.6969, 4.1181 ], [ 6.5394, 7.2362 ], [ 5.4016, 6.4528 ], [ 5.7756, 5.1142 ] ]
    
    vars = {
        'src': src,
        'dst': dst,
        'g': TPS_generate( src, dst )
    }
    
    tests.addTests( doctest.DocTestSuite( TPS.__init__, vars ) )
    
    return tests

if __name__ == "__main__":
    unittest.TextTestRunner( verbosity = 2 ).run( TPStests() )
else:
    def load_tests( loader, tests, ignore ):
        return TPStests()
