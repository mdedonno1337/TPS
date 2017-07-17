#!/usr/bin/python
# -*- coding: UTF-8 -*-

from TPS import TPSCy, TPSpy, TPS_generate
import numpy as np
import unittest

src = [ [ 3.6929, 10.3819 ], [ 6.5827, 8.8386 ], [ 6.7756, 12.0866 ], [ 4.8189, 11.2047 ], [ 5.6969, 10.0748 ] ]
dst = [ [ 3.9724, 6.5354 ], [ 6.6969, 4.1181 ], [ 6.5394, 7.2362 ], [ 5.4016, 6.4528 ], [ 5.7756, 5.1142 ] ]

class TestFonctionGet( unittest.TestCase ):
    def test_generate( self ):
        global src, dst
        
        src = np.asarray( src )
        dst = np.asarray( dst )
        
        expected = {
            'src': [[3.6929, 10.3819],
                    [6.5827, 8.8386],
                    [6.7756, 12.0866],
                    [4.8189, 11.2047],
                    [5.6969, 10.0748]],
            
            'dst': [[3.9724, 6.5354],
                    [6.6969, 4.1181],
                    [6.5394, 7.2362],
                    [5.4016, 6.4528],
                    [5.7756, 5.1142]],
            
            'weights': [[-0.038030135354319566, 0.042446928059415835],
                        [0.023187750609016955, 0.01591661176988479],
                        [-0.024755055674439617, 0.028813480594372044],
                        [0.0797822576121927, -0.045425521193799244],
                        [-0.04018481719245052, -0.041751499229873326]],
            
            'linear': [[1.3549958177370123, -2.945963080998363],
                       [0.8747258748666799, -0.2955605626378672],
                       [-0.028860413349607213, 0.9216325921181663]],
            
            'be': 0.0429994895805986,
            'mirror': False,
            'scale': 0.8931102258056604,
            'shearing': 250.32963702546,
        }
        
        for TPSModule in [ TPSCy, TPSpy ]:
            g = TPSModule.generate( src, dst )
            
            for key in list( set( g.keys() ) | set( expected.keys() ) ):
                self.assertTrue( key in g, "%s: '%s' key not present" % ( TPSModule.lang(), key ) )
                self.assertTrue( np.allclose( g[ key ], expected[ key ] ), "\n\nError on: '%s' -> '%s'\ngot\n\t%s\nexpected\n\t%s" % ( TPSModule.lang(), key, g[ key ], expected[ key ] ) )
    
    def test_project( self ):
        global src, dst
        
        g = TPS_generate( src, dst )
        
        for TPSModule in [ TPSCy, TPSpy ]:
            xp, yp, _ = TPSModule.project( g, 3.6929, 10.3819, 0 )
            p = [ xp, yp ]
            expected = [ 3.9724, 6.5354 ]
            
            self.assertTrue( np.allclose( p, expected ), "\n\nError on: '%s' -> project\ngot\n\t%s\nexpected\n\t%s" % ( TPSModule.lang(), p, expected ) )
    
    def test_r( self ):
        global src, dst
        
        g = TPS_generate( src, dst )
        
        expected = {
            'minx': 3.9529122211962227,
            'miny': 3.0935381368930255,
            'maxx': 8.467498872262812,
            'maxy': 8.324242405510011
        }
        
        for TPSModule in [ TPSCy, TPSpy ]:
            r = TPSModule.r( g = g, minx = 3.8, maxx = 8.6, miny = 8, maxy = 12.5 )
            
            for key in expected.keys():
                self.assertTrue( np.allclose( r[ key ], expected[ key ] ), "\n\nError on: '%s' -> '%s'\ngot\n\t%s\nexpected\n\t%s" % ( TPSModule.lang(), key, r[ key ], expected[ key ] ) )
    
if __name__ == '__main__':
    unittest.main()
