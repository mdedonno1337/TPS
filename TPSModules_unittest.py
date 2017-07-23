#!/usr/bin/python
# -*- coding: UTF-8 -*-

from TPS import TPSCy, TPSpy, TPS_generate, TPS_image, TPS_module

from hashlib import md5
from PIL import Image
from PIL.ImageDraw import ImageDraw

import numpy as np
import unittest

import random
random.seed( 1337 )

src = [ [ 3.6929, 10.3819 ], [ 6.5827, 8.8386 ], [ 6.7756, 12.0866 ], [ 4.8189, 11.2047 ], [ 5.6969, 10.0748 ] ]
dst = [ [ 3.9724, 6.5354 ], [ 6.6969, 4.1181 ], [ 6.5394, 7.2362 ], [ 5.4016, 6.4528 ], [ 5.7756, 5.1142 ] ]

class TestCore( unittest.TestCase ):
    def __init__( self, *args, **kwargs ):
        global src, dst
        self.g = TPS_generate( src, dst )
        
        return super( TestCore, self ).__init__( *args, **kwargs )
    
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
            for key in list( set( self.g.keys() ) | set( expected.keys() ) ):
                self.assertTrue( key in self.g, "%s: '%s' key not present" % ( TPSModule.lang(), key ) )
                self.assertTrue( np.allclose( self.g[ key ], expected[ key ] ), "\n\nError on: '%s' -> '%s'\ngot\n\t%s\nexpected\n\t%s" % ( TPSModule.lang(), key, self.g[ key ], expected[ key ] ) )
    
    def test_project( self ):
        expected = [ 3.9724, 6.5354 ]
        
        for TPSModule in [ TPSCy, TPSpy ]:
            xp, yp, _ = TPSModule.project( self.g, 3.6929, 10.3819, 0 )
            p = [ xp, yp ]
            
            self.assertTrue( np.allclose( p, expected ), "\n\nError on: '%s' -> project\ngot\n\t%s\nexpected\n\t%s" % ( TPSModule.lang(), p, expected ) )

class TestImages( unittest.TestCase ):
    def __init__( self, *args, **kwargs ):
        global src, dst
        self.g = TPS_generate( src, dst )
        
        self.img = Image.new( "L", ( 500, 500 ), 255 )
        draw = ImageDraw( self.img )
        for d in xrange( 10, 250, 10 ):
            draw.ellipse( ( d, d, 500 - d, 500 - d ) )
        
        return super( TestImages, self ).__init__( *args, **kwargs )
    
    def test_r( self ):
        expected = {
            'minx': 3.9529122211962227,
            'miny': 3.0935381368930255,
            'maxx': 8.467498872262812,
            'maxy': 8.324242405510011
        }
        
        for TPSModule in [ TPSCy, TPSpy ]:
            r = TPSModule.r( g = self.g, minx = 3.8, maxx = 8.6, miny = 8, maxy = 12.5 )
            
            for key in expected.keys():
                self.assertTrue( np.allclose( r[ key ], expected[ key ] ), "\n\nError on: '%s' -> '%s'\ngot\n\t%s\nexpected\n\t%s" % ( TPSModule.lang(), key, r[ key ], expected[ key ] ) )
    
    def test_grid( self ):
        expected = "314a94ea0a8cc3fe19a32bc05c39b168"
        
        for TPSModule in [ TPSCy, TPSpy ]:
            grid = TPSModule.grid( g = self.g, minx = 3.8, maxx = 8.6, miny = 8, maxy = 12.5, res = 2500 )
            h = md5( grid.tobytes() ).hexdigest()
            
            self.assertEqual( h, expected, "\n\nError on: '%s' -> 'grid'\ngot\n\t%s\nexpected\n\t%s" % ( TPSModule.lang(), h, expected ) )
    
    def test_image( self ):
        self.assertEqual( TPS_module(), "Cython", "Cython module not loaded correctly. Check the compilation or the import." )
        
        expected = "72538f9250785848ae956a9954347de7"
        
        img = TPS_image( inimg = self.img, g = self.g )
        h = md5( img.tobytes() ).hexdigest()
        
        self.assertEqual( h, expected, "\n\nError on: '%s' -> 'image'\ngot\n\t%s\nexpected\n\t%s" % ( TPS_module(), h, expected ) )
    
if __name__ == '__main__':
    unittest.main()
