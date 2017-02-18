#!/usr/bin/env python
#  *-* coding: utf-8 *-*

from __future__ import division, absolute_import

from scipy.linalg.basic import solve, inv
from scipy.spatial.distance import cdist

import numpy as np

from MDmisc.xfrange import xfrange

from .config import CONF_maxx
from .config import CONF_maxy
from .config import CONF_minx
from .config import CONF_miny

################################################################################
#    Autre
################################################################################

def unit_vector( vector ):
    v = vector / np.linalg.norm( vector )
    return v

def _angle( v1, deg = False ):
    v2 = np.array( ( 1, 0 ) )
    
    return angle_between( v1, v2, deg )
    
def angle_between( v1, v2, deg = False ):
    v1 = unit_vector( v1 )
    v2 = unit_vector( v2 )
    
    angle = np.arccos( np.dot( v1, v2 ) )
    if np.isnan( angle ):
        if ( v1 == v2 ).all():
            return 0.0
        else:
            return None
    
    if v1[1] < 0:
        angle = 2 * np.pi - angle
    
    angle = np.mod( angle, 2 * np.pi )
    
    if deg == False:
        return angle
    else:
        return angle / np.pi * 180

################################################################################
#    Bookstein 1989
################################################################################

@np.vectorize
def U( r ):
    if r == 0.0:
        return 0.0
    else:
        return ( r ** 2 ) * np.log( r ** 2 )

@np.vectorize
def U2( r ):
    if r == 0.0:
        return 0.0
    else:
        return ( r ) * np.log( r )

def generate( src, dst ):
    n = src.shape[0]
    
    K = U( cdist( src, src, metric = "euclidean" ) )
    
    P = np.hstack( ( np.ones( ( n, 1 ) ), src ) )
    
    L = np.vstack( ( np.hstack( ( K, P ) ), np.hstack( ( P.T, np.zeros( ( 3, 3 ) ) ) ) ) )
    
    V = np.hstack( ( dst.T, np.zeros( ( 2, 3 ) ) ) )
    
    Wa = solve( L, V.T )
        
    W = Wa[ :-3 , : ]
    a = Wa[ -3: , : ]
    
    scale = np.sqrt( np.linalg.det( Wa[ -2: , : ] ) )
    shearing = angle_between( Wa[ -2: , 0 ], Wa[ -2: , 1 ], True )
    
    WK = np.dot( W.T, K )
    WKW = np.dot( WK, W )
    
    be = 0.5 * np.trace( WKW )
    
    return {
        'src':      src,
        'dst':      dst,
        'linear':   a,
        'scale':    scale,
        'shearing': shearing,
        'weights':  W,
        'be':       be
    }

def _p( XY, linear, W, src ):
    p = np.dot( np.hstack( ( 1, XY ) ), linear ) + \
        np.dot( U2( np.sum( ( src - XY ) ** 2, axis = -1 ) ), W )
    
    return p[ 0, : ]

def project( *args, **kwargs ):
    g = kwargs.get( 'g', None )
    x = kwargs.get( 'x' )
    y = kwargs.get( 'y' )
    theta = kwargs.get( 'theta', None )
    
    if g == None:
        linear = kwargs.get( 'linear' )
        W = kwargs.get( 'W' )
        src = kwargs.get( 'src' )

    else:
        linear = g['linear'],
        W = g['weights'],
        src = g['src']
    
    XY = np.array( [ x, y ] )
    
    p = _p( XY, linear, W, src )
    
    if theta != None:
        theta = theta / 180.0 * np.pi
        
        XY = np.array( [ x + 0.1 * np.cos( theta ), y + 0.1 * np.sin( theta ) ] )
        
        p2 = _p( XY, linear, W, src )

        ang = _angle( p2 - p, deg = True )
                
        p = np.insert( p, 2, ang )
        
        return p
    else:
        return p

def projo( *a, **k ):
    XY = k.pop( "XY" )
    if type( XY ) in [ tuple, list ]:
        XY = np.array( [ XY ] )
    
    g = k.pop( "g", None )
    if g != None:
        linear = g['linear'],
        W = g['weights'],
        src = g['src']
    else:
        linear = k.pop( "linear" )
        W = k.pop( "W" )
        src = k.pop( "src" )
    
    if XY.shape[ 1 ] == 2:
        return np.apply_along_axis( 
                lambda x: project( 
                    x = x[0],
                    y = x[1],
                    
                    linear = linear,
                    W = W,
                    src = src
                ), 1, XY
            )
    elif XY.shape[ 1 ] == 3:
        return np.apply_along_axis( 
                lambda x: project( 
                    x = x[0],
                    y = x[1],
                    theta = x[2],
                    
                    linear = linear,
                    W = W,
                    src = src
                ), 1, XY

            )

################################################################################
#    Anti-error
################################################################################

def revert():
    return
 
def image():
    return
 
def grid():
    return
 
def r( *args, **kwargs ):
    if len( args ) == 5:
        g, minx, maxx, miny, maxy = args
    
    else:
        g = kwargs.get( "g" )
        
        minx = kwargs.get( "minx", CONF_minx )
        maxx = kwargs.get( "maxx", CONF_maxx )
        miny = kwargs.get( "miny", CONF_miny )
        maxy = kwargs.get( "maxy", CONF_maxy )
    
    minx = int( minx )
    maxx = int( maxx )
    miny = int( miny )
    maxy = int( maxy )
    
    nbstep = 200
    stepx = ( maxx - minx ) / nbstep
    stepy = ( maxy - miny ) / nbstep
    
    plist = []
    
    for x in xfrange( minx, maxx, stepx ):
        p = projo( g = g, XY = [ x, miny ] )
        plist.append( p )
        
        p = projo( g = g, XY = [ x, maxy ] )
        plist.append( p )
    
    for y in xfrange( miny, maxy, stepy ):
        p = projo( g = g, XY = [ minx, y ] )
        plist.append( p )
        
        p = projo( g = g, XY = [ maxx, y ] )
        plist.append( p )
    
    plist = np.array( plist )
    
    minx, miny = np.amin( plist, axis = 0 )[ 0 ]
    maxx, maxy = np.amax( plist, axis = 0 )[ 0 ]
    
    return {
        'minx': minx,
        'miny': miny,
        'maxx': maxx,
        'maxy': maxy
    }
