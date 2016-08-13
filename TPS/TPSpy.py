#!/usr/bin/env python
#  *-* coding: utf-8 *-*

from scipy.linalg.basic import solve, inv
from scipy.spatial.distance import cdist
import numpy as np

################################################################################
#    Autre
################################################################################

def unit_vector( vector ):
    v = vector / np.linalg.norm( vector )
    return v

def angle_between( v1, deg = False ):
    v1_u = unit_vector( v1 )
    v2_u = np.array( ( 1, 0 ) )
    
    angle = np.arccos( np.dot( v1_u, v2_u ) )
    if np.isnan( angle ):
        if ( v1_u == v2_u ).all():
            return 0.0
        else:
            return None
    
    if v1_u[1] < 0:
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
    
    WK = np.dot( W.T, K )
    WKW = np.dot( WK, W )
    
    be = 0.5 * np.trace( WKW )
    
    return {
        'src':      src,
        'dst':      dst,
        'linear':   a,
        'weights':  W,
        'be':       be
    }

def _p( XY, linear, W, src ):
    p = np.dot( np.hstack( ( 1, XY ) ), linear ) + \
        np.dot( U2( np.sum( ( src - XY ) ** 2, axis = -1 ) ), W )
    
    return p[ 0, : ]

def project( g, x, y, theta = None ):
    linear = g['linear'],
    W = g['weights'],
    src = g['src']
    
    XY = np.array( [ x, y ] )
    
    p = _p( XY, linear, W, src )
    
    if theta != None:
        theta = theta / 180.0 * np.pi
        
        XY = np.array( [ x + 0.1 * np.cos( theta ), y + 0.1 * np.sin( theta ) ] )
        
        p2 = _p( XY, linear, W, src )

        ang = angle_between( p2 - p, deg = True )
                
        p = np.insert( p, 2, ang )
        
        return p
    else:
        return p

def projo( **k ):
    XY = k.pop( "XY" )
    
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

def r():
    return
 
def revert():
    return
 
def image():
    return
 
def grid():
    return
 
def range():
    return
