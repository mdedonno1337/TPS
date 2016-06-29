#!/usr/bin/env python
#  *-* coding: cp850 *-*

from lib.TPS.TPSCy import generate as gCy
from lib.TPS.TPSCy import generatePy as gPyCy
from lib.TPS.TPSpy import generate as gPy
from lib.printCapturing import Capturing
from lib.profiler import Timer

from tqdm._tqdm import trange

import matplotlib.pyplot as plt
import numpy as np

################################################################################
#
#    TPSCy modification
#
#
# @np.vectorize
# def UPy( r ):
#     if r == 0.0:
#         return 0.0
#     else:
#         return ( r ** 2 ) * log( r ** 2 )
# 
# def generatePy(
#         double[ : , : ] src not None,
#         double[ : , : ] dst not None,
#     ):
#     
#     n = src.shape[0]
#     
#     K = UPy( cdist( src, src, metric = "euclidean" ) )
#     
#     P = np.hstack( ( np.ones( ( n, 1 ) ), src ) )
#     
#     L = np.vstack( ( np.hstack( ( K, P ) ), np.hstack( ( P.T, np.zeros( ( 3, 3 ) ) ) ) ) )
#     
#     V = np.hstack( ( dst.T, np.zeros( ( 2, 3 ) ) ) )
#     
#     Wa = solve( L, V.T ) # Wa = np.dot( inv( L ), V.T )
#         
#     W = Wa[ :-3 , : ]
#     a = Wa[ -3: , : ]
#     
#     WK = np.dot( W.T, K )
#     WKW = np.dot( WK, W )
#     
#     be = 0.5 * np.trace( WKW )
#     
#     return {
#         'src':      src,
#         'dst':      dst,
#         'linear':   a,
#         'weights':  W,
#         'be':       be
#     }
#
################################################################################

cylst = []
pycylst = []
pylst = []

for n in trange( 5, 500, 5, leave = True, nested = False ):
    np.random.seed( 1337 )
    
    src = np.random.rand( n, 2 ) * 10
    dst = src + np.random.rand( n, 2 ) * 0.1
    
    tmp1 = []
    tmp2 = []
    tmp3 = []
    
    for r in trange( 0, 10, 1, nested = True ):
        with Capturing():
            with Timer( "Cy" ) as t1:
                gCy( src, dst )
            
            tmp1.append( t1.interval )
            
            with Timer( "gPyCy" ) as t2:
                gPyCy( src, dst )
            
            tmp2.append( t2.interval )
            
            with Timer( "Py" ) as t3:
                gPy( src, dst )
            
            tmp3.append( t3.interval )
    
    cylst.append( [ n, np.log10( np.mean( tmp1 ) ) ] )
    pycylst.append( [ n, np.log10( np.mean( tmp2 ) ) ] )
    pylst.append( [ n, np.log10( np.mean( tmp3 ) ) ] )


fig = plt.figure()
ax = fig.add_subplot( 111 )

ax.plot( *zip( *cylst ), color = 'g' )
ax.plot( *zip( *pycylst ), color = 'b' )
ax.plot( *zip( *pylst ), color = 'r' )

ax.legend( [ "cython", "python compiled", "python" ], loc = 'upper left' )
ax.set_xlabel( 'Number of minutiae' )
ax.set_ylabel( 'Log-time [log10( s )]' )

fig.savefig( 'language_implementation.png', dpi = 300 )

plt.show()
