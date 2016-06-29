#!/usr/bin/env python
#  *-* coding: cp850 *-*

from lib.TPS.TPSCy import generate as gCy
from lib.TPS.TPSCy import generate2 as gCy2
from lib.printCapturing import Capturing
from lib.profiler import Timer

from tqdm._tqdm import trange

import matplotlib.pyplot as plt
import numpy as np

################################################################################
#
#    Modifications in TPSCy file
#
#
# def generate2( 
#         double[ : , : ] src not None,
#         double[ : , : ] dst not None,
#     ):
#     
#     cdef int n = src.shape[ 0 ]
#     
#     # Memory allocation C-pointer-style
#     cdef double * W      = < double * > malloc( n * 2 * sizeof( double ) )
#     cdef double * linear = < double * > malloc( 3 * 2 * sizeof( double ) )
#     cdef double * be     = < double * > malloc( 1 *     sizeof( double ) )
#     
#     cdef int x = 0
#     for x in xrange( 100 ):
#         _generate(
#             src, dst,
#             W, linear, be
#         )
#     
#     ############################################################################
#     #    
#     #    Python object return
#     #    
#     ############################################################################
#     
#     return {
#         'src':      np.asarray( src ),
#         'dst':      np.asarray( dst ),
#         'linear':   np.array( < double [ :3, :2 ] > linear ),
#         'weights':  np.array( < double [ :n, :2 ] > W ),
#         'be':       be[0],
#     }
################################################################################

cylst = []
cy2lst = []

for n in trange( 5, 100, 1, leave = True, nested = False ):
    np.random.seed( 1337 )
    
    src = np.random.rand( n, 2 ) * 10
    dst = src + np.random.rand( n, 2 ) * 0.1
    
    tmp1 = []
    tmp2 = []
    
    for r in trange( 0, 10, 1, nested = True ):
        with Capturing():
            with Timer( "Cy" ) as t1:
                gCy( src, dst )
            
            tmp1.append( t1.interval )
            
            with Timer( "Cy2" ) as t2:
                gCy2( src, dst )
            
            tmp2.append( t2.interval )
    
    cylst.append( [ n, np.mean( tmp1 ) ] )
    cy2lst.append( [ n, np.mean( tmp2 ) / 100.0 ] )

gain = []
for x in xrange( len( cylst ) ):
    n, t1 = cylst[x]
    _, t2 = cy2lst[x]
     
    gain.append( [ n, ( 100 * ( t1 - t2 ) ) / 99 ] )

fig = plt.figure()
ax = fig.add_subplot( 111 )

ax.plot( *zip( *cylst ), color = 'r' )
ax.plot( *zip( *cy2lst ), color = 'g' )
ax.plot( *zip( *gain ), color = 'b' )

ax.legend( [ "1 overhead and 1 call", "( 1 overhead and 100 C-calls ) / 100", "overhead time" ], loc = 'upper left' )
ax.set_xlabel( 'Number of minutiae' )
ax.set_ylabel( 'Time [s]' )

fig.savefig( 'overhead.png', dpi = 300 )

plt.show()
