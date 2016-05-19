#!/usr/bin/env python
#  *-* coding: cp850 *-*

from __future__ import absolute_import

import ast
import os
from pprint import pprint
import random

from scipy import misc

import numpy as np

from . import TPSCy
from .config import CONF_gridSize, CONF_res, CONF_ncores, CONF_minx, \
    CONF_maxx, CONF_miny, CONF_maxy, CONF_dm, CONF_useWeights, CONF_weightsLimit, \
    CONF_nbRandom
    
################################################################################
#    
#    The aim of this wrapper is to simplify the use of the Cython library,
#    implementing a fast and efficient version of TPS distorsion proposed by
#    Bookstein (1989).
#    
#    All the exposed functions are available in the TPSCy.__init__ file. The
#    TPSCy module should not (but can) be called directly.
#    
#    All codes should be platform-independant. A recompilation of the TPSCy
#    module is however needed. The debugging process on UNIX show some tweeks to
#    do, but nothing dramatically impossible (some duplicated memory free'ing).
#    The actual Scipy UNIX release (0.16.1) has incompatibility with the BLAS
#    library; fix present on the github, and will be patched in the 0.17.0
#    release.
#    
#    Some dependencies are needed:
#    
#        BLAS / LAPACK
#        fortran
#        python
#        numpy
#        scipy
#        cython
#    
#    Because this library is designed to work on fingerprints, all functions are
#    designes to work with coordinates in milimeters, with the origin in the
#    lower left corner, as the ANSI/NIST 2007 standard. All functions, except
#    "TPS_Image", are adimentional (can work with mm, inch, yard, pixels,
#    ngstr”m, light-year, parsec, or, if you want, the "Double-decker bus" unit).
#    
#                                               Marco De Donno
#                                               University of Lausanne
#                                               
#    
#    References:                                                                
#        Bookstein, F. L. (1989). Principal warps: Thin-plate splines and the   
#        decomposition of deformations. IEEE Transactions on Pattern Analysis   
#        and Machine Intelligence, Vol. 11 (6), pp. 567-585                     
#    
################################################################################
################################################################################
#    
#    Wrapping function
#    
#        Functions to simplity the calls of the TPSCy module. No new
#        functionnality are added by those functions.
#    
################################################################################
def TPS_generate( *args, **kwags ):
    ############################################################################
    #    
    #    TPS parameter function
    #    
    #        Generation of the TPS parameters for two sets of points. This
    #        function can be called by specifing the 'src' and 'dst parameters,
    #        or using positionnal arguments ( the first one is the 'src', and
    #        the second one the 'dst' parameters
    #    
    #        Required:
    #            @param 'src': Source coordinates ( x, y )
    #            @type  'src': python list of tuples or list of lists or numpy array
    #            
    #            @param 'dst': Destination coordinates ( x, y )
    #            @type  'dst': python list of tuples or list of lists or numpy array
    #
    #        Return:    
    #            @return     : TPS parameters
    #            @type       : python dictionnary
    #    
    ############################################################################
    
    try:
        src, dst = args
    except:
        src = kwags.get( 'src' )
        dst = kwags.get( 'dst' )
    
    src = np.array( src )
    dst = np.array( dst )
    
    return TPSCy.generate( src, dst )

def TPS_project( *args, **kwargs ):
    ############################################################################
    #    
    #    Point projection
    #    
    #        Projection of the ( x, y ) point with the TPS function 'g' given in
    #        parameters. If a angle 'theta' is given, the projected angle is
    #        given in return.
    #        
    #        Required:
    #            @param 'g'     : TPS parameters
    #            @type  'g'     : python dictionary
    #                         
    #            @param 'x'     : x coordinate
    #            @type  'x'     : float
    #            
    #            @param 'y'     : y coordinate
    #            @type  'y'     : float
    #                  
    #        Optional:
    #            @param 'theta' : minutia angle
    #            @type  'theta' : float
    #        
    #        Return:
    #            @return        : Projected point ( x, y ) of ( x, y, theta )
    #            @return        : python tuple
    #        
    ############################################################################
    
    try:
        g, x, y, theta = args
    except:
        try:
            g, x, y = args
            theta = None
        except:
            g = kwargs.get( 'g' )
            
            x = kwargs.get( 'x' )
            y = kwargs.get( 'y' )
            theta = kwargs.get( 'theta', None )
    
    if theta == None:
        x, y, _ = TPSCy.project( g, x, y, 0 )
        return x, y
    
    else:
        x, y, theta = TPSCy.project( g, x, y, theta )
        return x, y, theta

################################################################################
#    
#    TPS specific tools
#    
################################################################################

def TPS_loadFromFile( f ):
    ############################################################################
    #    
    #    TPS parameter file loading
    #        
    #        Load TPS parameters from a file. The parameters have to be stored
    #        as a python dictionnary.
    #        
    #        Required:
    #            @param 'f' : URI of the file to load
    #            @type  'f' : string
    #                       
    #        Return: 
    #            @return    : TPS parameters
    #            @return    : pyton dictionary
    #        
    ############################################################################
        
    with open( f, "r" ) as fp:
        g = ast.literal_eval( fp.read() )
    
    return TPS_fromListToNumpy( g = g )

def TPS_fromListToNumpy( *args, **kwargs ):
    ############################################################################
    #    
    #    TPS casting
    #        
    #        Change type of the variable of a TPS parameter object from python
    #        list to numpy array.
    #        
    #        Required:
    #            @param 'g' : TPS parameters
    #            @type  'g' : python dictionary
    #                       
    #        Return: 
    #            @return    : TPS parameters
    #            @return    : pyton dictionary
    #        
    ############################################################################
    
    try:
        g = args[0]
    except:
        g = kwargs.get( 'g' )
    
    for var in [ 'src', 'dst', 'linear', 'weights' ]:
        g[ var ] = np.array( g[ var ] )
    
    return g

def TPS_recenter( *args, **kwargs ):
    # TODO: on ditait que c'est pas si simple pour faire un shift ... RecontrÃ´ler depuis les donnÃ©es brutes de TWIG2, et TOUT refaire (les .tps et la logique d'application des tps).
    ############################################################################
    #    
    #    TPS recentering
    #        
    #        Change the coordinate of the reference point in a TPS parameter
    #        dictionary. This function affect only the linear part, without any
    #        rotation of the TPS parameters.
    #        
    #        Required:
    #            @param 'g'  : TPS parameters
    #            @type  'g'  : python dictionary
    #                       
    #        Optional:
    #            @param 'cx' : x coordinate 
    #            @type  'cx' : float
    #            @def   'cx' : 0
    #        
    #            @param 'cy' : y coordinate 
    #            @type  'cy' : float
    #            @def   'cy' : 0
    #        
    #        Return: 
    #            @return     : TPS parameters
    #            @return     : python dictionary
    #        
    ############################################################################
    
    try:
        g, cx, cy = args
    except:
        g = kwargs.get( "g" )
        cx = kwargs.get( "cx", 0 )
        cy = kwargs.get( "cy", 0 )
    
    g[ 'src' ] += np.array( [ [ cx, cy ] ] )
    g[ 'dst' ] += np.array( [ [ cx, cy ] ] )
    
    return g

def TPS_shift( *args, **kwargs ):
    ############################################################################
    #    
    #    TPS recentering
    #        
    #        Change the coordinate of the reference point in a TPS parameter
    #        dictionary. This function affect only the linear part, without any
    #        rotation of the TPS parameters.
    #        
    #        Required:
    #            @param 'g'  : TPS parameters
    #            @type  'g'  : python dictionary
    #                       
    #        Optional:
    #            @param 'dx' : dx shift 
    #            @type  'dx' : float
    #            @def   'dx' : 0
    #        
    #            @param 'dy' : dy shift 
    #            @type  'dy' : float
    #            @def   'dy' : 0
    #        
    #        Return: 
    #            @return     : TPS parameters
    #            @return     : python dictionary
    #        
    ############################################################################
        
    try:
        g, cx, cy = args
    except:
        g = kwargs.get( "g" )
        cx = kwargs.get( "cx", 0 )
        cy = kwargs.get( "cy", 0 )
    
    g[ 'linear' ] += np.array( [ [ cx, cy ], [ 0, 0 ], [ 0, 0 ] ] )
    
    return g

def TPS_rotate( **kwargs ):
    ############################################################################
    #    
    #    TPS rotation
    #        
    #        Change the rotation parameter in a TPS parameter dictionary. This
    #        function affect only the linear part. The angle imply the rotation
    #        of the output of the projection function 'g'.
    #        
    #        The angle of rotation is given in degree, anti-clockwise, with the
    #        zero on the right (like ANSI/NIST 2007).
    #        
    #        Required:
    #            @param 'g'     : TPS parameters
    #            @type  'g'     : python dictionary
    #                       
    #            @param 'theta' : angle of rotation
    #            @type  'theta' : float
    #        
    #        Return:           
    #            @return        : TPS parameters
    #            @return        : python dictionary
    #        
    ############################################################################
    
    g = kwargs.get( "g" )
    theta = kwargs.get( "theta", 0 )
    
    theta = -theta
    theta = theta / 180.0 * np.pi
    
    c, s = np.cos( theta ), np.sin( theta )
    rotmat = np.array( [ [ c, -s ], [ s, c ] ] )
    
    rot = g[ 'linear' ][ 1:, : ]
    rotbis = np.dot( rot, rotmat )
    g[ 'linear' ] = np.vstack( ( g[ 'linear' ][ 0, : ], rotbis ) )
    
    return g

def TPS_Image( **kwargs ):
    ############################################################################
    #    
    #    TPS distorsion on a image
    #        
    #        Application of the 'g' function on an image. This function applies
    #        the reverting-methodology: an estimation of the reverse function
    #        (backward projection) is estimated using a grid, which is projected
    #        by the 'g' function. This projected grid is then used as source,
    #        and the coordinates on the original image as destination. The
    #        TPSCy.image can not be called with a forward 'g' TPS function!
    #        
    #        Required:
    #            @param 'infile' : URI to the input file
    #            @type  'infile' : python string
    #            @rem   'infile' : Exclusive with and prioritary over 'inimg'
    #                            
    #            @param 'inimg' : numpy array with the input image
    #            @type  'inimg' : numpy.array
    #            @rem   'inimg' : Exclusive with 'infile'
    #                           
    #            @param 'gfile' : URI to the TPS parameters file
    #            @type  'gfile' : python string
    #            @rem   'gfile' : Exclusive with and prioritary over 'g'
    #            
    #            @param 'g' : TPS parameters
    #            @type  'g' : python dictionary
    #            @rem   'g' : Exclusive with 'g'
    #                       
    #        Optional:
    #            @param 'res' : Resolution of the input and output image
    #            @type  'res' : float
    #            @def   'res' : 500
    #                       
    #            @param 'outfile' : URI to the output image file
    #            @type  'outfile' : string
    #            @rem   'outfile' : Exclusive with 'g'
    #            
    #            @param 'reverseFullGrid' : Use the full-grid to revert the 'g' function (SLOW !)
    #            @type  'reverseFullGrid' : boolean
    #            @def   'reverseFullGrid' : False
    #            
    #            @param 'useWeights' : Use the weights to selects the optimal grid to revert the 'g' function
    #            @type  'useWeights' : boolean
    #            @def   'useweights' : True
    #                       
    #            @param 'gridSize' : Size of the grid to calclate the revert function
    #            @type  'gridSize' : float
    #            @def   'gridSize' : 0.75
    #                       
    #            @param 'cx' : x coordinate of the center
    #            @type  'cx' : float
    #            @def   'cx' : 0
    #                       
    #            @param 'cy' : y coordinate of the center
    #            @type  'cy' : float
    #            @def   'cy' : 0
    #                       
    #            @param 'ncores' : Number of cores used to do the projection of the input image.
    #            @type  'ncores' : int
    #            @def   'ncores' : 8
    #                       
    #        Return: 
    #            @return    : Distorted image or successuly-written image
    #            @return    : numpy.array or bool
    #        
    ############################################################################
        
    infile = kwargs.pop( "infile", None )
    inimg = kwargs.pop( "inimg", None )
    
    gfile = kwargs.pop( "gfile", None )
    g = kwargs.pop( "g", None )
    
    reverseFullGrid = kwargs.pop( "reverseFullGrid", False )
    
    gridSize = kwargs.pop( "gridSize", CONF_gridSize )
    
    cx = kwargs.pop( "cx", 0 )
    cy = kwargs.pop( "cy", 0 )
    res = kwargs.pop( "res", CONF_res )
    
    outfile = kwargs.pop( "outfile", None )
    
    ncores = kwargs.pop( "ncores", CONF_ncores )
    
    ############################################################################
    #    Image loading and conversion
    #        Loading the image in 'image coordinate' (top left) and flip in
    #        'ANSI/NIST 2007 coordinates' (bottom left)
    ############################################################################
    
    if infile != None:
        indata = misc.imread( infile ).astype( np.int )
    elif inimg != None:
        indata = np.asarray( inimg )
    else:
        raise Exception( "No input data (infile or image object)" )
    
    indata = np.flipud( indata )
    indata = indata.T
    
    ############################################################################
    #    Distorsion parameters
    #        The g file must be calculated in milimeters and not in pixels !
    #        The center is in "NIST coordinates" (mm from the bottom left)
    ############################################################################
    
    if gfile != None:
        g = TPS_loadFromFile( gfile )
    elif g != None:
        pass
    else:
        raise Exception( "No TPS parameters of file" )
    
    if cx != 0 and cy != 0:
        g = TPS_recenter( g = g, cx = cx, cy = cy )
    
    maxx, maxy = indata.shape
    maxx = maxx / float( res ) * 25.4
    maxy = maxy / float( res ) * 25.4
    
    ############################################################################
    #    Range calculation
    #        Because the borders could be in negative coordinate
    ############################################################################
    
    r = TPSCy.r( g, 0, maxx, 0, maxy )
    
    size = [ 
        int( float( res ) / 25.4 * ( r[ 'maxx' ] - r[ 'minx' ] ) ),
        int( float( res ) / 25.4 * ( r[ 'maxy' ] - r[ 'miny' ] ) )
    ]
    
    outimg = np.ones( size, dtype = np.intc )
    
    ############################################################################
    #    Calculation of the "inverse" function (approximation)
    #        The real inverse function of g is not calculable (the g function
    #        is not bijective). My approximation here is to project a grid of
    #        size gridSize with the g, and use this projected grid as source for
    #        the g2 projection function. This function is used by the
    #        TPSCy.image function.
    ############################################################################
    
    if reverseFullGrid:
        g2 = TPSCy.revert( g, 0, maxx, 0, maxy, gridSize )
    else:
        g2 = TPS_revertDownSampling( 
            g = g,
            
            minx = 0,
            maxx = maxx,
            miny = 0,
            maxy = maxy,
            
            gridSize = gridSize,
            
            **kwargs
        )
    
    ############################################################################
    #    Distorsion of the image
    #        Reverse calculation of the distorted image. The g2 function allow
    #        the TPSCy.image function to calculate the x, y coordinates on the
    #        original image. This methodoloy allow to have a continuous image.
    ############################################################################
    
    TPSCy.image( indata, g2, r, float( res ), outimg, ncores )
    
    ############################################################################
    #    Preparation of the output image (file or image object return)
    ############################################################################
    
    outimg = np.flipud( outimg.T )
    outimg = misc.toimage( outimg, cmin = 0, cmax = 255 )
    
    if outfile != None:
        outimg.save( outfile, dpi = ( int( res ), int( res ) ) )
        return os.path.isfile( outfile )
    else:
        return outimg

def TPS_Grid( **kwargs ):
    ############################################################################
    #    
    #    Distorsion grid
    #        
    #        Creation of a distorsion grid. This gris is created using the
    #        forward 'g' TPS parameters.
    #        
    #        Required:
    #            @param 'g' : TPS parameters
    #            @type  'g' : python dictionary
    #
    #        Optional:
    #            @param 'minx' : minimum x coordinate
    #            @type  'minx' : float
    #            @def   'minx' : 0
    #
    #            @param 'maxx' : maximum x coordinate
    #            @type  'maxx' : float
    #            @def   'maxx' : 25.4
    #
    #            @param 'miny' : minimum y coordinate
    #            @type  'miny' : float
    #            @def   'miny' : 0
    #
    #            @param 'maxy' : maximum y coordinate
    #            @type  'maxy' : float
    #            @def   'maxy' : 25.4
    #
    #            @param 'outfile' : URI to the output image. If None, the function will return the numpy.array of the image
    #            @type  'outfile' : string
    #            @def   'outfile' : None
    #
    #            @param 'res' : Resolution of the output image
    #            @type  'res' : float
    #            @def   'res' : 500 [dpi]
    #
    #            @param 'dm' : Border of the image added around the grid
    #            @type  'dm' : float
    #            @def   'dm' : 5 [px]
    #        
    #        Return: 
    #            @return    : Image or sucessfully writed image
    #            @return    : numpy.array or boolean
    #        
    ############################################################################
        
    
    ############################################################################
    #    Distorsion grid - Cython wrapper
    #        Produce the distorsion grid in a specific location given by the
    #        ( minx, maxx, miny maxy ) arguments
    ############################################################################
    
    g = kwargs.get( "g" )
    
    minx = kwargs.get( "minx", CONF_minx )
    maxx = kwargs.get( "maxx", CONF_maxx )
    miny = kwargs.get( "miny", CONF_miny )
    maxy = kwargs.get( "maxy", CONF_maxy )
    
    outfile = kwargs.get( "outfile", None )
    
    res = kwargs.get( "res", CONF_res )
    dm = kwargs.get( "dm", CONF_dm )
    
    ############################################################################
    #    Cython calls
    ############################################################################
    
    outimg = TPSCy.grid( g, minx, maxx, miny, maxy, res = res, dm = dm )
    
    outimg = misc.toimage( outimg, cmin = 0, cmax = 255 )
    
    ############################################################################
    #    Image writting on disk or return as numpy.array
    ############################################################################
    
    if outfile != None:
        outimg.save( outfile, dpi = ( res, res ) )
    else:
        return outimg

def TPS_range( **kwargs ):
    ############################################################################
    #    
    #    Range calculation
    #        
    #        Calculation of the range in x and y on the projected space. This
    #        function have to be called to be able to ensure that all borders of
    #        the projected image will be in the output image. The border could
    #        be bigger than the input image.
    #        
    #        Required:
    #            @param 'g' : TPS parameters
    #            @type  'g' : python dictionary
    #                       
    #        Optional:
    #            @param 'minx' : minimum x coordinate
    #            @type  'minx' : float
    #            @def   'minx' : 0
    #
    #            @param 'maxx' : maximum x coordinate
    #            @type  'maxx' : float
    #            @def   'maxx' : 25.4
    #
    #            @param 'miny' : minimum y coordinate
    #            @type  'miny' : float
    #            @def   'miny' : 0
    #
    #            @param 'maxy' : maximum y coordinate
    #            @type  'maxy' : float
    #            @def   'maxy' : 25.4
    #        
    #        Return: 
    #            @return    : Dictionnary with minx, maxx, miny and maxy
    #            @return    : python dict
    #        
    ############################################################################
        
    g = kwargs.get( "g" )
    
    minx = kwargs.get( "minx", CONF_minx )
    maxx = kwargs.get( "maxx", CONF_maxx )
    miny = kwargs.get( "miny", CONF_miny )
    maxy = kwargs.get( "maxy", CONF_maxy )
    
    return TPSCy.r( g, minx, maxx, miny, maxy )

def TPS_revertGrid( **kwargs ):
    ############################################################################
    #    
    #    TPS parameters reverting with a projected grid
    #        
    #        This function will calculate an approximation of the reverse 'g'
    #        projection function with a grid of size 'gridSize'.
    #        
    #        Required:
    #            @param 'g' : TPS parameters
    #            @type  'g' : python dictionary
    #                       
    #        Optional:
    #            @param 'minx' : minimum x coordinate
    #            @type  'minx' : float
    #            @def   'minx' : 0
    #
    #            @param 'maxx' : maximum x coordinate
    #            @type  'maxx' : float
    #            @def   'maxx' : 25.4
    #
    #            @param 'miny' : minimum y coordinate
    #            @type  'miny' : float
    #            @def   'miny' : 0
    #
    #            @param 'maxy' : maximum y coordinate
    #            @type  'maxy' : float
    #            @def   'maxy' : 25.4
    #        
    #            @param 'gridSize' : Size of the grid
    #            @type  'gridSize' : float
    #            @def   'gridSize' : 0.75
    #        
    #        Return: 
    #            @return    : reverted TPS parameters 
    #            @return    : python dictionary
    #        
    ############################################################################
    
    g = kwargs.get( "g" )
    
    minx = kwargs.get( "minx", CONF_minx )
    maxx = kwargs.get( "maxx", CONF_maxx )
    miny = kwargs.get( "miny", CONF_miny )
    maxy = kwargs.get( "maxy", CONF_maxy )

    gridSize = kwargs.get( "gridSize", CONF_gridSize )
    
    return TPSCy.revert( g, minx, maxx, miny, maxy, gridSize )

def TPS_revertDownSampling( **kwargs ):
    ############################################################################
    #    
    #    TPS parameters reverting with a projected grid
    #        
    #        This function will calculate an approximation of the reverse 'g'
    #        projection function with a grid of size 'gridSize'.
    #        
    #        Required:
    #            @param 'g' : TPS parameters
    #            @type  'g' : python dictionary
    #                       
    #        Optional:
    #            @param 'minx' : minimum x coordinate
    #            @type  'minx' : float
    #            @def   'minx' : 0
    #
    #            @param 'maxx' : maximum x coordinate
    #            @type  'maxx' : float
    #            @def   'maxx' : 25.4
    #
    #            @param 'miny' : minimum y coordinate
    #            @type  'miny' : float
    #            @def   'miny' : 0
    #
    #            @param 'maxy' : maximum y coordinate
    #            @type  'maxy' : float
    #            @def   'maxy' : 25.4
    #        
    #            @param 'gridSize' : Size of the grid
    #            @type  'gridSize' : float
    #            @def   'gridSize' : 0.75
    #        
    #            @param 'useWeights' : Use the weights of the TPS parameters to select only the important points
    #            @type  'useWeights' : boolean
    #            @def   'useWeights' : True
    #        
    #            @param 'weightslimit' : Minimum weight to select an important point
    #            @type  'weightslimit' : float
    #            @def   'weightslimit' : 0.003
    #        
    #            @param 'nbrandom' : Number of random points added on the grid
    #            @type  'nbrandom' : int
    #            @def   'nbrandom' : 10
    #        
    #        Return: 
    #            @return    : reverted TPS parameters 
    #            @return    : python dictionary
    #        
    ############################################################################
    
    g = kwargs.get( "g" )
    
    minx = kwargs.get( "minx", CONF_minx )
    maxx = kwargs.get( "maxx", CONF_maxx )
    miny = kwargs.get( "miny", CONF_miny )
    maxy = kwargs.get( "maxy", CONF_maxy )

    gridSize = kwargs.get( "gridSize", CONF_gridSize )
    
    useWeights = kwargs.get( "useWeights", CONF_useWeights )
    weightslimit = kwargs.get( "weightslimit", CONF_weightsLimit )
    
    nbrandom = kwargs.get( "nbrandom", CONF_nbRandom )
    
    ############################################################################
    #    Initialisation of the new source and destination points lists
    ############################################################################
    
    srcbis = []
    dstbis = []
    
    ############################################################################
    #    Add the originals points in the new set
    ############################################################################
    
    for src, dst in zip( g['src'], g['dst'] ):
        x, y = src
        xp, yp = TPS_project( x = x, y = y, g = g )
        dstbis.append( ( x, y ) )
        srcbis.append( ( xp, yp ) )
    
    ############################################################################
    #    Add the corners and the centers of the borders
    ############################################################################
    
    for x in ( minx, 0.5 * ( minx + maxx ), maxx ):
        for y in ( miny, 0.5 * ( miny + maxy ), maxy ):
            xp, yp = TPS_project( x = x, y = y, g = g )
            dstbis.append( ( x, y ) )
            srcbis.append( ( xp, yp ) )
            
    ############################################################################
    #    Selection of the important points based on the weight
    ############################################################################
    
    if useWeights:
        gr = TPS_revertGrid( 
            g = g,
            
            minx = minx,
            maxx = maxx,
            miny = miny,
            maxy = maxy,
            
            gridSize = gridSize
        )
        
        for src, dst, w in zip( gr['src'], gr['dst'], gr['weights'] ):
            wx, wy = np.absolute( w )
            if wx >= weightslimit or wy >= weightslimit:
                srcbis.append( src )
                dstbis.append( dst )
            
    ############################################################################
    #    Add random points
    ############################################################################
    
    np.random.seed( 1337 )
    
    for _ in xrange( nbrandom ):
        x = minx + random.random() * ( maxx - minx )
        y = miny + random.random() * ( maxy - miny )
         
        xp, yp = TPS_project( x = x, y = y, g = g )
         
        dstbis.append( ( x, y ) )
        srcbis.append( ( xp, yp ) )
    
    ############################################################################
    #    Return the generated TPS parameters
    ############################################################################
    
    return TPS_generate( src = srcbis, dst = dstbis )
