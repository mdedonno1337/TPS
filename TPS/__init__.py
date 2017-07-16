#!/usr/bin/env python
#  *-* coding: utf-8 *-*

from __future__ import absolute_import

import ast
import numpy as np
import os
import random
import sys

from copy import deepcopy
from PIL import ImageDraw, ImageFont, Image
from scipy import misc

from .config import *
from .functions import deprecated

try:
    from . import TPSCy as TPSModule

except:
    from . import TPSpy as TPSModule
    
################################################################################
#    
#    The aim of this wrapper is to simplify the use of the Cython library,
#    implementing a fast and efficient version of TPS distortion proposed by
#    Bookstein (1989).
#    
#    All the exposed functions are available in the TPSCy.__init__ file. The
#    TPSCy module should not (but can) be called directly.
#    
#    All codes should be platform-independent. A re-compilation of the TPSCy
#    module is, however, needed.
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
#    designed to work with coordinates in millimetres, with the origin on the
#    lower left corner, as the ANSI/NIST 2007 standard. All functions, except
#    "TPS_Image", are adimensional (can work with mm, inch, yard, pixels, light-
#    year, parsec, or, if you want, the "Double-decker bus" unit).
#    
#                                                Marco De Donno
#                                                University of Lausanne
#                                                
#                                                Marco.DeDonno@unil.ch
#                                                mdedonno1337@gmail.com
#    
#    
#    References:
#
#        Bookstein, F. L. (1989). Principal warps: Thin-plate splines and the   
#        decomposition of deformations. IEEE Transactions on Pattern Analysis   
#        and Machine Intelligence, Vol. 11 (6), pp. 567-585
#        
#        NIST. (2007). American National Standard for Information Systems – Data
#        Format for the Interchange of Fingerprint Facial, and Other Biometric
#        Information – Part 1 (NIST Special Publication 500-271)
#    
################################################################################

def TPS_module():
    return TPSModule.lang()

################################################################################
#    
#    Wrapping function
#    
#        Functions to simplity the calls of the TPSCy module. No new
#        functionnality are added by those functions.
#    
################################################################################

def TPS_generate( *args, **kwags ):
    """
        Generation of the TPS parameters for two sets of points. This
        function can be called by specifing the 'src' and 'dst parameters,
        or using positionnal arguments ( the first one is the 'src', and
        the second one the 'dst' parameters
    
        Required:
        
        :param src: Source coordinates ( x, y )
        :type src: python list of tuples or list of lists or numpy array
        
        :param dst: Destination coordinates ( x, y )
        :type dst: python list of tuples or list of lists or numpy array

        :return: TPS parameters
        :rtype: python dictionnary
        
        Usage:
            
            >>> from TPS import TPS_generate
            >>> src = [ [ 3.6929, 10.3819 ], [ 6.5827, 8.8386 ], [ 6.7756, 12.0866 ], [ 4.8189, 11.2047 ], [ 5.6969, 10.0748 ] ]
            >>> dst = [ [ 3.9724,  6.5354 ], [ 6.6969, 4.1181 ], [ 6.5394,  7.2362 ], [ 5.4016,  6.4528 ], [ 5.7756,  5.1142 ] ]

            >>> g = TPS_generate( src, dst )
            >>> g # doctest: +NORMALIZE_WHITESPACE
            {'src': array([[  3.6929,  10.3819],
               [  6.5827,   8.8386],
               [  6.7756,  12.0866],
               [  4.8189,  11.2047],
               [  5.6969,  10.0748]]), 'scale': 0.8931102258056604, 'linear': array([[ 1.35499582, -2.94596308],
               [ 0.87472587, -0.29556056],
               [-0.02886041,  0.92163259]]), 'be': 0.0429994895805986, 'dst': array([[ 3.9724,  6.5354],
               [ 6.6969,  4.1181],
               [ 6.5394,  7.2362],
               [ 5.4016,  6.4528],
               [ 5.7756,  5.1142]]), 'weights': array([[-0.03803014,  0.04244693],
               [ 0.02318775,  0.01591661],
               [-0.02475506,  0.02881348],
               [ 0.07978226, -0.04542552],
               [-0.04018482, -0.0417515 ]]), 'shearing': 250.32963702546}
        
        The independents variables can be accessed by the name:
        
            >>> g[ 'weights' ] # doctest: +NORMALIZE_WHITESPACE
            array([[-0.03803014,  0.04244693],
               [ 0.02318775,  0.01591661],
               [-0.02475506,  0.02881348],
               [ 0.07978226, -0.04542552],
               [-0.04018482, -0.0417515 ]])
            
            >>> g[ 'linear' ] # doctest: +NORMALIZE_WHITESPACE
            array([[ 1.35499582, -2.94596308],
               [ 0.87472587, -0.29556056],
               [-0.02886041,  0.92163259]])
            
            >>> g[ 'be' ]
            0.0429994895805986
        
        Two variables are defined in this library and not in the Bookstein article.
        The 'scale' factor represent the scaling factor of the linear part. 
        
            >>> g[ 'scale' ]
            0.8931102258056604
        
        The 'shearing' factor is the angle of the linear distortion.
        
            >>> g[ 'shearing' ]
            250.32963702546
            
    """
    try:
        src, dst = args
    except:
        src = kwags.get( 'src' )
        dst = kwags.get( 'dst' )
    
    src = np.array( src, dtype = np.double )
    dst = np.array( dst, dtype = np.double )
    
    return TPSModule.generate( src, dst )

def TPS_project( *args, **kwargs ):
    """  
        Projection of the ( x, y ) point with the TPS function 'g' given in
        parameters. If a angle 'theta' is given, the projected angle is
        given in return.
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
                     
        :param x: x coordinate
        :type x: float
        
        :param y: y coordinate
        :type y: float
                  
        Optional:
        
        :param theta: minutia angle
        :type theta: float
        
        Return:
        :return: Projected point ( x, y ) of ( x, y, theta )
        :rtype: python tuple
        
        Usage:
        
            >>> from TPS import TPS_project
            >>> TPS_project( g = g, x = 3.6929, y = 10.3819 )
            (3.9724000000000004, 6.535400000000002)
            
            >>> TPS_project( g, 3.6929, 10.3819 )
            (3.9724000000000004, 6.535400000000002)

        The angle at coordinate `(x,y)` can be also provided:
        
            >>> TPS_project( g = g, x = 3.6929, y = 10.3819, theta = 120 )
            (3.9724000000000004, 6.535400000000002, 107.4300206381953)
            
            >>> TPS_project( g, 3.6929, 10.3819, 120 )
            (3.9724000000000004, 6.535400000000002, 107.4300206381953)
    """
    
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
        x, y, _ = TPSModule.project( g = g, x = x, y = y, theta = 0 )
        return x, y
    
    else:
        x, y, theta = TPSModule.project( g = g, x = x, y = y, theta = theta )
        return x, y, theta

def TPS_project_list( *args, **kwargs ):
    """
        Projection of a list of ( x, y ) points with the TPS function 'g'
        passed in parameters.
     
        Required:
    
        :param g: TPS parameters
        :type g: python dictionary
        
        :param lst: List of points to distort
        :type lst: python list
            
        Return:
        
        :return: List of distorted points
        :rtype: python list
        
        Usage:
        
            >>> from TPS import TPS_project_list
            >>> p = TPS_project_list( g = g, lst = src )
            >>> p
            [(3.9724000000000004, 6.535400000000002), (6.6969, 4.1181), (6.5394, 7.2362), (5.4016, 6.452800000000001), (5.775600000000002, 5.114200000000001)]
        
        Because of the way the numbers are stored in a computer, the projected
        source points are not exactly the same as the destination points, as
        they should. The numpy `allclose` function allow to check with a small
        error accepted:
        
            >>> import numpy as np
            >>> np.allclose( dst, p )
            True
    """
    try:
        g, lst = args
    except:
        g = kwargs.get( "g" )
        lst = kwargs.get( "lst" )
    
    return [ TPS_project( g, *d ) for d in lst ]
    
################################################################################
#    
#    TPS specific tools
#    
################################################################################

def TPS_loadFromFile( f ):
    """
        Load TPS parameters from a file. The parameters have to be stored
        as a python dictionnary.
        
        Required:
    
        :param f: URI of the file to load
        :type f: string
                       
        Return:
    
        :return: TPS parameters
        :rtype: pyton dictionary
    """
    with open( f, "r" ) as fp:
        data = fp.read()
        data = data.replace( "array([[", "[[" )
        data = data.replace( "]])", "]]" )
        g = ast.literal_eval( data )
    
    return TPS_fromListToNumpy( g = g )

def TPS_fromListToNumpy( *args, **kwargs ):
    """
        Change type of the variable of a TPS parameter object from python
        list to numpy array.
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
        
        Return:
        
        :return: TPS parameters
        :rtype: pyton dictionary
    """
    try:
        g = args[0]
    except:
        g = kwargs.get( 'g' )
    
    for var in [ 'src', 'dst', 'linear', 'weights' ]:
        g[ var ] = np.array( g[ var ], dtype = np.double )
    
    return g

def TPS_recenter( *args, **kwargs ):
    """"
        Change the coordinate of the reference point in a TPS parameter
        dictionary. This function affect only the linear part, without any
        rotation of the TPS parameters.
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
                       
        Optional:
        
        :param cx: x coordinate 
        :type cx: float
        
        :param cy: y coordinate 
        :type cy: float
        
        Return: 
        
        :return: TPS parameters
        :rtype: python dictionary
        
        Usage:
        
            >>> from TPS import TPS_recenter
            >>> TPS_recenter( g, -5, -9 ) # doctest: +NORMALIZE_WHITESPACE
            {'src': array([[-1.3071,  1.3819],
               [ 1.5827, -0.1614],
               [ 1.7756,  3.0866],
               [-0.1811,  2.2047],
               [ 0.6969,  1.0748]]), 'scale': 0.8931102258056604, 'linear': array([[ 1.35499582, -2.94596308],
               [ 0.87472587, -0.29556056],
               [-0.02886041,  0.92163259]]), 'be': 0.0429994895805986, 'dst': array([[-1.0276, -2.4646],
               [ 1.6969, -4.8819],
               [ 1.5394, -1.7638],
               [ 0.4016, -2.5472],
               [ 0.7756, -3.8858]]), 'weights': array([[-0.03803014,  0.04244693],
               [ 0.02318775,  0.01591661],
               [-0.02475506,  0.02881348],
               [ 0.07978226, -0.04542552],
               [-0.04018482, -0.0417515 ]]), 'shearing': 250.32963702546}
            
    """
    try:
        g, cx, cy = args
    except:
        g = kwargs.get( "g" )
        cx = kwargs.get( "cx", 0 )
        cy = kwargs.get( "cy", 0 )
    
    g2 = deepcopy( g )
    
    g2[ 'src' ] += np.array( [ [ cx, cy ] ], dtype = np.double )
    g2[ 'dst' ] += np.array( [ [ cx, cy ] ], dtype = np.double )
    
    return g2

def TPS_shift( *args, **kwargs ):
    """
        Change the coordinate of the reference point in a TPS parameter
        dictionary. This function affect only the linear part, without any
        rotation of the TPS parameters.
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
        
        Optional:
        :param dx: dx shift 
        :type dx: float
        
        :param dy: dy shift 
        :type dy: float
        
        Return: 
        
        :return: TPS parameters
        :rtype: python dictionary
        
        Usage:
        
            >>> from TPS import TPS_shift, TPS_project
            >>> g2 = TPS_shift( g, 10, 10 )
            >>> TPS_project( g2, 3.6929, 10.3819 )
            (13.9724, 16.535400000000003)
    """
    try:
        g, cx, cy = args
    except:
        g = kwargs.get( "g" )
        cx = kwargs.get( "cx", 0 )
        cy = kwargs.get( "cy", 0 )
    
    g2 = deepcopy( g )
    g2[ 'linear' ] += np.array( [ [ cx, cy ], [ 0, 0 ], [ 0, 0 ] ], dtype = np.double )
    
    return g2

def TPS_rotate( **kwargs ):
    """
        Change the rotation parameter in a TPS parameter dictionary. This
        function affect only the linear part. The angle imply the rotation
        of the output of the projection function 'g'.
        
        The angle of rotation is given in degree, anti-clockwise, with the
        zero on the right (like ANSI/NIST 2007).
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
        
        :param theta: angle of rotation
        :type theta: float
        
        Return:    
               
        :return: TPS parameters
        :rtype: python dictionary
    """
    g = kwargs.get( "g" )
    g2 = deepcopy( g )
    
    theta = kwargs.get( "theta", 0 )
    
    theta = -theta
    theta = theta / 180.0 * np.pi
    
    c, s = np.cos( theta ), np.sin( theta )
    rotmat = np.array( [ [ c, -s ], [ s, c ] ], dtype = np.double )
    
    rot = g2[ 'linear' ][ 1:, : ]
    rotbis = np.dot( rot, rotmat )
    g2[ 'linear' ] = np.vstack( ( g2[ 'linear' ][ 0, : ], rotbis ) )
    
    return g2

def TPS_image( **kwargs ):
    """
        Application of the 'g' function on an image. This function applies
        the reverting-methodology: an estimation of the reverse function
        (backward projection) is estimated using a grid, which is projected
        by the 'g' function. This projected grid is then used as source,
        and the coordinates on the original image as destination. The
        TPSCy.image can not be called with a forward 'g' TPS function!
        
        Required:
        
        :param infile: URI to the input file
        :type infile: python string
        
        :param inimg: numpy array with the input image
        :type inimg: numpy.array
        
        .. note::
            `infile` and `inimg` are mutially exclusive. `inimg` has prioritary
        
        :param gfile: URI to the TPS parameters file
        :type gfile: python string
        
        :param g: TPS parameters
        :type g: python dictionary
        
        .. note::
            `gile` and `g` are mutially exclusive. `gfile` has prioritary
        
        Optional:
        
        :param res: Resolution of the input and output image
        :type res: float
        
        :param outfile: URI to the output image file
        :type outfile: string
        
        :param reverseFullGrid: Use the full-grid to revert the 'g' function (SLOW !)
        :type reverseFullGrid: boolean
        
        :param useWeights: Use the weights to selects the optimal grid to revert the 'g' function
        :type useWeights: boolean
        
        :param gridSize: Size of the grid to caculate the revert function
        :type gridSize: float
        
        :param cx: x coordinate of the center
        :type cx: float
        
        :param cy: y coordinate of the center
        :type cy: float
        
        :param ncores: Number of cores used to do the projection of the input image.
        :type ncores: int
        
        Return:
        
        :return: Distorted image or successuly-written image
        :rtype: numpy.array or bool
    """
    if TPSModule.lang() == "Python":
        raise NotImplementedError

    else:
        infile = kwargs.pop( "infile", None )
        inimg = kwargs.pop( "inimg", None )
        
        gfile = kwargs.pop( "gfile", None )
        g = kwargs.pop( "g", None )
        g2 = deepcopy( g )
        
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
            g2 = TPS_loadFromFile( gfile )
        elif g2 != None:
            pass
        else:
            raise Exception( "No TPS parameters of file" )
        
        if cx != 0 and cy != 0:
            g2 = TPS_recenter( g = g2, cx = cx, cy = cy )
        
        maxx, maxy = indata.shape
        maxx = maxx / float( res ) * 25.4
        maxy = maxy / float( res ) * 25.4
        
        ############################################################################
        #    Range calculation
        #        Because the borders could be in negative coordinate
        ############################################################################
        
        r = kwargs.get( 'r', None )
        
        if r == None:
            r = TPSModule.r( g2, 0, int( maxx ), 0, int( maxy ) )
        
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
            g2 = TPSModule.revert( g2, 0, maxx, 0, maxy, gridSize )
        else:
            g2 = TPS_revertDownSampling( 
                g = g2,
                
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
        
        TPSModule.image( indata, g2, r, float( res ), outimg, ncores )
        
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

def TPS_grid( **kwargs ):
    """
        Distorsion grid
            
        Creation of a distorsion grid. This gris is created using the
        forward 'g' TPS parameters.
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
        
        Optional:
        
        :param minx: minimum x coordinate
        :type minx: float
            
        :param maxx: maximum x coordinate
        :type maxx: float

        :param miny: minimum y coordinate
        :type miny: float

        :param maxy: maximum y coordinate
        :type maxy: float

        :param outfile: URI to the output image. If None, the function will return the numpy.array of the image
        :type outfile: string

        :param res: Resolution of the output image
        :type res: float

        :param dm: Border of the image added around the grid
        :type dm: float
        
        Return:
        
        :return: Image or sucessfully writed image
        :rtype: numpy.array or boolean
        
        Usage:
        
            >>> from TPS import TPS_grid
            >>> grid = TPS_grid( g = g, minx = 3.8, maxx = 8.6, miny = 8, maxy = 12.5, res = 2500, major_step = 0.1, minor_step = 0.01 )
            >>> grid # doctest: +ELLIPSIS
            <PIL.Image.Image image mode=RGB size=455x525 at ...>
            
        To check that the content of the image is the correct one, the MD5 hash
        of the string representation of the image can be check as follow:
        
            >>> from hashlib import md5
            >>> print md5( grid.tobytes() ).hexdigest()
            2dd43ca2594a3c4f89c991e56cf1949e
    """
    g = kwargs.get( "g" )
    
    minx = kwargs.get( "minx", CONF_minx )
    maxx = kwargs.get( "maxx", CONF_maxx )
    miny = kwargs.get( "miny", CONF_miny )
    maxy = kwargs.get( "maxy", CONF_maxy )
    
    plotpoints = kwargs.get( "plotpoints", True )
    
    outfile = kwargs.get( "outfile", None )
    
    res = kwargs.get( "res", CONF_res )
    dm = kwargs.get( "dm", CONF_dm )
    minor_step = kwargs.get( "minor_step", CONF_minorstep )
    major_step = kwargs.get( "major_step", CONF_majorstep )
    
    limit = kwargs.get( "limit", False )
    if limit != False and g[ 'be' ] > float( limit ):
        outimg = Image.new( "RGB", ( res, res ), ( 240, 240, 240 ) )
        draw = ImageDraw.Draw( outimg )
        font = ImageFont.truetype( "arial.ttf", 16 )
        draw.text( ( 50, 50 ), "No distorsion grid available:\nBending Energy to high.\nPlease check the pairing.", ( 0, 0, 0 ), font = font )
        
    else:
        params = {
            "g": g,
            "minx": minx,
            "maxx": maxx,
            "miny": miny,
            "maxy": maxy,
            "res": res,
            "dm": dm,
            "minor_step": minor_step,
            "major_step": major_step
        }
        
        outimg = TPSModule.grid( **params )
        
        if plotpoints:
            pointsize = int( kwargs.get( "pointsize", res / 250.0 ) )
            pointcolour = kwargs.get( "pointcolour", "#FF0000" )
            
            r = TPS_range( **params )
            
            outimg = outimg.convert( "RGB" )
            rec = Image.new( "RGB", ( pointsize, pointsize ), pointcolour )
            
            for x, y in TPS_project_list( g = g, lst = g[ 'src' ] ):
                x -= r[ 'minx' ]
                y -= r[ 'miny' ]
                
                x *= res / 25.4
                y *= res / 25.4
                
                x = x + dm
                y = outimg.size[ 1 ] - ( y + dm )
                
                x -= int( pointsize / 2 )
                y -= int( pointsize / 2 )
                
                x = int( x )
                y = int( y )
                
                outimg.paste( rec, ( x, y ) )
        
    ############################################################################
    #    Image writting on disk or return as numpy.array
    ############################################################################
    
    if outfile != None:
        outimg.save( outfile, dpi = ( res, res ) )
    else:
        return outimg

def TPS_range( **kwargs ):
    """
        Calculation of the range in x and y on the projected space. This
        function have to be called to be able to ensure that all borders of
        the projected image will be in the output image. The border could
        be bigger than the input image.
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
                       
        Optional:
        
        :param minx: minimum x coordinate
        :type minx: float

        :param maxx: maximum x coordinate
        :type maxx: float

        :param miny: minimum y coordinate
        :type miny: float

        :param maxy: maximum y coordinate
        :type maxy: float
        
        Return:
        
        :return: Dictionnary with minx, maxx, miny and maxy
        :return: python dict
        
        Usage:
        
            >>> from TPS import TPS_range
            >>> TPS_range( g = g )
            {'minx': 0.4248203385762995, 'miny': -8.18989068983607, 'maxx': 22.80355700551713, 'maxy': 21.933994861945017}
        
        By default, the range is calculated over a square of 25.4x25.4 mm. The
        size can be change as follow:
        
            >>> TPS_range( g = g, minx = 3.8, maxx = 8.6, miny = 8, maxy = 12.5 )
            {'minx': 3.182284625081442, 'miny': 3.2042699333294786, 'maxx': 7.971971092923031, 'maxy': 8.260636686117108}

    """
    g = kwargs.get( "g" )
    
    minx = kwargs.get( "minx", CONF_minx )
    maxx = kwargs.get( "maxx", CONF_maxx )
    miny = kwargs.get( "miny", CONF_miny )
    maxy = kwargs.get( "maxy", CONF_maxy )
    
    minx = int( minx )
    maxx = int( maxx )
    miny = int( miny )
    maxy = int( maxy )
    
    return TPSModule.r( g, minx, maxx, miny, maxy )

def TPS_revertGrid( **kwargs ):
    """
        This function will calculate an approximation of the reverse 'g'
        projection function with a grid of size 'gridSize'.
        
        Required:
        :param g: TPS parameters
        :type g: python dictionary
        
        Optional:
        
        :param minx: minimum x coordinate
        :type minx: float

        :param maxx: maximum x coordinate
        :type maxx: float

        :param miny: minimum y coordinate
        :type miny: float

        :param maxy: maximum y coordinate
        :type maxy: float
    
        :param gridSize: Size of the grid
        :type gridSize: float
        
        Return: 
        
        :return: reverted TPS parameters 
        :rtype: python dictionary
        
        Usage:
        
            >>> from TPS import TPS_revertGrid
            >>> TPS_revertGrid( g = g, minx = 3.8, maxx = 8.6, miny = 8, maxy = 12.5 ) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            {'src': array([[ 0.73196776, -1.04457354],
                [ 0.70280997, -0.37590164],
                [ 0.67290928,  0.29242804],
                ...
                [ 7.85309611,  5.39976625],
                [ 7.74903413,  6.16864023],
                [ 7.6712161 ,  6.95973553]]), 'scale': 1.1186499957839895, 'linear': array([[-1.17833397,  2.47171158],
                [ 1.14384952,  0.37575455],
                [ 0.02991905,  1.10383404]]), 'be': 0.08787071167463428, 'dst': array([[  0.  ,   0.  ],
                [  0.  ,   0.75],
                [  0.  ,   1.5 ],
                ...
                [  8.25,  10.5 ],
                [  8.25,  11.25],
                [  8.25,  12.  ]]), 'weights': array([[ -5.76470366e-04,   2.78386997e-04],
                [ -8.68889465e-05,   8.01920970e-06],
                [ -2.07563097e-04,   7.49686291e-05],
                ...
                [  1.41364134e-03,  -3.90979871e-03],
                [ -1.69564492e-03,  -3.67606453e-04],
                [ -1.79900827e-03,  -7.68862839e-04]]), 'shearing': 69.70267341107392}
           
    """
    if TPSModule.lang() == "Python":
        raise NotImplementedError
    
    else:
        g = kwargs.get( "g" )
        
        minx = kwargs.get( "minx", CONF_minx )
        maxx = kwargs.get( "maxx", CONF_maxx )
        miny = kwargs.get( "miny", CONF_miny )
        maxy = kwargs.get( "maxy", CONF_maxy )
    
        gridSize = kwargs.get( "gridSize", CONF_gridSize )
        
        return TPSModule.revert( g, minx, maxx, miny, maxy, gridSize )

def TPS_revertDownSampling( **kwargs ):
    """
        This function will calculate an approximation of the reverse 'g'
        projection function with a grid of size 'gridSize'.
        
        Required:
        
        :param g: TPS parameters
        :type g: python dictionary
                       
        Optional:
        
        :param minx: minimum x coordinate
        :type minx: float

        :param maxx: maximum x coordinate
        :type maxx: float

        :param miny: minimum y coordinate
        :type miny: float

        :param maxy: maximum y coordinate
        :type maxy: float
    
        :param gridSize: Size of the grid
        :type gridSize: float
    
        :param useWeights: Use the weights of the TPS parameters to select only the important points
        :type useWeights: boolean
    
        :param weightslimit: Minimum weight to select an important point
        :type weightslimit: float
    
        :param nbrandom: Number of random points added on the grid
        :type nbrandom: int
        
        Return: 
        
        :return: reverted TPS parameters 
        :rtype: python dictionary
        
        Usage:
        
            >>> from TPS import TPS_revertDownSampling
            >>> import random
            >>> random.seed( 1337 )
            >>> TPS_revertDownSampling( g = g, minx = 3.8, maxx = 8.6, miny = 8, maxy = 12.5 ) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            {'src': array([[ 3.9724    ,  6.5354    ],
                [ 6.6969    ,  4.1181    ],
                [ 6.5394    ,  7.2362    ],
                ...
                [ 4.9218699 ,  4.25749054],
                [ 6.31487586,  5.73546154],
                [ 6.72093514,  3.63999746]]), 'scale': 1.1215773696061617, 'linear': array([[-1.25509043,  2.50578527],
                [ 1.14208054,  0.37683407],
                [ 0.03529867,  1.11308922]]), 'be': 0.09332928719306405, 'dst': array([[  3.6929    ,  10.3819    ],
                [  6.5827    ,   8.8386    ],
                [  6.7756    ,  12.0866    ],
                ...
                [  4.76471117,   8.30285123],
                [  6.32212845,  10.78402949],
                [  6.57102918,   8.19746182]]), 'weights': array([[  5.06859268e-02,  -2.42368744e-02],
                [ -4.56663710e-02,  -4.84611532e-02],
                [  8.55181153e-02,  -5.03931910e-02],
                ...
                [  1.41541206e-03,  -4.29811681e-03],
                [  1.23069231e-02,  -1.74461562e-03],
                [  9.58254505e-03,   5.90492134e-03]]), 'shearing': 69.52627878026158}
        
        .. note::
            
            Because of the addition of random points to generate the TPS
            parameter, the PyUnit test need to fix the random.seed(). This is
            not mendatory in a real case.
    """
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

################################################################################
# 
#    Deprecated functions
# 
################################################################################

@deprecated( "Use the TPS_grid() function instead" )
def TPS_Grid( **kwargs ):
    return TPS_grid( **kwargs )

@deprecated( "Use the TPS_image() function instead" )
def TPS_Image( **kwargs ):
    return TPS_image( **kwargs )
