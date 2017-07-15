################################################################################
#
#                           TPS Cython implementation
#
#
#
#    The aim of this Cython library is to implement a fast and efficient version
#    of the article of Bookstein (1989). This library contains two types of
#    functions:
#
#    The functions with a "_" at the beginning are pure-C function, for high
#    performance. Those functions can not be called from Python code directly.
#    There are defined with the "cdef" statement.
#
#    The functions without a "_" and defined with the "def" statement are python
#    functions for non-critical parts, or for wrapping pure-C non callables
#    functions.
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
#
#             ┌──────────────────────────────────────────────────┐                          
#             │ ┌───────────┐           CODER'S LICENCE          │
#             │ │           │                                    │
#             │ │  !\/\/\/! │   Name:____De_Donno_Marco_________ │        
#             │ │  !  _  _! │   Address:_On_my_keyboard_________ │        
#             │ │ (! (.)(.) │   Date of birth:__21.03.1989______ │        
#             │ │  !   ___\ │   Sex: _Yes_please________________ │        
#             │ │  !____/   │   Height:_197cm__ Weight:__85__ Kg.│        
#             │ │  /    \   │   Coding restrictions:__None______ │        
#             │ │ //~~~~\\  │   Licence Number: #MD1337PR0G_____ │        
#             │ └───────────┘                                    │
#             └──────────────────────────────────────────────────┘                             
#
################################################################################

################################################################################
#    
#    Cythonization options
#    
#        The options here below are optimization options to disable some of the
#        security checks done in C to prevent out-of-memory errors and segfault.
#        
#        Because of performance concerns, all security checks are disabled. The
#        code is (normally) done to avoid any error in memory access. If any
#        error would appear, Python could simply crash without any explanation
#        or error signaling. Re-enable all Cython security checks (to True) and
#        debug the code...
#
################################################################################

#cython: boundscheck      = False
#cython: wraparound       = False
#cython: nonecheck        = False
#cython: initializedcheck = False
#cython: embedsignature   = True
#cython: cdivision        = True

from scipy.linalg.basic     import solve, inv
from scipy.spatial.distance import cdist
import numpy as np

cimport cython
cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack

from cython.parallel        cimport parallel, prange, threadid
from cython.view            cimport array as cvarray
from libc.math              cimport log, exp, sqrt
from libc.math              cimport floor, ceil, round
from libc.math              cimport sin, cos, tan, acos, M_PI
from libc.stdlib            cimport malloc, free

################################################################################
#
#    Memory allocation
#
#        All the memory allocations are made in row-major style (row-by-row
#        storage in the malloc'ed variable). The access of a n x m matrix,
#        with n elements by row, is made with the statement:
#        
#            var_container[ ( i ) + ( j * n ) ]
#                           ╘═╤═╛   ╘═══╤═══╛
#                            col       row
#
#        which is equivalent to:
#
#            var[ i , j ]
#
#        For readability, the row and column are indicated in parentheses.
#
#        The Cython code for memory allocation is C-style (n x m double matrix):
#
#            cdef double * var_container = < double * > malloc( n * m * sizeof( double ) )
#
#        The Cython code to access with two parameters would be:
#
#            cdef double[ : , : ] var = < double[ :n, :m ] > var_container
#
#        This variable would, however, be a CPython variable, and not a pure-C
#        variable. The performances would be lower.
#
#        Graphically, this management can be shown as:
#
#        ╒═══════╤═══════╤═══════╤═══════╤══   ══╤═══════╤═══════╤═══════╤═══════╕
#        │   1   │   2   │   3   │   4   │  ...  │  n-3  │  n-2  │  n-1  │   n   │─>┐
#        ╘═══════╧═══════╧═══════╧═══════╧══   ══╧═══════╧═══════╧═══════╧═══════╛  │
#     ┌───<──────────<───────────<───────────<─────────<───────────<───────────<────┘
#     │  ╒═══════╤═══════╤═══════╤═══════╤══   ══╤═══════╤═══════╤═══════╤═══════╕
#     └>─│  n+1  │  n+2  │  n+3  │  n+4  │  ...  │ 2*n-3 │ 2*n-2 │ 2*n-1 │  2*n  │─>┐
#        ╘═══════╧═══════╧═══════╧═══════╧══   ══╧═══════╧═══════╧═══════╧═══════╛  │
#     ┌───<──────────<───────────<───────────<─────────<───────────<───────────<────┘
#     │  ╒═══════╤═══════╤═══════╤═══════╤══   ══╤═══════╤═══════╤═══════╤═══════╕
#     └>─│ 2*n+1 │ 2*n+2 │ 2*n+3 │ 2*n+4 │  ...  │ 3*n-3 │ 3*n-2 │ 3*n-1 │  3*n  │─>┐
#        ╘═══════╧═══════╧═══════╧═══════╧══   ══╧═══════╧═══════╧═══════╧═══════╛  │
#     ┌───<──────────<───────────<────────  ...  ──────<───────────<───────────<────┘
#     │  ╒═══════╤═══════╤═══════╤═══════╤══   ══╤═══════╤═══════╤═══════╤═══════╕
#     └>─│m*n-n+1│m*n-n+2│m*n-n+3│m*n-n+4│  ...  │ m*n-3 │ m*n-2 │ m*n-1 │  m*n  │
#        ╘═══════╧═══════╧═══════╧═══════╧══   ══╧═══════╧═══════╧═══════╧═══════╛
#                                          _________________________________
#                                        ,'                                 `.
#                                       /                                     \
#                                      |       Of course, the indexes in       |
#                                      |    Cython starts at 0, and not 1 !!   |
#                                       \                                     /
#                                        `._________________________________,'
#                                               \   ^__^
#                                                \  (oo)\_______
#                                                   (__)\       )\/\
#                                                       ||----w |
#                                                       ||     ||
################################################################################

cdef double _euclidean_dist(
        double a_x,
        double a_y,
        double b_x,
        double b_y 
    ) nogil:
    
    ############################################################################
    #
    #    Euclidean distance
    #        
    #        Pure-C implementation of Euclidean distance calculation between
    #        two points. The function is called with 4 arguments instead of 2
    #        points because of performances.
    #
    ############################################################################
    
    return sqrt(
        ( b_x - a_x ) ** 2 +
        ( b_y - a_y ) ** 2
    )

cdef void _matrix_self_euclidean_dist(
        double [ : , : ] input,
        double * ret
    ) nogil:
    
    ############################################################################
    #    
    #    Matrix Euclidean distance
    #
    #        Calculation of Euclidean distances between all points in a matrix.
    #        The Euclidean distance between the point i and j is stored in the
    #        ret matrix in the ( i, j ) coordinates.
    #
    #        With different experimentations, the use of multiple cores to
    #        compute the _matrix_self_euclidean_dist matrix, the use of multiple
    #        cores on small matrices is useless, and more time consuming. If
    #        the matrix is greater than 150 x 150, then the use of 4 processors
    #        is the more efficient.
    #
    ############################################################################
    
    cdef int i, j
    cdef int n = input.shape[ 0 ]
    
    if n < 150: 
        for i from 0 <= i < n:
            for j from 0 <= j < n:
                if i == j:
                    ret[ i + j * n ] = 0
                else:
                    ret[ i + j * n ] = _euclidean_dist( input[ i, 0 ], input[ i, 1 ], input[ j, 0 ], input[ j, 1 ] )
    
    else:
        with nogil, parallel( num_threads = 4 ):
            for i in prange( 0, n ):
                for j from 0 <= j < n:
                    if i == j:
                        ret[ i + j * n ] = 0
                    else:
                        ret[ i + j * n ] = _euclidean_dist( input[ i, 0 ], input[ i, 1 ], input[ j, 0 ], input[ j, 1 ] )

cdef void _U(
        double * vec,
        long n
    ) nogil:
    
    ############################################################################
    #    
    #    U function
    #
    #        Implementation of the U function as defined by Bookstein.
    #
    #        Even if the U function could be applied on multiples processors,
    #        the time to calculate the TPS parameters is shorter on one core.
    #
    ############################################################################
    
    cdef int x
    
    for x in xrange( n ):
        if vec[ x ] == 0:
            continue
        else:
            vec[ x ] *= vec[ x ] 
            vec[ x ]  = vec[ x ] * log( vec[ x ] )
        
def generate( 
        double[ : , : ] src not None,
        double[ : , : ] dst not None,
    ):
    
    ############################################################################
    #
    #    TPS distortion parameter calculation
    #
    #        Wrapper for the pure-C function _generate
    #
    #        The memory allocation is also made here because of the construction
    #        of the _generate function (in-place variable allocation). The
    #        _generate function does not make any return, but allocate directly
    #        the memory space given in the pointers (W, linear and be).
    #
    #        The memory allocation is almost stable for any matrix size (between
    #        ~10^-5 and ~10^-4 seconds between n = 5 and n = 1000). The gain
    #        in time is important compared with the python implementation
    #        (almost 2 orders of magnitude with the memory allocation and the
    #        _generate call).
    #
    ############################################################################
    
    cdef int n = src.shape[ 0 ]
    
    # Memory allocation C-pointer-style
    cdef double * W         = < double * > malloc( n * 2 * sizeof( double ) )
    cdef double * linear    = < double * > malloc( 3 * 2 * sizeof( double ) )
    cdef double * be        = < double * > malloc( 1 *     sizeof( double ) )
    cdef double * scale     = < double * > malloc( 1 *     sizeof( double ) )
    cdef double * shearing  = < double * > malloc( 1 *     sizeof( double ) )
    
    # Call of the pure-C function
    _generate(
        src, dst,     # input variables
        W, linear, be # output storage
    )
    
    scale[ 0 ] = sqrt( ( linear[ 2 ] * linear[ 5 ] ) - ( linear[ 3 ] * linear[ 4 ] ) )
    shearing[ 0 ] = _angle_between( linear[ 2 ], linear[ 4 ], linear[ 3 ], linear[ 5 ], 1 )
    
    ############################################################################
    #    
    #    Python object return
    #    
    ############################################################################
    
    return {
        'src':      np.asarray( src ),
        'dst':      np.asarray( dst ),
        'linear':   np.array( < double [ :3, :2 ] > linear ),
        'scale':    scale[ 0 ],
        'shearing': shearing[ 0 ],
        'weights':  np.array( < double [ :n, :2 ] > W ),
        'be':       be[ 0 ],
    }
    
cdef void _generate(
        double [ : , : ] src,
        double [ : , : ] dst,
        
        double * out_W,
        double * out_linear,
        double * out_be
    ):
    
    ############################################################################
    #    
    #    TPS parameter - Fast calculation with a pure-C function
    #
    #        Direct implementation of the Bookstein (1989) article, with some
    #        mathematical simplifications / optimizations. The results are
    #        however exactly the same.
    #
    #        Fast pure-C function. The time gain is important compared to the
    #        pure Python implementation. The main gains are made by variable
    #        typing, manual memory management and manual mathematical
    #        calculation.
    #
    #        An interesting point is to be able to call this function directly
    #        from pure-C functions, to be able to optimize some heavy parts of
    #        calculations and optimization (reversing the TPS function for the
    #        image distortion. To be honest, this pure-C function provides no
    #        time gain for very small matrices (up to ~10 minutiae), but when
    #        the number of minutiae increase, the calculation time is almost
    #        flat, compared with an exponential calculation time for the Python
    #        version. This is due to the memory management overhead. The
    #        critical part (the L|P matrix solving) is the same because the same
    #        optimize function is called (the BLAS/LAPACK library, coded in
    #        Fortran).
    #                                                     \\\///
    #                                                    / _  _ \
    #                                                  (| (.)(.) |)
    ############################################### --OOOo--()--oOOO. -- #######
    
 
    # Generic variable definition
    cdef int n = src.shape[ 0 ]
    cdef int nrhs = 2
    
    # Loop iterators
    cdef int i, j, k, m
    
    ############################################################################
    #
    #    K matrix
    #
    #        Size : n x n
    #
    ############################################################################
    
    cdef double * K_container = < double * > malloc( ( n ** 2 ) * sizeof( double ) )
    
    _matrix_self_euclidean_dist( src, K_container )
    _U( K_container, n ** 2 ) 
    
    ############################################################################
    #
    #    L matrix
    #
    #        Size : ( n + 3 ) x ( n + 3 )
    #
    #        The filling of the L matrix is made iteratively, by variable. The
    #        first variable to be filled is the K variable, then the source 
    #        points. The null matrix is filled in the initialization phase.
    #
    #        This filling process is very fast; the use of multiple processors
    #        is useless.
    #        
    ############################################################################
    
    cdef double * L_container = < double * > malloc( ( ( n + 3 ) ** 2 ) * sizeof( double ) )
    
    # Initialization
    for i from 0 <= i < ( n + 3 ) ** 2:
        L_container[ i ] = 0
    
    # K loading
    for j from 0 <= j < n:
        for i from 0 <= i < n:
            #             col           row                          col       row
            #            -----   -----------------                  -----   ---------
            L_container[ ( i ) + ( j * ( n + 3 ) ) ] = K_container[ ( i ) + ( j * n ) ]
    
    # Source loading
    for k from 0 <= k < n:
        i = n
        j = k
        
        #               col           row
        #            ---------   -----------------
        L_container[ ( i     ) + ( j * ( n + 3 ) )        ] = 1
        L_container[ ( i + 1 ) + ( j * ( n + 3 ) )        ] = src[ k, 0 ]
        L_container[ ( i + 2 ) + ( j * ( n + 3 ) )        ] = src[ k, 1 ]
        
        #               col           row
        #            ---------   -------------------------
        L_container[ ( j )     + ( ( i     ) * ( n + 3 ) )] = 1
        L_container[ ( j )     + ( ( i + 1 ) * ( n + 3 ) )] = src[ k, 0 ]
        L_container[ ( j )     + ( ( i + 2 ) * ( n + 3 ) )] = src[ k, 1 ]
        
    ############################################################################
    #
    #    Transposed V
    #
    ############################################################################
    
    cdef double * V_container = < double * > malloc( ( ( n + 3 ) * 2 ) * sizeof( double ) )
        
    for i from 0 <= i < ( ( n + 3 ) * 2 ):
        V_container[ i ] = 0
    
    for i from 0 <= i < n:
        #             col      row
        #            -----   ---------
        V_container[ ( i )             ] = dst[ i, 0 ]
        V_container[ ( i ) + ( n + 3 ) ] = dst[ i, 1 ]
    
    ############################################################################
    #
    #    Calculation of the distortion parameters
    #
    #        Equivalent in the original article of:
    #
    #            Wa = np.dot( inv( L ), V.T )
    #
    #        simplified as (because of math...):
    #
    #            Wa = scipy.linalg.solve( L, V.T )
    #
    #        and optimized by calling the Cython Lapack implementation directly
    #        
    #            cython_lapack.dgesv(
    #               &nbis, &nrhs,
    #               L_container, &nbis, piv_pointer,
    #               V_container, &nbis,
    #               &info
    #            )
    #        
    #        This implementation is a C-wrapper of the Lapack library. The
    #        function call is a pointer call: the values are directly read and
    #        written to the memory, without any return value (the solution is
    #        stored in the "V_container" memory location). This implementation
    #        is called instead of the Scipy one because the Cython produced code
    #        is pure C, so directly callable without the Python wrapper present
    #        in the Scipy version.
    #
    #        The cython_lapack module is available in the Scipy distribution
    #        ONLY if the BLAS/LAPACK distribution is present. The compilation
    #        process need to be done by hand on the local machine. A copy of the
    #        cython_lapack.pyd library should be present, if needed, in this
    #        package distribution.
    #
    #        A test should be done to evaluate the performance, on really large
    #        matrices of a full-manual multi-core matrix solving (the LAPACK
    #        implementation seems to be on one core...).
    #
    ############################################################################
    
    # Pivoting array
    cdef int * piv_pointer = < int * > malloc( ( n + 3 ) * sizeof( int ) )
    
    # Information vector
    cdef int info = 0
    cdef int nbis = n + 3
    
    # Solver
    cython_lapack.dgesv(
        &nbis, &nrhs,
        L_container, &nbis, piv_pointer,
        V_container, &nbis,
        &info
    )
    
    ############################################################################
    #
    #    Separation between linear and non-linear distortion
    #
    #        The linear and non-linear parameters are stored n the "V_container"
    #        matrix. A separation have to be done, allowing the storage in two
    #        separate variables. This is done simply by coping the data from the
    #        "W_container" matrix (of size ( n + 3 ) x 2) in the "W_container"
    #        (of size n x 2) and the "a_container" (of size 3 x 2).
    #
    ############################################################################
    
    cdef double * W_container = < double * > malloc( n * 2 * sizeof( double ) )
    cdef double * a_container = < double * > malloc( 3 * 2 * sizeof( double ) )
    
    for i from 0 <= i < n:
        #             col           row
        #            -----   -------------------------
        W_container[ ( i )         ] = V_container[ i ]
        W_container[ ( i ) + ( n ) ] = V_container[ i + n + 3 ]
    
    for i from 0 <= i < 3:
        #             col     row
        #            -----   -----
        a_container[ ( i )         ] = V_container[ n + i ]
        a_container[ ( i ) + ( 3 ) ] = V_container[ n + i + n + 3 ]
    
    ############################################################################
    #
    #    Bending energy calculation
    #
    #        The bending energy is calculated in two steps. The fist one is the
    #        calculation of ( W.t . K ), stored in the "WK_container" malloc.
    #        The second step is the calculation of the ( W.t . K . W ) matrix,
    #        stored in the "WKW_container". The be is then directly evaluated on
    #        the "WKW_container", and returned as a double variable.
    #
    ############################################################################
    
    cdef double * WK_container = < double * > malloc( n * 2 * sizeof( double ) )
    
    for i from 0 <= i < ( n * 2 ):
        WK_container[ i ] = 0
    
    for j from 0 <= j < 2:
        for i from 0 <= i < n:
            for k from 0 <= k < n:
                #              col       row                       col       row                      col       row   
                #             -----   ---------                   -----   ---------                  -----   ---------
                WK_container[ ( j ) + ( i * 2 ) ] += W_container[ ( k ) + ( j * n ) ] * K_container[ ( i ) + ( k * n ) ]
    
    ############################################################################
    #
    #    Optimizations
    #
    #        Since only 2 out of 4 elements in the WKW_container are relevant
    #        (especially the positions 0 (top left) and 3 (bottom right)), the 2
    #        other variables are not calculated. The original code would be this
    #        one:
    #
    #            cdef double * WKW_container = < double * > malloc( 2 * 2 * sizeof( double ) )
    #                
    #            for i from 0 <= i < 4:
    #                WKW_container[ i ] = 0
    #           
    #            for j from 0 <= j < 2:
    #                for i from 0 <= i < 2:
    #                    for k from 0 <= k < n:
    #                        WKW_container[ ( j ) + ( i * 2 ) ] +=   \
    #                            WK_container[ ( j ) + ( k * 2 ) ] * \
    #                            W_container [ ( k ) + ( i * n ) ]
    #            
    #            cdef double be = 0.5 * ( WKW_container[ 0 ] + WKW_container[ 3 ] ) 
    #
    #        Optimization is made by using only the coordinates
    #            
    #            ( i, j ) = ( 0, 0 )
    #
    #        and
    #
    #            ( i, j ) = ( 1, 1 )
    #
    #        The second optimization is made by the distributivity property. The
    #        dot product is distributed, and directly accumulated in the be
    #        variable. This allows not to store the "WKW_container" variable, and
    #        therefore, not iterate a second time over it.
    #
    #        Effective gain in time is negligible, but still cool to do.
    #
    ############################################################################
    
    cdef double be = 0
    
    for m from 0 <= m < 2:
        i = m
        j = m
    
        for k from 0 <= k < n:
            #                            col       row                      col       row   
            #                           -----   ---------                  -----   ---------
            be += 0.5 * ( WK_container[ ( j ) + ( k * 2 ) ] * W_container[ ( k ) + ( i * n ) ] )
    
    ############################################################################
    #
    #    Returns - memory allocation
    #
    #        The memory allocation is not made before because of the
    #        dissimilarity in structure. Since the matrices are relatively small
    #        (n x 2 and 3 x 2), the memory and time loss is very small.
    #
    ############################################################################
    
    for i from 0 <= i < n:
        for j from 0 <= j < 2:
            #         row       col                    col       row
            #      ---------   -----                  -----   ---------
            out_W[ ( 2 * i ) + ( j ) ] = W_container[ ( i ) + ( j * n ) ]
        
    for i from 0 <= i < 3:
        for j from 0 <= j < 2:
            #              row       col                    col       row
            #           ---------   -----                  -----   ---------
            out_linear[ ( 2 * i ) + ( j ) ] = a_container[ ( i ) + ( j * 3 ) ]
    
    out_be[ 0 ] = be
    
    ############################################################################
    #    Free allocated memory
    ############################################################################
    
    free( K_container   )
    free( L_container   )
    free( V_container   )
    free( piv_pointer   )
    free( W_container   )
    free( a_container   )
    free( WK_container  )

def revert( 
        dict g,
        np.float64_t minx,
        np.float64_t maxx,
        np.float64_t miny,
        np.float64_t maxy,
        np.float64_t step
    ):
    
    ############################################################################
    #
    #    Reverting function
    #
    #        Function to "revert" a TPS function:
    #
    #            1. Project a grid with the TPS function
    #            2. Use the projected points as src, and initial grid as dst
    #            3. Calculate the TPS function
    #
    #        Since the TPS projection function is not bijective, This ugly
    #        method is the only one capable of estimate the reverted TPS function.
    #
    ############################################################################
    
    src2 = []
    dst2 = []
    
    cdef double tmp[ 2 ]
    cdef double [ : ] tmp_view = tmp
    
    cdef double x, y
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] W = g[ 'weights' ]
    cdef double[ : , : ] W_view = W
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] linear = g[ 'linear' ]
    cdef double[ : , : ] linear_view = linear
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] src = g[ 'src' ]
    cdef double[ : , : ] src_view = src
    
    for x from 0 <= x <= maxx by step:
        for y from 0 <= y <= maxy by step:
            _project( x, y, linear_view, W_view, src_view, tmp_view )
            
            src2.append( ( tmp[ 0 ], tmp[ 1 ] ) )
            dst2.append( ( x, y ) )
    
    src2 = np.asarray( src2, dtype = np.float64 )
    dst2 = np.asarray( dst2, dtype = np.float64 )
    
    return generate( src2, dst2 )

def r(
        dict g not None,
        
        np.float64_t minx,
        np.float64_t maxx,
        np.float64_t miny,
        np.float64_t maxy
    ):
    
    ############################################################################
    #
    #    Range function
    #
    #        This function allow to calculate the coordinates of the border
    #        (rectangle of coordinates ((minx, maxx), (miny, maxy)). This
    #        function is fundamental to calculate the offset of a TPS projection
    #        function since the coordinates, in the TPS space, can be negative.
    #        The coordinate in the Image space can not be negative.
    #
    ############################################################################
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] W = g[ 'weights' ]
    cdef double[ : , : ] W_view = W
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] linear = g[ 'linear' ]
    cdef double[ : , : ] linear_view = linear
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] src = g[ 'src' ]
    cdef double[ : , : ] src_view = src
    
    cdef double out[ 4 ]
    cdef double[ : ] out_view = out
    
    _r( linear_view, W_view, src_view, minx, maxx, miny, maxy, out_view )
    
    return {
        "minx": out[ 0 ],
        "maxx": out[ 1 ],
        "miny": out[ 2 ],
        "maxy": out[ 3 ]
    }
    
cdef void _r(
        double[ : , : ] linear,
        double[ : , : ] W,
        double[ : , : ] src,
        
        double minx,
        double maxx,
        double miny,
        double maxy,
        
        double[ : ] ret
    ):
    
    ############################################################################
    # 
    #    Range function - Pure C
    #
    #        Pure C implementation of the r() function.
    #
    ############################################################################
    
    cdef double cx = 0.5 * ( minx + maxx )
    cdef double cy = 0.5 * ( miny + maxy )
     
    cdef double out[ 2 ]
    cdef double[ : ] out_view = out
    _project( cx, cy, linear, W, src, out_view )
    cx, cy = out
    
    cdef double retminx = cx
    cdef double retmaxx = cx
    cdef double retminy = cy
    cdef double retmaxy = cy
    
    ############################################################################
    #    Preparation of the borders
    ############################################################################
     
    cdef int i
    cdef double x, y
    cdef double xp, yp
    cdef int nbstep = 200
    
    cdef double * t = < double * > malloc( ( nbstep + 1 ) * 4 * 2 * sizeof( double ) )
    
    for x from minx <= x <= maxx by ( maxx - minx ) / float( nbstep ):
        _project( x, miny, linear, W, src, out_view )
        t[ i ], t[ i + 1 ] = out
        i += 2
        
        _project( x, maxy, linear, W, src, out_view )
        t[ i ], t[ i + 1 ] = out
        i += 2
        
    for y from miny <= y <= maxy by ( maxy - miny ) / float( nbstep ):
        _project( minx, y, linear, W, src, out_view )
        t[ i ], t[ i + 1 ] = out
        i += 2
        
        _project( maxx, y, linear, W, src, out_view )
        t[ i ], t[ i + 1 ] = out
        i += 2
    
    ############################################################################
    #    Extremums
    ############################################################################
     
    i = 0
    while i < nbstep * 4 * 2:
        xp = t[ i ]
        yp = t[ i + 1 ]
         
        if xp < retminx:
            retminx = xp
          
        elif xp > retmaxx:
            retmaxx = xp
          
        if yp < retminy:
            retminy = yp
          
        elif yp > retmaxy:
            retmaxy = yp
         
        i += 2
    
    ret[ 0 ] = retminx
    ret[ 1 ] = retmaxx
    ret[ 2 ] = retminy
    ret[ 3 ] = retmaxy
    
    ############################################################################
    #    Garbage collector
    ############################################################################
    
    free( t )
    
def image(
        long[ : , : ] indata,
        dict g not None,
        dict range not None,
        double res,
        long[ : , : ] voidimg,
        int ncore = 8
    ):

    ############################################################################
    #    
    #    Image distortion function
    #    
    #        This function allows to distort an image based on a distortion
    #        parameter g. This function is a inplace function to allow multi-
    #        core processing.
    #    
    #        The distortion process is made with a reverse projection: The range
    #        of distortion is calculated (the 'range' parameter) in the first
    #        place. This range represents the distorted image. This function then
    #        create an empty image, and make the iteration over it. Each point of
    #        this destination image is projected on the original image to infer
    #        the colour of the specific pixel.
    #    
    #        This methodology implies the calculation of an estimation of the
    #        reverse projection function. The quality of the image depends on the
    #        estimation grid size (see the revert function documentation).
    #    
    #        The _project function cannot (for the moment) be used because of
    #        the memory management for the multi-core processing. Some
    #        adjustment has to be done.
    #    
    ############################################################################
    
    # Resolution factor (px <-> mm)
    cdef np.float64_t fac = res / 25.4
    
    # mm
    cdef np.float64_t x_mm, y_mm
    cdef np.float64_t xp_mm, yp_mm
    
    # Px
    cdef np.int_t x, y
    cdef np.int_t xp, yp
    
    cdef np.int_t minx = 0
    cdef np.int_t maxx = voidimg.shape[ 0 ] - 1
    cdef np.int_t miny = 0
    cdef np.int_t maxy = voidimg.shape[ 1 ] - 1
    
    cdef np.float64_t rminx = range[ 'minx' ]
    cdef np.float64_t rminy = range[ 'miny' ]
    
    # Color value
    cdef np.int_t c
    
    # Distorsion parameters
    cdef np.ndarray[ dtype = np.float64_t, ndim = 1 ] XY     = np.zeros( [ 2 ], dtype = np.float64, order = 'F' )
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] W      = g[ 'weights' ]
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] linear = g[ 'linear' ]
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] src    = g[ 'src' ]
    
    cdef np.int_t nbsrc = src.shape[ 0 ]
    
    cdef np.int_t jj, jjj
    cdef int tid = 0
    
    cdef np.float64_t dx, dy
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] s  = np.ones( [ ncore, 3 ],         dtype = np.float64, order = 'F' )
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] d1 = np.zeros( [ ncore, 3 ],        dtype = np.float64, order = 'F' )
    cdef np.ndarray[ dtype = np.float64_t, ndim = 3 ] sd = np.zeros( [ ncore, nbsrc, 2 ], dtype = np.float64, order = 'F' )
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] su = np.zeros( [ ncore, nbsrc ],    dtype = np.float64, order = 'F' )
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] us = np.zeros( [ ncore, nbsrc ],    dtype = np.float64, order = 'F' )
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] d2 = np.zeros( [ ncore, 2 ],        dtype = np.float64, order = 'F' )
    
    with nogil, parallel( num_threads = ncore ):
        for x in prange( minx, maxx + 1 ):
            tid = threadid()
            
            for y from miny <= y <= maxy:
                xp = 0
                yp = 0
                
                x_mm = x / fac + rminx
                y_mm = y / fac + rminy
                
                s[ tid, 1 ] = x_mm
                s[ tid, 2 ] = y_mm
                
                ################################################################
                #    Projection of ( x, y ) in the original image
                ################################################################
                 
                # d1 = np.dot( s, linear )
                for jj from 0 <= jj < 2:
                    d1[ tid, jj ] = 0
                     
                    for jjj from 0 <= jjj < 3:
                        d1[ tid, jj ] += s[ tid, jjj ] * linear[ jjj, jj ]
                         
                # sd = ( src - XY ) ** 2
                for jj from 0 <= jj < nbsrc:
                    sd[ tid, jj, 0 ] = src[ jj, 0 ] - x_mm
                    sd[ tid, jj, 1 ] = src[ jj, 1 ] - y_mm
                     
                for jj from 0 <= jj < nbsrc:
                    sd[ tid, jj, 0 ] *= sd[ tid, jj, 0 ]
                    sd[ tid, jj, 1 ] *= sd[ tid, jj, 1 ]
                 
                # su = np.sum( sd, axis = -1 )
                for jj from 0 <= jj < nbsrc:
                    su[ tid, jj ] = sd[ tid, jj, 0 ] + sd[ tid, jj, 1 ]
     
                # U2( su )
                for jj from 0 <= jj < nbsrc:
                    if su[ tid, jj ] != 0:
                        su[ tid, jj ] = su[ tid, jj ] * log( su[ tid, jj ] )
                 
                # d2 = np.dot( us, W )
                for jj from 0 <= jj < 2:
                    d2[ tid, jj ] = 0
                     
                    for jjj from 0 <= jjj < nbsrc:
                        d2[ tid, jj ] += su[ tid, jjj ] * W[ jjj, jj ]
                 
                ################################################################
                #    Input data searching
                ################################################################
                
                xp = int( ( d1[ tid, 0 ] + d2[ tid, 0 ] ) * fac )
                yp = int( ( d1[ tid, 1 ] + d2[ tid, 1 ] ) * fac )
                
                if xp < 0 or xp >= indata.shape[ 0 ] - 1 or yp < 0 or yp >= indata.shape[ 1 ] - 1:
                    c = 255
                else:
                    # Bilinear Interpolation
                    dx = ( d1[ tid, 0 ] + d2[ tid, 0 ] ) % 1
                    dy = ( d1[ tid, 1 ] + d2[ tid, 1 ] ) % 1
                      
                    c = int( 
                            ( 1 - dy ) * ( ( 1 - dx ) * indata[ xp, yp ] + dx * indata[ xp + 1, yp ] ) + 
                            dy * ( ( 1 - dx ) * indata[ xp, yp + 1 ] + dx * indata[ xp + 1, yp + 1 ] )
                        )
                
                voidimg[ x, y ] = c

def grid(
        dict g not None,
        
        double minx,
        double maxx,
        double miny,
        double maxy,
        
        double res = 500,
        
        double major_step = 1,
        double minor_step = 0.02,
        
        int dm = 5
    ):
    
    ############################################################################
    #
    #    Distorsion grid
    #
    #        Creation of the distorsion grid passed in argument.
    #
    ############################################################################
    
    ############################################################################
    #    Parsing of the parameters in memoryview for fast access
    ############################################################################
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] linear = g[ 'linear' ]
    cdef double[ : , : ] linear_view = linear
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] W = g[ 'weights' ]
    cdef double[ : , : ] W_view = W
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] src = g[ 'src' ]
    cdef double[ : , : ] src_view = src
    
    ############################################################################
    #    Upsampling the range, to avoid the open sqare of the grid
    ############################################################################
    
    cdef double dx = maxx - minx
    cdef double dy = maxy - miny
    
    maxx = minx + ceil( dx / major_step ) * major_step
    maxy = miny + ceil( dy / major_step ) * major_step
    
    ############################################################################
    #    Determination of the distortion range
    ############################################################################
    
    cdef double range[ 4 ]
    cdef double[ : ] range_view = range
    
    _r( linear_view, W_view, src_view, minx, maxx, miny, maxy, range_view )
    
    cdef double rminx = range[0]
    cdef double rmaxx = range[1]
    cdef double rminy = range[2]
    cdef double rmaxy = range[3]
    
    cdef int sizex = int( ( 1 + float( res ) / 25.4 * ( rmaxx - rminx ) ) + 2 * dm )
    cdef int sizey = int( ( 1 + float( res ) / 25.4 * ( rmaxy - rminy ) ) + 2 * dm )
    
    size = [
        sizex,
        sizey
    ]
    
    cdef np.ndarray[ dtype = int, ndim = 2 ] outimg
    outimg = np.empty( size, dtype = int )
    outimg.fill( 255 )
    
    ############################################################################
    #    Temporary variables
    ############################################################################
    
    cdef double i, j, x, y
    cdef int xp, yp
    
    cdef double tmp[ 2 ]
    cdef double[ : ] tmp_view = tmp
    
    ############################################################################
    #    Creation of the grid
    #        major_step is the distance between lines in the grid 
    #        minor_step is the distance between two consecutive points on a line
    ############################################################################
    
    # Vertical lines
    for i from minx <= i <= maxx by major_step:
        for j from miny <= j <= maxy by minor_step:
            _project( i, j, linear_view, W_view, src_view, tmp_view )
            x, y = tmp
            
            xp = int( ( x - rminx ) * float( res ) / 25.4 )
            yp = int( ( y - rminy ) * float( res ) / 25.4 )
            
            if xp >= 0 and yp >= 0 and xp < sizex and yp < sizey:
                outimg[ xp + dm, yp + dm ] = 0
    
    # Horizontal lines
    for i from miny <= i <= maxy by major_step:
        for j from minx <= j <= maxx by minor_step:
            _project( j, i, linear_view, W_view, src_view, tmp_view )
            x, y = tmp
               
            xp = int( ( x - rminx ) * float( res ) / 25.4 )
            yp = int( ( y - rminy ) * float( res ) / 25.4 )
            
            if xp >= 0 and yp >= 0 and xp < sizex and yp < sizey:
                outimg[ xp + dm, yp + dm ] = 0
            
    ############################################################################
    #    Returning the image
    #        The transposition and flip is because of the coordinates change
    #        between image and NIST standard
    ############################################################################
    
    outimg = outimg.T
    outimg = np.flipud( outimg )
    
    return outimg

cdef double _norm(
        double [ : ] vector
    ):
    
    ############################################################################
    #    
    #    Norm
    #        
    #        Calculate the norm of a vector.
    #        
    #        Required:
    #            @param 'vector' : Vector
    #            @type  'vector' : object with memory-view (numpy array or malloc'ed object)
    #                       
    #        Return: 
    #            @return    : Norm of the vector
    #            @return    : double
    #        
    ############################################################################
    
    return sqrt( vector[ 0 ] * vector[ 0 ] + vector[ 1 ] * vector[ 1 ] )

cdef void _unit_vector(
        double [ : ] vector
    ):
    
    ############################################################################
    #    
    #    Unit vector
    #        
    #        Calculate the unit vector of a vector given in input. This
    #        calculation is made inplace.
    #        
    #        Required:
    #            @param 'vector' : Input vector
    #            @type  'vector' : object with memory-view (numpy array or malloc'ed object)
    #                       
    #        Return: 
    #            @return    : None
    #            @return    : void
    #        
    ############################################################################
        
    cdef double n = _norm( vector )
    
    vector[ 0 ] /= n
    vector[ 1 ] /= n

cdef double _angle(
        double [ : ] vector,
        bint deg = 1
    ):
    
    ############################################################################
    #    
    #    Angle calculation
    #        
    #        Calculation of the angle, as defined in the ANSI/NIST 2007
    #        standard, between the 3 o'clock vector ( ( x, y ) = ( 1, 0 ) ) and
    #        the input vector.
    #        
    #        Required:
    #            @param 'vector' : Input vector
    #            @type  'vector' : object with memory-view (numpy array or malloc'ed object)
    #                       
    #        Optional:
    #            @param 'deg' : Return the value in degree
    #            @type  'deg' : boolean
    #            @def   'deg' : True
    #        
    #        Return: 
    #            @return    : Angle
    #            @return    : double
    #        
    ############################################################################
    
    # In-place unit vector calculation
    _unit_vector( vector )
    
    cdef double angle = acos( vector[ 0 ] )
    
    if vector[ 1 ] < 0:
        angle = 2 * M_PI - angle
    
    if deg == 1:
        return angle / M_PI * 180.0
    else:
        return angle

def angle_between(
        double a,
        double b,
        double c,
        double d,
        bint deg = 0,
    ):
    
    ############################################################################
    # 
    #    angle_between
    #
    #        Calculate the angle between two vectors ( a, b ) and ( c, d ),
    #        passed to the function as ( a, b, c, d ).
    # 
    ############################################################################
    
    return _angle_between( a, b, c, d, deg )

cdef double _angle_between(
        double a,
        double b,
        double c,
        double d,
        bint deg = 0,
    ):
    
    ############################################################################
    # 
    #    _angle_between
    #
    #        Pure-C implementation of the angle_between function
    # 
    ############################################################################
    
    cdef double angle = 0
    
    cdef double nv1 = sqrt( a * a + b * b ) 
    cdef double nv2 = sqrt( c * c + d * d ) 
    
    a /= nv1
    b /= nv1
    c /= nv2
    d /= nv2
    
    if a == c and b == d:
        return angle
    
    else:
        angle =  acos( a * c + b * d )
        
        if b < 0:
            angle = 2 * M_PI - angle
        
        angle = angle % ( 2 * M_PI )
        
        if deg == 0:
            return angle
        
        else:
            return angle / M_PI * 180
    
def project(
        dict g,
        
        double x,
        double y,
        double theta,
        
        double dh = 0.01
    ):
    
    ############################################################################
    #    
    #    Point projection
    #    
    #        The projection of the points are done like in the original article
    #        of Bookstein (1989). The calculation of the angle is, however, not
    #        directly calculable in the project function. The partial derivative
    #        could be calculable if the bijective property is assumed true,
    #        which is not the general case. In fingerprint, if the distortion is
    #        reasonable and actually physically plausible, then the bijective
    #        property will be true.
    #    
    #        Here, the general case is used. The definition of the partial
    #        derivative is used to calculate the angle of the minutiae:
    #    
    #    
    #                                  f( x + dh ) - f( x )  
    #                   f'( x ) = lim ──────────────────────
    #                             h→0           h
    #    
    #        This definition is extended in the 2-dimensional plan by projecting
    #        two points spaced by 'dh' unit. By default, the 'dh' distance is
    #        set to 0.01 unit (mm in the fingerprint area). The angle is
    #        calculated by taking the angle between this line and the reference. 
    #    
    #        A faster version could be done by using pointers instead of
    #        memoryview, but the function _project has to be redesigned. The
    #        needs are for the moment not predominant.
    #    
    ############################################################################
    
    ############################################################################
    #    Parsing of the parameters in memoryview for faster access
    ############################################################################
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] linear = g[ 'linear' ]
    cdef double[ : , : ] linear_view = linear
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] W = g[ 'weights' ]
    cdef double[ : , : ] W_view = W
    
    cdef np.ndarray[ dtype = np.float64_t, ndim = 2 ] src = g[ 'src' ]
    cdef double[ : , : ] src_view = src
    
    ############################################################################
    #    Temporary variables
    ############################################################################
    
    cdef double p1[ 2 ]
    cdef double[ : ] p1_view = p1
    
    cdef double p2[ 2 ]
    cdef double[ : ] p2_view = p2
    
    cdef double dp[ 2 ]
    cdef double[ : ] dp_view = dp
    
    cdef double ang
    
    theta = theta / 180.0 * M_PI
    
    ############################################################################
    #    Points projection
    ############################################################################
    
    _project( x, y, linear_view, W_view, src_view, p1_view )
    _project( x + dh * cos( theta ), y + dh * sin( theta ), linear_view, W_view, src_view, p2_view )
    
    dp[0] = p2[0] - p1[0]
    dp[1] = p2[1] - p1[1]
    
    ang = _angle( dp_view )
    
    return p1[0], p1[1], ang

cdef void _project( 
        double x,
        double y,
        
        double[ : , : ] linear,
        double[ : , : ] W,
        double[ : , : ] src,
        
        double[ : ] out
    ) nogil:
    
    ############################################################################
    #    
    #    Projection function
    #        
    #        This function project the point ( x, y ) with the TPS parameters
    #        given by the 'linear', 'W' and 'src' variables. This function is,
    #        for performance purpose, an in-place function. The result is not
    #        returned, but directly stored in the 'out' variable pointer.
    #        
    #        Because of performance, all mathematical operations are manually
    #        coded. The numpy library was to slow and not allowing to have a
    #        pure-C function.
    #        
    #        This code is not designed to work on multiple cores because there
    #        no needs. Moreover, 
    #        
    #        Required:
    #            @param 'x' : x coordinate
    #            @type  'x' : float
    #
    #            @param 'y' : x coordinate
    #            @type  'y' : float
    #
    #            @param 'linear' : x coordinate
    #            @type  'linear' : float
    #
    #            @param 'W' : TPS weights
    #            @type  'W' : object with memory-view (numpy array or malloc'ed object)
    #
    #            @param 'src' : TPS source points
    #            @type  'src' : object with memory-view (numpy array or malloc'ed object)
    #
    #            @param 'out' : Return variable for the ( x, y ) projected point 
    #            @type  'out' : object with memory-view (numpy array or malloc'ed object)
    #
    #        Return: 
    #            @return    : Nothing
    #            @return    : void
    #        
    ############################################################################
    
    cdef int jj, jjj
    
    cdef int nbsrc = src.shape[ 0 ]
    
    cdef double s[ 3 ]
    cdef double d1[ 2 ]
    
    cdef double * sd = < double * > malloc( nbsrc * nbsrc * sizeof( double ) )
    cdef double * su = < double * > malloc( nbsrc * sizeof( double ) )
    cdef double d2[ 2 ]
    cdef double ret[ 2 ]
    
    cdef double xp, yp
    
    ############################################################################
    #    
    #    Projection of the points. All numpy correspondent code is present in
    #    comments to facilitate the review and comprehension of this code.
    #    
    ############################################################################
    
    with nogil:
        s[ 0 ] = 1
        s[ 1 ] = x
        s[ 2 ] = y
        
        # d1 = np.dot( s, linear )
        for jj from 0 <= jj < 2:
            d1[ jj ] = 0
            
            for jjj from 0 <= jjj < 3:
                d1[ jj ] += s[ jjj ] * linear[ jjj, jj ]
        
        # sd = ( src - XY ) ** 2
        for jj from 0 <= jj < nbsrc:
            sd[ jj * nbsrc + 0 ] = src[ jj, 0 ] - x
            sd[ jj * nbsrc + 1 ] = src[ jj, 1 ] - y
            
        for jj from 0 <= jj < nbsrc:
            sd[ jj * nbsrc + 0 ] *= sd[ jj * nbsrc + 0 ]
            sd[ jj * nbsrc + 1 ] *= sd[ jj * nbsrc + 1 ]
        
        # su = np.sum( sd, axis = -1 )
        for jj from 0 <= jj < nbsrc:
            su[ jj ] = sd[ jj * nbsrc + 0 ] + sd[ jj * nbsrc + 1 ]
        
        # U2( su )
        for jj from 0 <= jj < nbsrc:
            if su[ jj ] != 0:
                su[ jj ] = su[ jj ] * log( su[ jj ] )
        
        # d2 = np.dot( us, W )
        for jj from 0 <= jj < 2:
            d2[ jj ] = 0
            
            for jjj from 0 <= jjj < nbsrc:
                d2[ jj ] += su[ jjj ] * W[ jjj, jj ]
        
        ########################################################################
        #    Saving the results in the 'out' variable.
        ########################################################################
            
        out[ 0 ] = d1[ 0 ] + d2[ 0 ]
        out[ 1 ] = d1[ 1 ] + d2[ 1 ]
        
        # garbage collector
        free( sd )
        free( su )
