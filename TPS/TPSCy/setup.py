#!/usr/bin/env python
#  *-* coding: cp850 *-*

from distutils.core         import setup
from distutils.extension    import Extension
from Cython.Distutils       import build_ext

import numpy

setup( 
    cmdclass = { 'build_ext': build_ext },
    ext_modules = [
        Extension( name = "TPSCy",
           sources = [ "TPSCy.pyx" ],
           libraries = [ 'm', 'gomp', 'pthread' ],
           include_dirs = [ numpy.get_include() ],
           extra_compile_args = [ '-fopenmp', '-fdce', '-ffast-math', '-Wfatal-errors', '-w' ],
           extra_link_args = []
        )
    ]
 )
