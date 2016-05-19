#!/usr/bin/env python
#  *-* coding: cp850 *-*

from distutils.core         import setup
from distutils.extension    import Extension
from Cython.Distutils       import build_ext


setup( 
    cmdclass = { 'build_ext': build_ext },
    ext_modules = [
        Extension( name = "TPSCy",
           sources = [ "TPSCy.pyx" ],
           libraries = [ 'm', 'gomp', 'pthread' ],
           extra_compile_args = [ '-O3', '-fopenmp', '-fdce', '-ffast-math', '-Wfatal-errors', '-w' ],
           extra_link_args = []
        )
    ]
 )
