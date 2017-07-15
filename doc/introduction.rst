Introduction
############

This library is a direct implementation of the article of Bookstein [Bookstein1989]_.

This library is composed of a Python and a Cython version of all functions. The Python version have been coded for simplicity and portability, and the Cython version for performance issues.

The main wrapper will select the TPS core module automatically (the Cython version if available, otherwise the Python one). This module offer also a common API for all core functions (the Python and Cython function could be called directly, but not recommended).

The image projection can only be used if the Cython module is loaded (to heavy for the Python version).

The Cython module is composed of Cython wrapper functions, and Pure-C core functions (not callable from Python, but only from the Cython wrapper functions).

References
~~~~~~~~~~

.. [Bookstein1989] Bookstein, F. L. (1989). Principal warps: Thin-plate splines and the decomposition of deformations. IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 11 (6), pp. 567--585
