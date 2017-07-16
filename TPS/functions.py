#!/usr/bin/python
# -*- coding: UTF-8 -*-

from inspect import isfunction
import functools
import warnings

def deprecated( func = None, msg = None, *args ):
    if isfunction( func ):
        warnmsg = boxer( "Call to deprecated function %s" % func.__name__, msg )
        
        @functools.wraps( func )
        def f( *args, **kwargs ):
            warnings.simplefilter( 'always', DeprecationWarning )
            warnings.warn( warnmsg, category = DeprecationWarning, stacklevel = 2 )
            warnings.simplefilter( 'default', DeprecationWarning )
            
            return func( *args, **kwargs )
        
        return f
    else:
        if type( func ) == str:
            try:
                msg = " ".join( ( func, msg, ) + args ) 
            except:
                msg = func
        elif type( func ) == list:
            msg = " ".join( func )
        else:
            msg = msg

        return functools.partial( deprecated, msg = msg )

from textwrap import wrap

def boxer( doc, comp = None ):
    if comp != None:
        ret = "\n#        "
        
        doc += ret * 2
        doc += "\n".join( wrap( comp, 65 ) ).replace( "\n", ret )
    
    return """
############################################################################
#
#    %s
#
############################################################################
    """ % doc
