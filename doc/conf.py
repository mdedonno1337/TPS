#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert( 0, os.path.abspath( "./" ) )

################################################################################

project = u'TPS'
copyright = u'2016-2017, Marco De Donno'
author = u'Marco De Donno'

try:
    from version import __version__
except:
    __version__ = "dev"

version = __version__
release = __version__

################################################################################

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx_git',
    'sphinx.ext.viewcode'
]

templates_path = [ '_templates' ]

source_suffix = '.rst'

master_doc = 'index'

language = None

exclude_patterns = ['_build']

show_authors = True

pygments_style = 'sphinx'

todo_include_todos = False

################################################################################
#    Options for HTML output
################################################################################

html_theme = 'classic'

html_theme_options = {}

html_static_path = [ '_static' ]

html_last_updated_fmt = '%b %d, %Y'

html_show_sourcelink = True

html_show_sphinx = False

htmlhelp_basename = 'TPSdoc'

################################################################################
#    Options for LaTeX output
################################################################################

latex_elements = {
    'papersize': 'a4paper'
}

latex_documents = [
   ( 
        master_doc,
        'TPS.tex',
        u'TPS Documentation',
        u'Marco De Donno',
        'howto'
    ),
]

latex_show_pagerefs = True

latex_show_urls = 'true'

################################################################################
#    Options for manual page output
################################################################################

man_pages = [
    ( 
        master_doc,
        'tps',
        u'TPS Documentation',
        [author],
        1
    )
]

man_show_urls = True
