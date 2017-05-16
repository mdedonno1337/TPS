#!/usr/bin/env python
#  *-* coding: cp850 *-*

################################################################################
#    Image
################################################################################

# Image resolution
CONF_res = 500

# Number of cores used in the image projection process
CONF_ncores = 8

# Number of pixels to add as margin to the images
CONF_dm = 5

# Default zone of interest
CONF_minx = 0
CONF_maxx = 25.4
CONF_miny = 0
CONF_maxy = 25.4

# Steps
CONF_majorstep = 1
CONF_minorstep = 0.02

# Delta arround
CONF_dm = 5

################################################################################
#    Reverting
################################################################################

# Grid used in the revert function
CONF_gridSize = 0.75

# Use the weights of the TPS parameters to select the points of interest
CONF_useWeights = True
# Limit to include a point in the interest list
CONF_weightsLimit = 0.003

# Number of random points to add to the revert grid
CONF_nbRandom = 20
