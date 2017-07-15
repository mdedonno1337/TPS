################################################################################
###		Global variables

MODNAME = TPSCy

################################################################################
###		Determination of the system

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	OPTS =
	EXT = so

else
	OPTS = --compiler=mingw32
	EXT = pyd

endif

################################################################################
###		Rules

all: clean c $(EXT)

c:
	cython $(MODNAME).pyx

$(EXT) gcc compile:
	python setup.py build_ext --inplace $(OPTS)

clean clear cls:
	-rm $(MODNAME).$(EXT)
	-rm $(MODNAME).c
	-rm $(MODNAME).html
	-rm -rf ./build

a annotate html:
	cython -a $(MODNAME).pyx
