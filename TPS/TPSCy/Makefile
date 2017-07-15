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
###		Compilation

all: c $(EXT)

c: cleanc
	cython -V -v $(MODNAME).pyx

$(EXT) gcc compile: clean$(EXT)
	python setup.py build_ext --inplace $(OPTS)

a annotate html: cleanhtml
	cython -a $(MODNAME).pyx

################################################################################
###		Cleaning functions

cleanc:
	-rm $(MODNAME).c

clean$(EXT):
	-rm $(MODNAME).$(EXT)

cleanhtml:
	-rm $(MODNAME).html

cleanbuild:
	-rm -rf ./build

clean clear cls:
	$(MAKE) cleanc
	$(MAKE) cleanhtml
	$(MAKE) cleanbuild
