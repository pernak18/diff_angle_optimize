#
# Meant to be called as a sub-make; relies on variables set in ../../Makefile
#
RRTMGP_ROOT = $(RRTMGP)/RRTMGP_Git_Master
RRTMGP_BUILD = $(RRTMGP_ROOT)/build
SVN_ROOT = $(RRTMGP)/RRTMGP_SVN/trunk
include $(RRTMGP_BUILD)/Makefile.rules $(RRTMGP_BUILD)/Makefile.conf $(SVN_ROOT)/test/Makefile.libs

VPATH = ./:$(RRTMGP_BUILD):$(RRTMGP_ROOT)/examples:$(RRTMGP_ROOT)/extensions:$(SVN_ROOT)/test/util/src/

#
# RRTMGP library, module files
#
LDFLAGS   += -L$(RRTMGP_BUILD)
LIBS      += -lrrtmgp -lrte
FCINCLUDE += -I$(RRTMGP_BUILD)

#
# Extra sources -- extensions to RRTMGP classes, shared infrastructure, local sources
#
ADDITIONS = mo_simple_netcdf.o mo_test_files_io.o mo_fluxes_bygpoint.o

#
# Targets
#

test_lw_solver_noscat: $(ADDITIONS) test_lw_solver_noscat.o

test_lw_solver_noscat.o: $(ADDITIONS) test_lw_solver_noscat.F90

mo_test_files_io.o: mo_simple_netcdf.o mo_test_files_io.F90

clean:
	-rm *.o *.optrpt ../*.optrpt *.mod
