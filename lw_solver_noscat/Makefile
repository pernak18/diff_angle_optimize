#
# Meant to be called as a sub-make; relies on variables set in ../../Makefile
#
include $(RRTMGP_BUILD)/Makefile.rules

VPATH = ./:$(RRTMGP_BUILD):$(RRTMGP_ROOT)/examples:$(RRTMGP_ROOT)/extensions:$(TEST_ROOT)/test/util/src/

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

test_lw_solver: $(ADDITIONS) test_lw_solver.o

test_lw_solver.o: $(ADDITIONS) test_lw_solver.F90

mo_test_files_io.o: mo_simple_netcdf.o mo_test_files_io.F90

clean:
	-rm ../test_lw_solver . *.o *.optrpt ../*.optrpt *.mod
