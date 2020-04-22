#
# Meant to be called as a sub-make; relies on variables set in ../../Makefile
#
# This version of this suite of code must be attached a version of RRTMGP
# updated to include the optimal angle solution.
RRTMGP_ROOT = /project/p2130/git/rte-rrtmgp-add-alternate-angle-2
RRTMGP_BUILD = $(RRTMGP_ROOT)/build
TEST_ROOT = /project/p2130/git/rte-rrtmgp-test-val-add-alternate-angle-2

include $(RRTMGP_BUILD)/Makefile.conf $(TEST_ROOT)/test/Makefile.libs

# Look for executables in these directores
VPATH = ./:$(RRTMGP_BUILD):$(RRTMGP_ROOT)/examples/:$(RRTMGP_ROOT)/extensions/:$(TEST_ROOT)/test/util/src

#
# RRTMGP library, module files
#
FCFLAGS += -I${RRTMGP_BUILD} -I${NFHOME}/include -I${NCHOME}/include
LDFLAGS += -L${RRTMGP_BUILD} -L${NFHOME}/lib     -L${NCHOME}/lib -lrte -lrrtmgp -lnetcdff -lnetcdf

# Compilation rules
COMPILE = $(FC) $(FCFLAGS) $(FCINCLUDE) -c
%.o: %.F90
	$(COMPILE) $<
%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)
#
# Extra sources -- extensions to RRTMGP classes, shared infrastructure, local sources
#
ADDITIONS = mo_simple_netcdf.o mo_test_files_io.o mo_fluxes_bygpoint.o

#
# Targets
#

all: lw_solver_opt_angs

#lw_solver_1ang: $(ADDITIONS) lw_solver_1ang.o

#lw_solver_1ang.o: $(ADDITIONS) lw_solver_1ang.F90

lw_solver_opt_angs: $(ADDITIONS) lw_solver_opt_angs.o

lw_solver_opt_angs.o: $(ADDITIONS) lw_solver_opt_angs.F90

mo_test_files_io.o: mo_simple_netcdf.o mo_test_files_io.F90

clean:
	-rm *.o *.optrpt ../*.optrpt *.mod
