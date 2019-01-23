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
LDFLAGS   += -L$(RRTMGP_BUILD) -L/nas/project/p1770/dependencies/lib
LIBS      += -lrrtmgp -lrte -lnetcdff -lnetcdf
FCINCLUDE += -I$(RRTMGP_BUILD)

#
# Extra sources -- extensions to RRTMGP classes, shared infrastructure, local sources
#
ADDITIONS = mo_simple_netcdf.o mo_test_files_io.o mo_fluxes_bygpoint.o

#
# Targets
#

all: test_lw_solver_noscat_1ang test_lw_solver_noscat_opt_angs

test_lw_solver_noscat_1ang: $(ADDITIONS) test_lw_solver_noscat_1ang.o

test_lw_solver_noscat_1ang.o: $(ADDITIONS) test_lw_solver_noscat_1ang.F90

test_lw_solver_noscat_opt_angs: $(ADDITIONS) test_lw_solver_noscat_opt_angs.o

test_lw_solver_noscat_opt_angs.o: $(ADDITIONS) test_lw_solver_noscat_opt_angs.F90

mo_test_files_io.o: mo_simple_netcdf.o mo_test_files_io.F90

clean:
	-rm *.o *.optrpt ../*.optrpt *.mod
