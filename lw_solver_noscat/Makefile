#
# Meant to be called as a sub-make; relies on variables set in ../../Makefile
#
RRTMGP_ROOT = $(RRTMGP)/obsolete/RRTMGP_Git_Master
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

all: lw_solver_opt_angs

lw_solver_1ang: $(ADDITIONS) lw_solver_1ang.o

lw_solver_1ang.o: $(ADDITIONS) lw_solver_1ang.F90

lw_solver_opt_angs: $(ADDITIONS) lw_solver_opt_angs.o

lw_solver_opt_angs.o: $(ADDITIONS) lw_solver_opt_angs.F90

mo_test_files_io.o: mo_simple_netcdf.o mo_test_files_io.F90

clean:
	-rm *.o *.optrpt ../*.optrpt *.mod
