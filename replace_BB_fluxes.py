#!/usr/bin/env python

import os, sys, argparse
import numpy as np
import netCDF4 as nc

# git submodule
sys.path.append('common')
import utils

parser = argparse.ArgumentParser(\
  description='Replace 1-angle flux calculations ' + \
  'with the fluxes computed angle_optimize.py.')
parser.add_argument('--reference_nc', '-r', type=str, \
  default='rrtmgp-lw-inputs-outputs-default.nc', \
  help='netCDF file with fluxes. Used in RRTMGP runs. It is ' + \
  'overwritten with new up and down broadband fluxes.')
parser.add_argument('--optmized_nc', '-o', type=str, \
  default='rrtmgp-inputs-outputs_opt_ang_weighted_rel.nc', \
  help='netCDF generated with angle_optimize.py module. It ' + \
  'contains downwelling and upwelling g-point spectral fluxes.')
args = parser.parse_args()

refNC = args.reference_nc; utils.file_check(refNC)
optNC = args.optimized_nc; utils.file_check(optNC)

# overwrite current reference broadband fluxes with optimized 
with nc.Dataset(optNC, 'r') as optObj, \
  nc.Dataset(refNC, 'r+') as refObj:

  optFlux = np.array(optObj.variables['gpt_flux_dn'])
  refFlux = np.array(refObj.variables['flux_dn'])

  # downwelling flux
  # convert from spectral to broadband
  optBB = optFlux.sum(axis=0)
  flux = refObj.variables['flux_dn']
  flux[:] = np.array(optBB)

  optFlux = np.array(optObj.variables['gpt_flux_up'])
  refFlux = np.array(refObj.variables['flux_up'])

  # upwelling
  # convert from spectral to broadband
  optBB = optFlux.sum(axis=0)
  flux = refObj.variables['flux_up']
  flux[:] = np.array(optBB)
# end with

