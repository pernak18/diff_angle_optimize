#!/usr/bin/env python

from __future__ import print_function

import os, sys, argparse
import numpy as np
import netCDF4 as nc
import shutil

# git submodule
sys.path.append('common')
import utils

class newRef:
  def __init__(self, inDict):
    """
    Replace 1-angle flux and HR calculations with the fluxes computed 
    angle_optimize.py

    Input
      inDict -- dictionary with the following keys:
        reference_nc -- string, netCDF file with fluxes. Used in 
          RRTMGP runs. It is overwritten with new up and down 
          broadband fluxes.
        optmized_nc -- string, netCDF generated with 
          angle_optimize.py module. It contains downwelling and 
          upwelling g-point spectral fluxes.
        mv -- boolean, move the new netCDF to the Jupyter notebook 
          directory for comparison with LBLRTM
        notebook_path -- string, move netCDF to this path if mv is set
    """

    self.refNC = inDict['reference_nc']; utils.file_check(self.refNC)
    self.optNC = inDict['optimized_nc']; utils.file_check(self.optNC)
    self.moveNC = inDict['mv']
    self.pathNB = inDict['notebook_path']

    # for heating rate calculations
    self.heatFactor = 8.4391
  # end constructor

  def extractVars(self):
    """
    Extract variables that will be necessary for by-band and broadband
    flux replacement
    """

    # overwrite current reference broadband fluxes with optimized 
    with nc.Dataset(self.optNC, 'r') as optObj, \
      nc.Dataset(self.refNC, 'r+') as refObj:
      self.optDown = np.array(optObj.variables['gpt_flux_dn'])
      self.optUp = np.array(optObj.variables['gpt_flux_up'])
      self.pLevel = np.array(refObj.variables['p_lev'])
      self.diffAngle = np.array(optObj.variables['diff_angle_g'])

      # want these indices to be zero-offset
      self.bandLimsG = np.array(optObj.variables['band_lims_gpt'])-1
    # end with

    self.nGpt, self.nLev, nProf = self.optDown.shape
  # end extractRef()

  def calcBands(self):
    """
    Convert from spectral g-point fluxes to by-band fluxes and heating
    rates. Also do broadband calculations
    """
    
    bandDown, bandUp, hrArg = [], [], []
    for iBand in self.bandLimsG:
      down = self.optDown[iBand[0]:iBand[1]+1, :, :].sum(axis=0)
      up = self.optUp[iBand[0]:iBand[1]+1, :, :].sum(axis=0)
      bandDown.append(down)
      bandUp.append(up)

      # i think for heating rates, pressures have to be in hPa (mbar)
      # but in RRTMGP they are in Pa
      hrArg.append(np.diff(down-up, axis=0) / \
        np.diff(self.pLevel/100, axis=0))
    # end band loop

    # band flux arrays should be nLev x nProf x nBand
    outDim = (1,2,0)
    self.downBand =np.transpose(np.array(bandDown), axes=outDim)
    self.upBand = np.transpose(np.array(bandUp), axes=outDim)
    self.netBand = self.downBand - self.upBand
    self.hrBand = self.heatFactor * \
      np.transpose(np.array(hrArg), axes=outDim)

    # now broadband
    self.downBB = self.downBand.sum(axis=2)
    self.upBB = self.upBand.sum(axis=2)
    self.netBB = self.netBand.sum(axis=2)
    self.hrBB = self.hrBand.sum(axis=2)
  # end calcBands()

  def fieldReplace(self):
    """
    Replace the flux and heating rate arrays in refNC with the fluxes
    in optNC
    """

    # overwrite current reference broadband fluxes with optimized 
    with nc.Dataset(self.refNC, 'r+') as refObj:

      # variables that will be replaced (netCDF variable names)
      refVars = ['band_flux_dn', 'band_flux_net', 'band_flux_up', \
        'flux_dn', 'flux_net', 'flux_up', \
        'band_heating_rate', 'heating_rate']

      # optimized calculations corresponding to refVars
      # (object attributes)
      optVars = ['downBand', 'netBand', 'upBand', \
        'downBB', 'netBB', 'upBB', \
        'hrBand', 'hrBB']

      for rVar, oVar in zip(refVars, optVars):
        ncVar = refObj.variables[rVar]
        ncVar[:] = getattr(self, oVar)
      # end var replacement
    # end with

    print('Replaced flux fields in %s' % self.refNC)

    if self.moveNC:
      shutil.copyfile(self.refNC, self.pathNB)
      print('%s copied to %s' % (self.refNC, self.pathNB))
    # endif move

  # end fieldReplace()
# end newRef

if __name__ == '__main__':
  parser = argparse.ArgumentParser(\
    description='Replace 1-angle flux calculations ' + \
    'with the fluxes computed angle_optimize.py.')
  parser.add_argument('--reference_nc', '-r', type=str, \
    default='rrtmgp-lw-inputs-outputs-default.nc', \
    help='netCDF file with fluxes. Used in RRTMGP runs. It is ' + \
    'overwritten with new up and down broadband fluxes.')
  parser.add_argument('--optimized_nc', '-o', type=str, \
    default='rrtmgp-inputs-outputs_opt_ang_weighted_rel.nc', \
    help='netCDF generated with angle_optimize.py module. It ' + \
    'contains downwelling and upwelling g-point spectral fluxes.')
  parser.add_argument('--mv', '-mv', action='store_true', \
    help='Move the reference netCDF with newly-calculated ' + \
    'fluxes to the Jupyter notebook directory.')
  parser.add_argument('--notebook_path', '-n', type=str, \
    default='/rd47/scratch/RRTMGP/paper/' + \
    'rte-rrtmgp-paper-figures/data/' + \
    'rrtmgp-lw-inputs-outputs-optAng.nc', \
    help='Path to file in Jupyter notebook that compares ' + \
    'RRTMGP results to reference results. The new reference ' + \
    'file will be moved to this destination if --mv is set.')
  args = parser.parse_args()

  newRefObj = newRef(vars(args))
  newRefObj.extractVars()
  newRefObj.calcBands()
  newRefObj.fieldReplace()
# end main()

