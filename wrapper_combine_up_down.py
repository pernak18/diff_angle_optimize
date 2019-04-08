#!/usr/bin/env python

from __future__ import print_function

import os, sys, argparse, glob, shutil
import numpy as np

# local library
# have to do the * import so all necessary classes are imported
from combine_up_down import *

class combineBandmerge:
  def __init__(self, inDict):
    """
    Combine up and down solution for optimized diffusivity angles
    for each band (1 file per band), then merge all of the band 
    results together into a single merged file
    """

    self.templateNC = inDict['template_nc']
    self.outDir = inDict['out_dir']

    self.exe = 'lw_solver_opt_angs'
    self.refNC1 = 'rrtmgp-lw-inputs-outputs-clear.nc'
    self.refNC3 = 'rrtmgp-lw-inputs-outputs-clear-ang-3.nc'

    # make sure all of the inputs exist
    inPaths = ['templateNC', 'outDir', 'exe', 'refNC1', 'refNC3']
    for path in inPaths: utils.file_check(getattr(self, path))

    self.mergeNC = inDict['merge_file']

    # grab up and down compressed NumPy files generated for each 
    # band by angle_optimize.py
    self.upFiles = sorted(glob.glob('%s/*.npz' % inDict['up_dir']))
    self.downFiles = \
      sorted(glob.glob('%s/*.npz' % inDict['down_dir']))
    nUp = len(self.upFiles); nDown = len(self.downFiles)

    # 16 is kind of a magic number, but i think we'll always have 
    # that many LW bands
    self.nBands = 16
    if nUp != self.nBands or nDown != self.nBands:
      sys.exit('Inconsistent numbers of .npz files')

    self.bands = range(1, self.nBands+1)
    self.rank = inDict['rank']

    # for heating rate calculations
    self.heatFactor = 8.4391
  # end constructor

  def combineUpDown(self):
    """
    For each band, find the combined up-at-TOA and down-at-surface 
    solutions into one data set, then fit a polynomial to the combined
    data set
    """

    self.allOutNC = []
    for uFile, dFile, iBand in \
      zip(self.upFiles, self.downFiles, self.bands):

      base = os.path.basename(uFile)
      outPNG = 'up_down_sec_T_lin_band%02d.png' % iBand
      outNC = '%s/rrtmgp-inputs-outputs_opt_ang_band%02d.nc' % \
        (self.outDir, iBand)
      self.allOutNC.append(outNC)

      # input dictionary for combinedSolution object
      inDict = {}
      inDict['up_npz'] = uFile
      inDict['down_npz'] = dFile
      inDict['reference_nc_1ang'] = self.refNC1
      inDict['reference_nc_3ang'] = self.refNC3
      inDict['exe_rrtmgp'] = self.exe
      inDict['band'] = iBand
      inDict['out_png'] = '%s/%s' % (self.outDir, outPNG)
      inDict['out_nc'] = outNC
      inDict['rank'] = self.rank

      # work with the combinedSolution object (from combine_up_down)
      cObj = combinedSolution(inDict)
      cObj.upDownComponents()
      cObj.mergeUpDown()
      cObj.plotUpDown()
      cObj.plotCompErr()
      cObj.runRRTMGP()

      # and diffusivity angle array to output file for all bands
      with nc.Dataset(inDict['out_nc'], 'r+') as outObj:
        # coefficients of the fit, which for now is assumed to be no 
        # greater than the default of rank 3
        outObj.createDimension('coeffs', 2)
        secFit = outObj.createVariable('secant_fit', float, \
          ('band', 'coeffs'))
        secFit.description = 'Coefficients of fit to secant vs. ' + \
          'transmittance curve. Highest order is first.'
        secFit[iBand-1, :] = cObj.coeffs
      # endwith
    # end band loop
  # end combineUpDown

  def bandmerge(self):
    """
    simple bandmerge: combine flux arrays, calculate corresponding 
    heating rates
    """

    print('Merging band results')

    bandFluxUp, bandFluxDown, bandFluxNet, bandHR, bandSecFit = \
      [], [], [], [], []

    # read NC file for each band and calculate and save fluxes for it
    for iBand, ncf in enumerate(self.allOutNC):
      with nc.Dataset(ncf, 'r') as ncObj:
        # need to find a way to get better precision
        # these are the same for every band
        pLev = np.array(ncObj.variables['p_lev'])/100.0
        pLay = np.array(ncObj.variables['p_lay'])/100.0
        bandLims= np.array(ncObj.variables['band_lims_wvn'])
        i1, i2 = np.array(ncObj.variables['band_lims_gpt'])[iBand] - 1
        iArr = np.arange(i1, i2+1)

        # convert from g-point to band fluxes
        bandUpG = np.array(ncObj.variables['gpt_flux_up'])[iArr,:,:]
        bandUp = bandUpG.sum(axis=0)
        bandFluxUp.append(bandUp)

        bandDownG = np.array(ncObj.variables['gpt_flux_dn'])[iArr,:,:]
        bandDown = bandDownG.sum(axis=0)
        bandFluxDown.append(bandDown)

        bandNet = bandUp-bandDown
        bandFluxNet.append(bandNet)

        bandHR.append(\
          np.diff(bandNet, axis=0)/np.diff(pLev, axis=0) / \
          86400 * self.heatFactor)
          
        # store the fitted polynomial coefficients for this band
        bandSecFit.append(\
          np.array(ncObj.variables['secant_fit'])[iBand, :])
      # end netCDF object
    # end netCDF loop

    # output dimensions should be lev x prof x band
    outDims = (1,2,0)
    bandFluxUp = np.transpose(np.array(bandFluxUp), axes=outDims)
    bandFluxDown = np.transpose(np.array(bandFluxDown), axes=outDims)
    bandFluxNet = np.transpose(np.array(bandFluxNet), axes=outDims)
    bandHR = np.transpose(np.array(bandHR), axes=outDims)
    bandSecFit = np.array(bandSecFit)

    # variables that will go into merged netCDF
    outVars = ['band_flux_up', 'band_flux_dn', 'band_flux_net', \
      'band_heating_rate', \
      'flux_up', 'flux_dn', 'flux_net', 'heating_rate', \
      'p_lev', 'p_lay', 'band_lims_wvn', 'secant_fit']
    outDat = [bandFluxUp, bandFluxDown, bandFluxNet, bandHR, \
      bandFluxUp.sum(axis=2), bandFluxDown.sum(axis=2), \
      bandFluxNet.sum(axis=2), bandHR.sum(axis=2), pLev, pLay, \
      bandLims, bandSecFit]
    outDim = [('lev', 'col', 'band'), ('lev', 'col', 'band'), \
      ('lev', 'col', 'band'), ('lay', 'col', 'band'), \
      ('lev', 'col'), ('lev', 'col'), ('lev', 'col'), \
      ('lay', 'col'), \
      ('lev', 'col'), ('lay', 'col'), ('band', 'pair'), \
      ('band', 'pair')]

    with nc.Dataset(self.mergeNC, 'w') as ncObj:
      # since we're starting from an empty file, we need to 
      # define the dimensions that we'll be using for the arrays
      ncObj.createDimension('col', 42)
      ncObj.createDimension('band', 16)
      ncObj.createDimension('lev', 43)
      ncObj.createDimension('lay', 42)
      ncObj.createDimension('pair', 2)

      # now write the relevant variables (what are needed for 
      # LBLRTM_RRTMGP_compare.py to work) to bandmerged netCDF
      for outv, outd, dim in zip(outVars, outDat, outDim):
        ncVar = ncObj.createVariable(outv, float, dim)
        ncVar[:] = outd
      # end out array loop
    # end nc write

    print('Wrote %s' % self.mergeNC)
  # end bandmerge
# end combineBandmerge

if __name__ == '__main__':
  parser = argparse.ArgumentParser(\
    description='Call the up/down combine script for multiple ' + \
    'bands, then merge the bands together. THIS IS RESEARCH-GRADE ' + \
    'CODE AND NOT READY FOR PRIMETIME!')
  parser.add_argument('--up_dir', '-u', type=str, \
    default='band_calculations/flux_weights/up', \
    help='Directory with upwelling solutions (.npz files)')
  parser.add_argument('--down_dir', '-d', type=str, \
    default='band_calculations/flux_weights/down', \
    help='Directory with downwelling solutions (.npz files)')
  parser.add_argument('--out_dir', '-o', type=str, \
    default='band_calculations/flux_weights/combined', \
    help='Path to directory to which the PNG and netCDF files ' + \
    'that are generated with this wrapper are moved.')
  parser.add_argument('--template_nc', '-t', type=str, \
    default='rrtmgp-lw-inputs-outputs-default.nc', \
    help='RRTMGP broadband netCDF file that will be used as a ' + \
    'template.')
  parser.add_argument('--merge_file', '-m', type=str, \
    default='rrtmgp-inputs-outputs_opt_ang_merged.nc', 
    help='Name of output netCDF that will contain merged band ' + \
    'fluxes, heating rates, and optimization angles.')
  parser.add_argument('--rank', '-r', type=int, default=3, \
    help='Rank of polynomial to use in fit to combined solution.')
  args = parser.parse_args()

  cbm = combineBandmerge(vars(args))
  cbm.combineUpDown()
  cbm.bandmerge()
# end main()

