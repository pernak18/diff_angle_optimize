#!/usr/bin/env python

from __future__ import print_function

import os, sys, argparse, glob, shutil
import numpy as np

# local library
# have to do the * import so all necessary classes are imported
from combine_up_down import *

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

templateNC = args.template_nc; utils.file_check(templateNC)
mergeNC = args.merge_file

upFiles = sorted(glob.glob('%s/*.npz' % args.up_dir))
downFiles = sorted(glob.glob('%s/*.npz' % args.down_dir))
nUp = len(upFiles); nDown = len(downFiles)

if nUp != 16 or nDown != 16:
  sys.exit('Inconsistent numbers of .npz files')

outDir = args.out_dir; utils.file_check(outDir)
bands = range(1, 17)

# some constants over all bands
exe = 'lw_solver_opt_angs'
refNC1 = 'rrtmgp-lw-inputs-outputs-clear.nc'
refNC3 = 'rrtmgp-lw-inputs-outputs-clear-ang-3.nc'

# stack solutions into an array that will eventually be 
# (nGpt x nProfile), 256 x 42
secantOpt = np.zeros((256, 42)) * np.nan

# for each band, find the combined up-at-TOA/down-at-surface solution
allOutNC = []
for uFile, dFile, iBand in zip(upFiles, downFiles, bands):
  base = os.path.basename(uFile)
  outPNG = 'up_down_sec_T_lin_band%02d.png' % iBand
  outNC = '%s/rrtmgp-inputs-outputs_opt_ang_band%02d.nc' % \
    (outDir, iBand)
  allOutNC.append(outNC)

  # input dictionary for combinedSolution object
  inDict = {}
  inDict['up_npz'] = uFile
  inDict['down_npz'] = dFile
  inDict['reference_nc_1ang'] = refNC1
  inDict['reference_nc_3ang'] = refNC3
  inDict['exe_rrtmgp'] = exe
  inDict['band'] = iBand
  inDict['out_png'] = '%s/%s' % (outDir, outPNG)
  inDict['out_nc'] = outNC
  inDict['rank'] = args.rank

  # work with the combinedSolution object (from combine_up_down)
  cObj = combinedSolution(inDict)
  cObj.upDownComponents()
  cObj.mergeUpDown()
  cObj.plotUpDown()
  cObj.plotCompErr()
  cObj.runRRTMGP()

  # only grab slice of array that corresponds to this band
  bandSecOpt = cObj.secantOpt[cObj.bandInd1: cObj.bandInd2+1, :]
  secantOpt[cObj.bandInd1: cObj.bandInd2+1, :] = bandSecOpt

  # and diffusivity angle array to output file for all bands
  with nc.Dataset(inDict['out_nc'], 'r+') as outObj:
    outVar = outObj.createVariable(\
      'diff_angle_g', float, ('gpt', 'col'))
    outVar.units = 'Degrees'
    outVar.description = \
      'Band-optimized diffusivity secant(angle) for flux calculations'
    outVar[:] = np.array(secantOpt)

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

# simple bandmerge -- no new calculations, just combine flux arrays
# start with copy of template, then replace flux, HR, and angle arrays
#shutil.copyfile(templateNC, mergeNC)
bandFluxUp, bandFluxDown, bandFluxNet, bandHR, bandSecFit = \
  [], [], [], [], []

heatFactor = 8.4391

# read NC file for each band and calculate and save fluxes for it
for iBand, ncf in enumerate(allOutNC):
  with nc.Dataset(ncf, 'r') as ncObj:
    # need to find a way to get better precision
    # these are the same for every band
    pLev = np.array(ncObj.variables['p_lev'])/100.0
    pLay = np.array(ncObj.variables['p_lay'])/100.0
    bandLims= np.array(ncObj.variables['band_lims_wvn'])
    i1, i2 = np.array(ncObj.variables['band_lims_gpt'])[iBand] - 1
    iArr = np.arange(i1, i2+1)

    bandUpG = np.array(ncObj.variables['gpt_flux_up'])[iArr,:,:]
    bandUp = bandUpG.sum(axis=0)
    bandFluxUp.append(bandUp)

    bandDownG = np.array(ncObj.variables['gpt_flux_dn'])[iArr,:,:]
    bandDown = bandDownG.sum(axis=0)
    bandFluxDown.append(bandDown)

    bandNet = bandUp-bandDown
    bandFluxNet.append(bandNet)

    bandHR.append(np.diff(bandNet, axis=0)/np.diff(pLev, axis=0) / \
      86400 * heatFactor)
      
    """
    ang = np.array(ncObj.variables['diff_angle_g'])[i1:i2+1,:]
    diffAng = np.array(ang) if iBand == 0 else \
      np.vstack( (diffAng, ang) )
    """
    bandSecFit.append(np.array(ncObj.variables['secant_fit'])[iBand, :])
  # end netCDF object
# end netCDF loop

# output dimensions should be lev x prof x band
outDims = (1,2,0)
bandFluxUp = np.transpose(np.array(bandFluxUp), axes=outDims)
bandFluxDown = np.transpose(np.array(bandFluxDown), axes=outDims)
bandFluxNet = np.transpose(np.array(bandFluxNet), axes=outDims)
bandHR = np.transpose(np.array(bandHR), axes=outDims)
bandSecFit = np.array(bandSecFit)

# now write the variables to bandmerged netCDF
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
  ('lev', 'col'), ('lev', 'col'), ('lev', 'col'), ('lay', 'col'), \
  ('lev', 'col'), ('lay', 'col'), ('band', 'pair'), ('band', 'pair')]

with nc.Dataset(mergeNC, 'w') as ncObj:
  # flux vars
  """
  for outv, outd in zip(outVars, outDat):
    ncVar = ncObj.variables[outv]
    ncVar[:] = outd
  # end out array loop
  """

  ncObj.createDimension('col', 42)
  ncObj.createDimension('band', 16)
  ncObj.createDimension('lev', 43)
  ncObj.createDimension('lay', 42)
  ncObj.createDimension('pair', 2)

  for outv, outd, dim in zip(outVars, outDat, outDim):
    ncVar = ncObj.createVariable(outv, float, dim)
    ncVar[:] = outd
  # end out array loop

  """
  # and diffusivity angle array to output file for all bands
  ncObj.createDimension('gpt', 256)
  outVar = ncObj.createVariable('diff_angle_g', float, ('gpt', 'col'))
  outVar.units = 'Degrees'
  outVar.description = \
    'Band-optimized diffusivity secant(angle) for flux calculations'
  outVar[:] = np.array(secantOpt)
  """
# end nc write

print('Wrote %s' % mergeNC)
sys.exit()

# merge bands of up/down combined solutions
mergeDict = {}
mergeDict['reference_nc'] = 'rrtmgp-lw-inputs-outputs-default.nc'
mergeDict['in_dir'] = outDir
mergeDict['outfile'] = 'rrtmgp-inputs-outputs_opt_ang_merged.nc'
mergeDict['mv'] = False
mergeDict['notebook_path'] = os.devnull
bObj = bandMerge(mergeDict)
bObj.calcBands()

