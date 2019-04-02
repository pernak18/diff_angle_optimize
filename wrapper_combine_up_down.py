#!/usr/bin/env python

from __future__ import print_function

import os, sys, argparse, glob
import numpy as np

# local library
# have to do the * import so all necessary classes are imported
from combine_up_down import *
from bandmerge import *

parser = argparse.ArgumentParser(\
  description='Call the up/down combine script for multiple ' + \
  'bands, then merge the bands together.')
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
args = parser.parse_args()

upFiles = sorted(glob.glob('%s/*.npz' % args.up_dir))
downFiles = sorted(glob.glob('%s/*.npz' % args.down_dir))
nUp = len(upFiles); nDown = len(downFiles)

if nUp != 16 or nDown != 16:
  sys.exit('Inconsistent numbers of .npz files')

outDir = args.out_dir; utils.file_check(outDir)
bands = range(1, 17)

# some constants over all bands
exe = 'lw_solver_opt_angs'
refNC = 'rrtmgp-lw-inputs-outputs-clear.nc'

# for each band, find the combined up-at-TOA/down-at-surface solution
for uFile, dFile, iBand in zip(upFiles, downFiles, bands):
  if iBand != 2: continue
  base = os.path.basename(uFile)
  print(uFile)
  outPNG = 'up_down_sec_T_lin_band%02d.png' % iBand
  outNC = 'rrtmgp-inputs-outputs_opt_ang_band%02d.nc' % iBand

  # input dictionary for combinedSolution object
  inDict = {}
  inDict['up_npz'] = uFile
  inDict['down_npz'] = dFile
  inDict['reference_nc'] = refNC
  inDict['exe_rrtmgp'] = exe
  inDict['by_band'] = True
  inDict['out_png'] = '%s/%s' % (outDir, outPNG)
  inDict['out_nc'] = '%s/%s' % (outDir, outNC)

  # work with the combinedSolution object (from combine_up_down)
  cObj = combinedSolution(inDict)
  cObj.mergeUpDown()
  cObj.plotUpDown()
  cObj.runRRTMGP()

  # and diffusivity angle array to output file for this band
  with nc.Dataset('optimized_secants.nc', 'r') as secObj, \
    nc.Dataset(inDict['out_nc'], 'r+') as outObj:

    outVar = outObj.createVariable(\
      'diff_angle_g', float, ('gpt', 'col'))
    outVar.units = 'Degrees'
    outVar.description = \
      'Optimized diffusivity secant(angle) for flux calculations'
    outVar[:] = np.array(secObj.variables['secant'])
  # endwith
  sys.exit('1 Band')
# end band loop

# merge bands of up/down combined solutions
mergeDict = {}
mergeDict['reference_nc'] = refNC
mergeDict['in_dir'] = outDir
mergeDict['outfile'] = 'rrtmgp-inputs-outputs_opt_ang_merged.nc'
mergeDict['mv'] = False
mergeDict['notebook_path'] = os.devnull
bObj = bandMerge(mergeDict)
bObj.calcBands()
