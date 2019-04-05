#!/usr/bin/env python

import os, sys, pickle, argparse, shutil
import numpy as np
import matplotlib.pyplot as plot
import subprocess as sub

# Git submodule
sys.path.append('common')
import utils

# local module
from angle_optimize import *

class combinedSolution:
  def __init__(self, inDict):
    """
    Combine arrays from angle_optimize.py runs for up and down flux, 
    then run the fitting routines to calculate optimized angle for 
    combined up (TOA) and down (surface).

    Input
      inDict -- dictionary with inputs from main()
    """

    self.upNPZ = inDict['up_npz']
    self.downNPZ = inDict['down_npz']
    self.refNC = inDict['reference_nc']
    self.exe = inDict['exe_rrtmgp']

    if inDict['band']:
      self.band = inDict['band']-1
    else:
      self.band = None
    # endif band

    self.outPNG = inDict['out_png']
    self.outNC = inDict['out_nc']

    inPaths = [self.upNPZ, self.downNPZ, self.refNC, self.exe]
    for f in inPaths: utils.file_check(f)

    # each g-point has a weight. this is in the literature but 
    # was also provided by Eli in 
    # https://rrtmgp2.slack.com/archives/D942AU7QE/p1550264049021200
    # these will only be used if we're doing the by-band calculations
    self.gWeights = np.array([0.1527534276, 0.1491729617, \
      0.1420961469, 0.1316886544,  0.1181945205,  0.1019300893, \
      0.0832767040, 0.0626720116, 0.0424925000,0.0046269894, \
      0.0038279891, 0.0030260086, 0.0022199750, 0.0014140010, \
      0.0005330000, 0.0000750000])

    # eventually, this is gonna have to be more flexible, but for now
    # every band has 16 g-points and we're only doing Garand profiles
    # except i don't think ./lw_solver_opt_angs will work with 16
    self.nGpt = 16 if self.band is not None else 256
    self.nGpt = 256
    self.nProf = 42

    self.rank = inDict['rank']
  # end constructor

  def mergeUpDown(self):
    """
    Combine optimized diffusivity angle solutions for upwelling and 
    downwelling
    """

    # unpack the dictionaries from compressed NumPy files (.npz)
    # in this context, we mean "combined over all angles" for either
    # downwelling or upwelling
    upDat = np.load(self.upNPZ)
    downDat = np.load(self.downNPZ)
    upComb = upDat['combined'].item()
    dnComb = downDat['combined'].item()

    # points used for fitting
    self.upTran = upComb.transmittance
    self.upRoots = upComb.rootsErrAng
    self.upWeights = upComb.weights
    self.dnTran = dnComb.transmittance
    self.dnRoots = dnComb.rootsErrAng
    self.dnWeights = dnComb.weights

    # combine up and down solutions
    allTran = np.append(self.upTran, self.dnTran)
    allRoots = np.append(self.upRoots, self.dnRoots)
    allWeights = np.append(self.upWeights, self.dnWeights)

    # grab flux errors from new optimized angle calculations
    # we want separate up and down errors
    # no fitting is done with these
    upOpt = upDat['optAng'].item()
    self.xErrUp, self.yErrUp = upOpt.transmittance, upOpt.err
    dnOpt = downDat['optAng'].item()
    self.xErrDn, self.yErrDn = dnOpt.transmittance, dnOpt.err

    # sort by transmittance
    iSort = np.argsort(allTran)
    self.allTran, self.allRoots, self.allWeights = \
      allTran[iSort], allRoots[iSort], allWeights[iSort]

    # compute fit to combined up and down solution save for later
    coeffs = np.polyfit(self.allTran, self.allRoots, self.rank, \
      w=self.allWeights)
    self.secTFit = np.poly1d(coeffs)
  # end mergeUpDown()

  def plotCompErr(self):
    """
    plot error component -- i.e. flux differences for each of the 
    separate up and down solutions
    """

    outErrPNG = 'opt_ang_up_down_errors.png' if self.band is None \
      else 'opt_ang_up_down_errors_band%02d.png' % (self.band+1)

    # now lets just try to do what we do in combineErr.fitAngT
    plot.plot(self.xErrUp, self.yErrUp, 'ro')
    plot.plot(self.xErrDn, self.yErrDn, 'bo', alpha=0.5)

    # horizontal zero line
    plot.gca().axhline(0, linestyle='--', color='k')

    plot.legend(['Up', 'Down'], numpoints=1, loc='best')
    plot.xlabel('Transmittance')
    plot.ylabel(r'$(F_{1-angle}-F_{3-angle})/F_{3-angle}$')
    plot.savefig(outErrPNG)
    plot.close()
  # end plotUpDownErr()

  def plotUpDown(self):
    """
    plot combined solution (fitted curve to up and down root vs. 
    transmittance)
    """

    # now lets just try to do what we do in combineErr.fitAngT
    plot.plot(self.upTran, self.upRoots, 'r.')
    plot.plot(self.dnTran, self.dnRoots, 'b.', alpha=0.5)
    plot.plot(self.allTran, self.secTFit(self.allTran), 'c--')
    plot.legend(['Up', 'Down'], numpoints=1, loc='best')
    plot.xlabel('Transmittance')
    plot.ylabel('secant(roots)')
    plot.savefig(self.outPNG)
    plot.close()
  # end plotUpDown()

  def runRRTMGP(self):
    """
    run RRTMGP code to calculate new fluxes with combined solution
    """

    # write netCDF for usage in RRTMGP executable that calculates 
    # g-point fluxes at different angles
    # hard coding filenames for now, since i've edited 
    # lw_solver_noscat code to look specifically for 
    # 'optimized_secants.nc'
    with nc.Dataset('optimized_secants.nc', 'w') as ncObj, \
      nc.Dataset(self.refNC, 'r') as origObj:

      od = np.array(origObj.variables['tau']).sum(axis=1)

      if self.band is not None:
        bandInds = np.array(origObj.variables['band_lims_gpt'])
        self.bandInd1, self.bandInd2 = bandInds[self.band, :] - 1

        """
        # using this weird sequence of transposes because 
        # `self.gWeights * od[bandInd1:bandInd2+1, :]` doesn't work, 
        # and i want an nGpt x nProfile array
        od = self.gWeights * od[bandInd1:bandInd2+1, :].T
        od = od.T
        """
      # endif band

      origTran = np.exp(-od)

      ncObj.description = 'Secant values for every g-point of ' + \
        'every (Garand) profile to fluxErr class in ' + \
        'angle_optimize.py for which errors between a 1-angle ' + \
        'test RRTGMP run and a 3-angle reference RRTMGP run were ' + \
        'calcualated. These are the angles at which the errors ' + \
        'were minimized.'
      ncObj.createDimension('profiles', self.nProf)
      ncObj.createDimension('gpt', self.nGpt)
      secant = ncObj.createVariable(\
        'secant', float, ('gpt', 'profiles'))
      secant.units = 'None'
      secant.description = 'Optimized secants for flux calculations'
      secant[:] = np.array(self.secTFit(origTran))
    # endwith

    self.secantOpt = self.secTFit(origTran)

    # stage input file expected by RRTMGP executable
    inNC = 'rrtmgp-inputs-outputs.nc'
    if not os.path.exists(inNC): shutil.copyfile(self.refNC, inNC)

    # run the executable with optimized_secants.nc and refNC
    sub.call([self.exe])

    # move newly populated netCDF to a different name so we don't 
    # overwrite it in a subsequent RRTMGP run
    os.rename(inNC, self.outNC)
    print('Wrote %s' % self.outNC)
  # end runRRTMGP()
# end combinedSolution

if __name__ == '__main__':
  parser = argparse.ArgumentParser(\
    description='Combine arrays from angle_optimize.py runs for ' + \
    'up and down flux, then run the fitting routines to calculate ' + \
    'optimized angle for combined up (TOA) and down (surface).')
  parser.add_argument('--up_npz', '-u', type=str, \
    default='save_files/secantRecalc_inputs_up.npz', \
    help='Compressed NumPy file with up angle_optimize.py objects.')
  parser.add_argument('--down_npz', '-d', type=str, \
    default='save_files/secantRecalc_inputs_dn.npz', \
    help='Compressed NumPy file with down angle_optimize.py objects.')
  parser.add_argument('--reference_nc', '-r', type=str, \
    default='rrtmgp-lw-inputs-outputs-clear.nc', \
    help='RRTMGP by-g-point netCDF file that will be used as a ' + \
    'template.')
  parser.add_argument('--exe_rrtmgp', '-e', type=str, \
    default='lw_solver_opt_angs', \
    help='RRTMGP executable that calculates g-point fluxes for ' + \
    'angles determined with this code and atmospheric states ' + \
    '(and other parameters) from reference_nc.')
  parser.add_argument('--band', '-b', type=int, \
    help='Combine up and down solutions for this band ' + \
    'only (unit offset).')
  parser.add_argument('--out_png', '-op', type=str, \
    default='up_down_sec_T_lin.png', help='Path for plot file.')
  parser.add_argument('--out_nc', '-on', type=str, \
    default='rrtmgp-inputs-outputs_opt_ang.nc', \
    help='Path for netCDF file with fluxes computed with ' + \
    'optimized up+down solution.')
  parser.add_argument('--rank', '-r', type=int, default=3, \
    help='Rank of polynomial to use in fit to combined solution.')
  args = parser.parse_args()

  cObj = combinedSolution(vars(args))
  cObj.mergeUpDown()
  cObj.plotUpDown()
  cObj.runRRTMGP()
# end main()

