#!/usr/bin/python

from __future__ import print_function

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.font_manager as font
import netCDF4 as nc
import subprocess as sub
import shutil
from scipy import interpolate as INTER

# Gitlab submodule
sys.path.append('common')
import utils

# for plots
font_prop = font.FontProperties()
font_prop.set_size(8)

class fluxErr():
  def __init__(self, inAng, profNum, inDict):
    """
    For a given profile and angle:
      - grab reference fluxes and optical depths (tau)
        at each of the 256 g-points
      - calculate t = exp(-tau) and use it as a proxy for opacity
      - run RRTMGP over many single angles spanning 45-65 degrees -- 
        the diffusivity angle is ~53 degrees, and we wanna have a 
        fair representation of angles on each side of this, so the 
        range can be larger. these will be the test fluxes
      - for every g-point, calculate the ref-test error as a function 
        of t

    Input
      inAng -- int, angle at which RT calculation is done
      profNum -- int, profile number
      relative_err -- boolean, plot relative instead of absolute 
        flux differences
      inDict -- dictionary with the following keys:
        reference -- str, netCDF with 3-angle reference fluxes
        flux_str -- str, netCDF variable name of fluxes to compare
        executable -- str, path to flux calculation executable to run
        template_nc -- str, path to netCDF used for executable I/O
        test_dir -- str, path to which the results from RRTMGP runs 
          will be written
    """

    self.refNC = str(inDict['reference'])
    utils.file_check(self.refNC)

    self.angle = float(inAng)
    self.profNum = int(profNum)
    self.fluxStr = str(inDict['flux_str'])
    self.iLayer = int(inDict['layer_index'])
    self.relErr = bool(inDict['relative_err'])

    ncVars = nc.Dataset(self.refNC, 'r').variables
    if self.fluxStr not in ncVars:
      print('Please choose --flux_str from:')
      for ncVar in ncVars: print('  %s' % ncVar)
      sys.exit(1)
    # endif fluxStr

    self.exe = inDict['executable']
    utils.file_check(self.exe)

    # executable needs a reference file, which is probably different 
    # from self.refNC since the latter is a 3-angle calculation
    self.exeRef = inDict['template_nc']
    utils.file_check(self.exeRef)

    self.outDir = inDict['test_dir']; utils.file_check(self.outDir)

    self.template = 'rrtmgp-inputs-outputs.nc'
    split = self.template.split('.')
    self.outNC = '%s/%s_ang%02d.nc' % (self.outDir, split[0], inAng)
  # end constructor

  def refExtract(self):
    """
    Extract all the information we need from the reference netCDF
    """

    with nc.Dataset(self.refNC, 'r') as ncObj:
      # grab specified flux at specified layer
      self.fluxRef = \
        np.array(ncObj.variables[self.fluxStr])\
        [:, self.iLayer, self.profNum]

      # calculate the transmitance
      od = np.array(ncObj.variables['tau'])\
        [:, :, self.profNum].sum(axis=1)
      self.transmittance = np.exp(-od)
    # endwith

  # end refExtract

  def runRRTMGP(self):
    """
    Run RRTMGP flux calculator at single angle, then calculate 
    test-reference errors, then plot as a function of transmittance
    """

    # RRTMGP code does not need to be run for every profile (the 
    # output contains results for *all* profiles), just every angle
    if os.path.exists(self.outNC):
      print('%s already exists, not recomputing' % self.outNC)
    else:
      # stage the template into a file that RRTMGP expects
      shutil.copyfile(self.exeRef, self.template)

      # run the RRTMGP flux calculator
      sub.call([self.exe, str(self.angle)])

      # move the output so it won't be overwritten
      os.rename(self.template, self.outNC)
    # endif self.outNC

    with nc.Dataset(self.outNC, 'r') as ncObj:
      self.fluxTest = \
        np.array(ncObj.variables[self.fluxStr])\
          [:, self.iLayer, self.profNum]
    # endwith

    self.fluxErr = self.fluxTest-self.fluxRef
    if self.relErr: self.fluxErr /= self.fluxRef

  # end runRRTMGP()
# end fluxErr

class combineErr():
  def __init__(self, fluxErrList, inDict):
    """
    Consolidates flux errors calculated for multiple fluxErr objects,
    saves them into a netCDF, and optionally generates plots of 
    the trends

    Input
      fluxErrList -- list of fluxErr objects. Should be an 
        nProfile-element list of nAngle-element lists of objects, 
        each of which should have transmittances and errors for every
        g-point

    Keywords
      inDict -- dictionary with the following keys:
        angle_range -- 3-element float list of starting and ending 
          angles and sampling rate for plot. By default, we span 45 
          to 65 degrees and plot every third angle.
        prefix -- string, prefix for output PNG file
        relative_err -- boolean, plot relative differences rather 
          than absolute flux differences
        smooth -- boolean, plot smooth curves
        binning -- float, transmittance binning for smoothing
        prob_dist -- boolean, plot probability distribution
        plot_fit -- boolean, plot fits to the flux errors as a 
          function of t rather than raw flux errors
    """

    self.allObj = list(fluxErrList)
    self.relErr = bool(inDict['relative_err'])
    self.smooth = bool(inDict['smooth'])
    self.binning = inDict['binning']
    self.probDist = inDict['prob_dist']
    self.samplingAng = inDict['angle_range'][2]
    self.pngPrefix = str(inDict['prefix'])
    self.plotFit = bool(inDict['plot_fit'])
    if args.smooth: self.pngPrefix += '_smooth'

    # we're assuming all profiles were run over the same amount of 
    # angles and g-points
    self.nAngles = len(fluxErrList[0])
    self.nG = fluxErrList[0][0].fluxErr.size
    self.nProfiles = len(fluxErrList)
    self.iLayer = fluxErrList[0][0].iLayer

    self.yLab = r'$\frac{F_{1-angle}-F_{3-angle}}{F_{3-angle}}$' if \
      self.relErr else '$F_{1-angle}-F_{3-angle}$'
    self.xLab = 'Transmittance'

    if self.smooth:
      self.yLab = \
        r'($\overline{\frac{F_{1-angle}-F_{3-angle}}{F_{3-angle}}}$)'
      self.xLab += ' (Binning = %.2f)' % self.binning
    # end smooth labels
  # end constructor

  def makeArrays(self):
    """
    Generate and transformt arrays of transmittances and flux error
    for plotting
    """

    # we wanna loop over all profiles and append their "spectra" 
    # onto each other so we can combine all profiles' err vs. t onto 
    # same plot

    # initialize lists and arrays
    angles, profs, err, tran = [], [], [], []
    err = np.zeros((self.nProfiles, self.nAngles, self.nG))
    tran = np.zeros((self.nProfiles, self.nG))

    for iProf, profObj in enumerate(self.allObj):
      for iAng, angObj in enumerate(profObj):
        # these two lists will end up having nAng and nProf dim
        angles.append(angObj.angle)
        profs.append(angObj.profNum)

        # transmittance is the same for every angle run
        tran[iProf, :] = angObj.transmittance
        err[iProf, iAng, :] = angObj.fluxErr
      # end loop over fluxErr objects
    # end profile loop

    self.angles = np.unique(angles)
    self.profs = np.unique(profs)

    # combine transmittances from all profiles together
    self.transmittance = np.array(tran).flatten()

    # these two conditionals really should never fail
    if self.angles.size != self.nAngles:
      print('Cannot continue, angles are not consistent')
      sys.exit(1)
    # endif angles

    if self.profs.size != self.nProfiles:
      print('Cannot continue, profiles are not consistent')
      sys.exit(1)
    # endif profiles

    # want a [nAng x (nProf * nG)] 2-D array of errors where we 
    # combine all of the errors from profiles
    temp = np.transpose(np.array(err), axes=(1, 0, 2))
    self.err = temp.reshape(\
      (self.nAngles, self.nG * self.nProfiles))

    if self.smooth:
      tran = np.arange(0, 1+self.binning, self.binning)
      sErr = []

      for it, t2 in enumerate(tran):
        if it == 0:
          sErr.append(np.repeat(np.nan, self.nAngles))
          continue
        # endif 0

        t1 = tran[it-1]
        itBin = np.where((self.transmittance >= t1) & \
          (self.transmittance < t2 ))[0]

        # fill values for all angles if no t exist in bin
        if itBin.size == 0:
          sErr.append(np.repeat(np.nan, self.nAngles))
          continue
        # endif 0

        sErrAng = []
        for iAng, ang in enumerate(self.angles):
          smoothErr = self.err[iAng, itBin].mean()
          sErrAng.append(smoothErr)
        # end angle loop

        sErr.append(sErrAng)
      # end t bin loop

      self.transmittance = np.array(tran)
      self.err = np.array(sErr).T
    # end smoothing
  # end makeArrays()

  def plotErrT(self):
    """
    Plot flux error as a function of transmittance for every angle.
    """

    outPNG = '%s_flux_errors_transmittance.png' % self.pngPrefix
    leg, fits, roots, newAng = [], [], [], []
    tran = np.array(self.transmittance)
    for iErr, errAng in enumerate(self.err):
      # don't plot curves for all angles
      if iErr % self.samplingAng != 0: continue

      iSort = np.argsort(tran)
      plot.plot(tran[iSort], errAng[iSort], '-')

      # legend string -- ?? degrees
      leg.append('%d' % self.angles[iErr] + r'$^{\circ}$')
    # end angle loop

    # aesthetics
    plot.xlabel(self.xLab)
    plot.ylabel(self.yLab)
    plot.title('Flux Error')
    plot.legend(leg, numpoints=1, loc='upper left', \
      prop=font_prop, framealpha=0.5)
    plot.gca().axhline(0, linestyle='--', color='k')
    plot.savefig(outPNG)
    plot.close()
    print('Wrote %s' % outPNG)
  # end plotErrT()

  def plotThetaOptT(self):
    """
    Plot optimized diffusivity angle (i.e., angle the minimizes error 
    error between 1- and 3-angle flux calculations) as a function of
    transmittance
    """

    outPNG = '%s_opt_angle_transmittance.png' % self.pngPrefix
    iErrMin = np.argmin(np.abs(self.err), axis=0)
    iSort = np.argsort(self.transmittance)
    plot.plot(self.transmittance[iSort], self.angles[iErrMin][iSort], 'o')
    plot.xlabel(self.xLab)
    plot.ylabel(r'$\theta_{optimized}$')
    plot.title('Diffusivity Angle Optimization')
    plot.savefig(outPNG)
    plot.close()

    print('Wrote %s' % outPNG)
  # end plotThetaOptT()

  def fitErrT(self):
    """
    Fit a cubic spline to the curve for each angle for better 
    determination of the root
    """

    leg, fits, roots, newAng = [], [], [], []
    tran = np.array(self.transmittance)
    for iErr, errAng in enumerate(self.err):
      # don't plot curves for all angles
      if iErr % self.samplingAng != 0: continue

      iSort = np.argsort(tran)

      # fit a cubic spline to the curve for this angle
      fit = INTER.UnivariateSpline(tran[iSort], errAng[iSort])
      fitDat = fit(tran[iSort])

      # some a posteriori hand waving here...
      angRoots = fit.roots()
      if len(angRoots) == 0: continue
      root = angRoots[-1] if len(angRoots) > 1 else angRoots[0]
      fits.append(fitDat)
      roots.append(root)
      newAng.append(self.angles[iErr])

      # legend string -- ?? degrees
      leg.append('%d' % self.angles[iErr] + r'$^{\circ}$')
    # end angle loop

    self.fits = np.array(fits)
    self.roots = np.array(roots)
    self.angles = np.array(newAng)
    self.secant = 1/np.cos(np.radians(self.angles))
  # end fitErrT()

  def playground(self):
    
    newFit = INTER.UnivariateSpline(self.roots, self.secant)
    plot.plot(self.roots, self.secant, 'bo', \
      self.roots, newFit(self.roots), 'r')
    plot.show()
    print(newFit.get_coeffs())
  # end 
# end combineErr

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(\
    description='Module that contains class that finds optimized ' + \
    'angle for RRTMGP flux calculations.')
  parser.add_argument('--reference', '-ref', type=str, \
    default='rrtmgp-lw-inputs-outputs-clear-ang-3.nc', \
    help='Reference netCDF file from test_lw_solver run.')
  parser.add_argument('--flux_str', '-fs', type=str, \
    default='gpt_flux_dn', \
    help='String of netCDF flux variable to extract.')
  parser.add_argument('--layer_index', '-l', type=int, default=0, \
    help='Zero-offset index of layer to use in comparison.')
  parser.add_argument('--profiles', '-p', type=int, default=42, \
    help='Number of profiles over which to loop.')
  parser.add_argument('--angle_range', '-a', type=float, \
    default=[45, 65, 3], nargs=3, \
    help='Starting and ending angles over which to loop and ' + \
    'the sampling rate for the plot. By default, we span 45 to ' + \
    '65 degrees at the specified angular resolution and plot ' + \
    'every third angle.')
  parser.add_argument('--angle_resolution', '-res', type=int, \
    default=1, help='Angle resolution over which to loop.')
  parser.add_argument('--executable', '-e', type=str, \
    default='test_lw_solver_noscat', \
    help='RRTMGP flux solver executable')
  parser.add_argument('--template_nc', '-temp', type=str, \
    default='rrtmgp-lw-inputs-outputs-clear.nc', \
    help='netCDF that is used as input into executable. The ' + \
    'code will copy it and use a naming convention that the ' + \
    'executable expects.')
  parser.add_argument('--test_dir', '-td', type=str, \
    default='./trial_results', \
    help='Directory to which the results from RRTMGP runs are saved.')
  parser.add_argument('--relative_err', '-r', action='store_true', \
    help='Plot relative rather than absolete flux errors')
  parser.add_argument('--save_file', '-s', type=str, \
    default='Garand_optimize_diffusivity_angle.p', \
    help='Name of pickle file that contains list of ' + \
    'fluxErr objects. This can be used to save time if the ' + \
    'analysis has already been done and one just needs to ' + \
    'proceed to plotting. If the file exists, the code will ' + \
    'skip to plotting. If it does not, the list of objects will ' + \
    'be saved to it.')
  parser.add_argument('--prefix', '-pre', type=str, \
    default='flux_errors_Garand', help='Prefix for output PNG File')
  parser.add_argument('--smooth', '-sm', action='store_true', \
    help='Smooth the curves using t binning and averaging.')
  parser.add_argument('--binning', '-b', type=float, default=0.01, \
    help='Binning to use in smoothing of the curves.')
  parser.add_argument('--prob_dist', '-pd', action='store_true', \
    help='Plot probability distribution.')
  parser.add_argument('--plot_fit', '-fit', action='store_true', \
    help='Plot the cubic spline fit to the errors instead of ' + \
    'raw errors.')
  args = parser.parse_args()

  angles, res = args.angle_range, args.angle_resolution
  pFile = args.save_file

  if os.path.exists(pFile):
    fErrAll = pickle.load(open(pFile, 'rb'))
  else:
    # loop over all angles (inclusive), generating a fluxErr object 
    # for each angle and profile, which we'll combine using another 
    # class; fErrAll contains objects for all profiles and angles
    fErrAll = []
    for iProf in range(args.profiles):
      # fErrAng contains objects for all angles for a given profile
      fErrAng = []
      for ang in np.arange(angles[0], angles[1]+res, res):
        print('Running Profile %d, %d degrees' % (iProf+1, ang))
        fErr = fluxErr(ang, iProf, vars(args))
        fErr.refExtract()
        fErr.runRRTMGP()
        fErrAng.append(fErr)
      # end angle loop

      fErrAll.append(fErrAng)

      # save the output for later plotting
      pickle.dump(fErrAll, open(pFile, 'wb'))
    # end profile loop
  # endif npzFile

  combObj = combineErr(fErrAll, vars(args))
  combObj.makeArrays()
  if args.plot_fit:
    combObj.fitErrT()
    combObj.playground()
  else:
    combObj.plotErrT()
    combObj.plotThetaOptT()
  # endif plot_fit

# end main()

