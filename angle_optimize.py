#!/usr/bin/python

from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.font_manager as font
import netCDF4 as nc
import subprocess as sub
import shutil

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
      inDict -- dictionary with the following keys:
        reference -- str, netCDF with 3-angle reference fluxes
        flux_str -- str, netCDF variable name of fluxes to compare
        executable -- str, path to flux calculation executable to run
        template_nc -- str, path to netCDF used for executable I/O
    """

    self.refNC = str(inDict['reference'])
    utils.file_check(self.refNC)

    self.angle = float(inAng)
    self.profNum = int(profNum)+1
    self.fluxStr = str(inDict['flux_str'])
    self.iLevel = int(inDict['level_index'])

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
    base = os.path.basename(self.exeRef)
    split = base.split('.')

    self.template = 'rrtmgp-inputs-outputs.nc'
  # end constructor

  def refExtract(self):
    """
    Extract all the information we need from the reference netCDF
    """

    with nc.Dataset(self.refNC, 'r') as ncObj:
      # grab specified flux at specified level
      self.fluxRef = \
        np.array(ncObj.variables[self.fluxStr])\
        [:, self.iLevel, self.profNum]

      # calculate the transmitance
      od = np.array(ncObj.variables['tau'])\
        [:, self.iLevel, self.profNum]
      self.transmittance = np.exp(-od)
    # endwith

  # end refExtract

  def runRRTMGP(self):
    """
    Run RRTMGP flux calculator at single angle, then calculate 
    test-reference errors, then plot as a function of transmittance
    """

    # stage the template into a file that RRTMGP expects
    shutil.copyfile(self.exeRef, self.template)

    # run the RRTMGP flux calculator
    sub.call([self.exe])

    # move the output so it won't be overwritten
    #os.rename(self.template, self.outNC)

    with nc.Dataset(self.template, 'r') as ncObj:
      self.fluxTest = \
        np.array(ncObj.variables[self.fluxStr])\
          [:, self.iLevel, self.profNum]
    # endwith

    self.fluxErr = self.fluxTest-self.fluxRef
  # end runRRTMGP()
# end fluxErr

class combineErr():
  def __init__(self, fluxErrList):
    """
    Consolidates flux errors calculated for multiple fluxErr objects,
    saves them into a netCDF, and optionally generates plots of 
    the trends

    Input
      fluxErrList -- list of fluxErr objects. Should be an 
        nProfile-element list of nAngle-element lists of objects, 
        each of which should have transmittances and errors for every
        g-point
    """

    self.allObj = list(fluxErrList)

    # we're assuming all profiles were run over the same amount of 
    # angles and g-points
    self.nAngles = len(fluxErrList[0])
    self.nG = fluxErrList[0][0].fluxErr.size
    self.nProfiles = len(fluxErrList)
    self.iLevel = fluxErrList[0][0].iLevel

    self.pngPrefix = 'flux_errors_Garand'
  # end constructor

  def makeArrays(self):
    """
    Generate nProfile x nAngle x nG arrays of transmittances and 
    flux error
    """

    # initialize lists and arrays
    angles, profs, = [], []
    err = np.zeros((self.nProfiles, self.nAngles, self.nG))
    tran = np.zeros((self.nProfiles, self.nG))

    for iProf, profObj in enumerate(self.allObj):
      for iAng, angObj in enumerate(profObj):
        angles.append(angObj.angle)
        profs.append(angObj.profNum)
        tran[iProf, :] = angObj.transmittance
        err[iProf, iAng, :] = angObj.fluxErr
      # end loop over fluxErr objects

    self.angles = np.unique(angles)
    self.profs = np.unique(profs)
    self.err = np.array(err)

    # transmittance is the same for every angle run
    # convert it from an array to a vector
    self.transmittance = np.reshape(np.array(tran), (tran.size))

    # these two conditionals really should never fail
    if self.angles.size != self.nAngles:
      print('Cannot continue, angles are not consistent')
      sys.exit(1)
    # endif angles

    if self.profs.size != self.nProfiles:
      print('Cannot continue, profiles are not consistent')
      sys.exit(1)
    # endif profiles
  # end makeArrays()

  def plotErrT(self):
    """
    Plot flux error as a function of transmittance for every angle.

    Make a separate figure for each profile.
    """

    # plotting parameters
    leg = ['%d' % ang + r'$^{\circ}$' for ang in self.angles]

    tran = np.array(self.transmittance)
    for iProf, errProf in enumerate(self.err):
      outPNG = '%s_%03d.png' % (self.pngPrefix, iProf+1)
      for errAng in errProf:
        iSort = np.argsort(tran)
        plot.plot(tran[iSort], errAng[iSort], '-')
      # end angle loop

      # aesthetics
      plot.xlabel('Transmittance, Level %d' % self.iLevel)
      plot.ylabel('F$_{1-angle}$-F$_{3-angle}$')
      plot.title('Flux Error, Profile %d' % (iProf+1) )
      plot.legend(leg, numpoints=1, loc='upper left', \
        prop=font_prop, framealpha=0.5)
      plot.savefig(outPNG)
      plot.close()
      print('Wrote %s' % outPNG)

    # end profile loop
  # end plotErrT()
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
  parser.add_argument('--level_index', '-l', type=int, default=0, \
    help='Zero-offset index of level to use in comparison.')
  parser.add_argument('--profiles', '-p', type=int, default=42, \
    help='Number of profiles over which to loop.')
  parser.add_argument('--angle_range', '-a', type=float, \
    default=[45, 65], nargs=2, \
    help='Starting and ending angles over which to loop')
  parser.add_argument('--angle_resolution', '-res', type=int, \
    default=1, help='Angle resolution over which to loop.')
  parser.add_argument('--executable', '-e', type=str, \
    default='test_lw_solver', \
    help='RRTMGP flux solver executable')
  parser.add_argument('--template_nc', '-temp', type=str, \
    default='rrtmgp-lw-inputs-outputs-clear.nc', \
    help='netCDF that is used as input into executable. The ' + \
    'code will copy it and use a naming convention that the ' + \
    'executable expects.')
  args = parser.parse_args()

  angles, res = args.angle_range, args.angle_resolution

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
  # end profile loop

  combObj = combineErr(fErrAll)
  combObj.makeArrays()
  combObj.plotErrT()
# end main()

