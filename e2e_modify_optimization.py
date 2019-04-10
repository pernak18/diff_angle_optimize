#!/usr/bin/env python

from __future__ import print_function

# standard libraries
import os, sys, argparse, glob, shutil
import numpy as np
import pandas as pd
import configparser as ConfigParser
import netCDF4 as nc
import subprocess as sub

import LBLRTM_RRTMGP_compare as COMPARE
import wrapper_combine_up_down as WRAP

# local module (utils.py in the common Git submodule did not work)
import utils

class e2e(WRAP.combineBandmerge):
  def __init__(self, inFile):
    """
    Not a whole lot happens in this constructor because we need to 
    extract information from the configuration file

    We are inheriting the wrapper combineBandmerge class only for 
    the bandmerge() method
    """

    utils.file_check(inFile)
    self.iniFile = inFile
    self.nBands = 16

    # for heating rate calculations
    self.heatFactor = 8.4391

    # each g-point has a weight. this is in the literature but 
    # was also provided by Eli in 
    # https://rrtmgp2.slack.com/archives/D942AU7QE/p1550264049021200
    # these will only be used if we're doing the by-band calculations
    self.gWeights = np.array([0.1527534276, 0.1491729617, \
      0.1420961469, 0.1316886544,  0.1181945205,  0.1019300893, \
      0.0832767040, 0.0626720116, 0.0424925000,0.0046269894, \
      0.0038279891, 0.0030260086, 0.0022199750, 0.0014140010, \
      0.0005330000, 0.0000750000])
  # end constructor

  def readConfig(self):
    """
    Read in the configuration file parameters that will be used with 
    the COMPARE module
    """

    # all of the valid affirmative entries for booleans in .ini file
    yes = ['y', 'ye', 'yes', 't', 'tr', 'tru', 'true'] 

    print('Reading %s' % self.iniFile)

    cParse = ConfigParser.ConfigParser()
    cParse.read(self.iniFile)

    self.bandAvg = cParse.get('Computation', 'band_average')

    self.refName = cParse.get('Plot Params', 'reference_model')
    self.refDesc = cParse.get('Plot Params', 'reference_description')
    self.testName = cParse.get('Plot Params', 'test_model')
    self.TestDesc = cParse.get('Plot Params', 'test_description')
    self.atmType = cParse.get('Plot Params', 'atmosphere')
    self.yLog = cParse.get('Plot Params', 'log')
    self.bands = cParse.get('Plot Params', 'bands')

    self.refFile = cParse.get('Filename Params', 'reference_path')
    self.testFile = cParse.get('Filename Params', 'test_path')
    self.profPrefix = cParse.get('Filename Params', 'profiles_prefix')
    self.statPrefix = cParse.get('Filename Params', 'stats_prefix')

    self.csvFile = cParse.get('Filename Params', 'coefficients_file')
    self.outDir = cParse.get('Filename Params', 'output_dir')
    self.exe = cParse.get('Filename Params', 'solver')
    self.ncDir = cParse.get('Filename Params', 'netcdf_dir')

    # handle forcing parameters
    self.forcing = False
    self.rForceFile = \
      cParse.get('Filename Params', 'reference_force_path')
    self.tForceFile = cParse.get('Filename Params', 'test_force_path')

    self.rForceName = \
      cParse.get('Plot Params', 'reference_forcing_model')
    self.rForceDesc = \
      cParse.get('Plot Params', 'reference_description')
    self.tForceName = \
      cParse.get('Plot Params', 'test_forcing_model')
    self.tForceDesc = \
      cParse.get('Plot Params', 'test_description')

    if os.path.exists(self.rForceFile) and \
      os.path.exists(self.tForceFile):
      self.forcing = True
      self.xTitle = self.rForceName
      yt = '%s - %s' % (cTestForceName, cRefForceName)

      self.refFile, self.testFile = \
        COMPARE.forcingDiff(self.refFile, self.testFile, \
        self.rForceFile, self.tForceFile)
    else:
      forceFile = self.rForceFile if not \
        os.path.exists(self.rForceFile) else self.tForceFile

      if forceFile != '':
        print('Could not find %s, forcing not done' % forceFile)
    # end forcing

    # check that everything that is required exists
    paths = [self.refFile, self.testFile, self.csvFile, \
      self.exe, self.ncDir]
    for path in paths: utils.file_check(path)
    if not os.path.exists(self.outDir): os.makedirs(self.outDir)

    # for consistency with bandmerge() method -- the bandmerged 
    # file is the same as the test netCDF file
    self.mergeNC = self.testFile

    # standardize band averaging, log plot and bands inputs
    self.bandAvg = True if self.bandAvg in yes else False
    self.yLog = True if self.yLog.lower() in yes else False
    self.bands = np.arange(self.nBands) if self.bands == '' else \
      np.array(self.bands.split()).astype(int)-1

    self.xTitle = self.refName
    self.yTitle = '%s - %s' % (self.testName, self.refName)
  # end readConfig()

  def getFilesNC(self):
    """
    For given bands, find the associated RRTMGP netCDF with g-point 
    fluxes produces by self.exe. Find a file for each band even though
    only a subset will be modified (this is necessary for bandmerge())
    """

    print('Gathering band netCDF files in %s' % self.ncDir)

    prefix = 'rrtmgp-inputs-outputs_opt_ang_band'
    ncFiles = {'modified': [], 'unmodified': []}
    for iBand in range(self.nBands):
      ncFile = '%s/%s%02d.nc' % (self.ncDir, prefix, iBand+1)

      if os.path.exists(ncFile):
        if iBand in self.bands:
          ncFiles['modified'].append(ncFile)
        else:
          ncFiles['unmodified'].append(ncFile)
        # endif modified
      else:
        # this has not really been tested
        print('Could not find %s,' % ncFile, end=" ")
        print('further processing cannot be performed', end=" ")
        print('(all bands are needed).')
        sys.exit(1)
      # endif ncFile exists
    # end band loop

    self.ncFiles = ncFiles

    # list for bandmerge() method
    self.allOutNC = sorted(self.ncFiles['modified'] + \
      self.ncFiles['unmodified'])

    # sanity check
    nOut = len(self.allOutNC)
    if nOut != self.nBands:
      print('Inconsistent number of netCDF files found')
      for nc in self.allOutNC: print(nc)
      print('Expected: %d, Actual: %d' % (self.nBands, nOut))
      sys.exit(1)
    # end n check
  # end gPtFiles()

  def readCoeffs(self):
    """
    Read and store user-specified coefficients of fit to optimized 
    angle vs. transmittance data (linear fit assumed)
    """

    print('Reading user coefficients from %s' % self.csvFile)

    dat = pd.read_csv(self.csvFile, header=None)
    self.coeffs = np.array(dat)
  # end readCoeffs()

  def reFit(self):
    """
    Replace coefficients of solution (secant_fit) with user-supplied 
    coefficients, then re-run RRTMGP to calculate new g-point fluxes
    """

    for ncFile, iBand in zip(self.ncFiles['modified'], self.bands):
      print('Reprocessing Band %d' % (iBand+1))

      # calculate whole-column transmittances to be used in fit
      with nc.Dataset(ncFile, 'r') as ncObj:
        od = np.array(ncObj.variables['tau']).sum(axis=1)

        if self.bandAvg:
          i1, i2 = np.array(ncObj.variables['band_lims_gpt'])[iBand]-1
          iArr = np.arange(i1, i2+1)

          # extract only the optical depths for this band use g-point
          # weights to calculate band average
          bandOD = od[iArr, :].T * self.gWeights
          sumOD = bandOD.T.sum(axis=0)
          sumOD = np.reshape(sumOD, (1, 42))

          # populate the average ODs for this band to the rest of the 
          # OD array
          od = np.repeat(sumOD, 256, axis=0)
        # end band averaging

        origTran = np.exp(-od)
      # endwith ncObj

      # write new secants (based on user coefficients ) to input 
      # file for self.exe
      with nc.Dataset('optimized_secants.nc', 'r+') as ncObj:
        # we have to assign the secants for all bands because 
        # self.exe expects a 256 x 42 array; eventually this is 
        # reorganized by-band in bandmerge(). for now, we just treat
        # the optimized solution for one band (16 g-points) as the 
        # optimized solution for all points (256 g-points)
        secantFit = np.poly1d(self.coeffs[iBand, :])
        newSecants = secantFit(origTran)
        ncVar = ncObj.variables['secant']
        ncVar[:] = newSecants
      # endwith ncObj

      # stage input file expected by RRTMGP executable
      stageNC = 'rrtmgp-inputs-outputs.nc'
      if not os.path.exists(stageNC): shutil.copyfile(ncFile, stageNC)

      # run the executable with optimized_secants.nc and refNC
      print('Running LW Solver')
      sub.call([self.exe])

      # replace original ncFile with one with newly-calculated fluxes
      os.rename(stageNC, ncFile)
      print('Wrote %s' % ncFile)
    # end ncFile loop
  # end reFit()

  def plotProf(self):
    """
    Plot test and reference flux and heating rate profiles and the 
    test-ref differences
    """

    for iBand in self.bands:
      COMPARE.profPDFs(self.refFile, self.testFile, self.yTitle, \
        prefix=self.profPrefix, atmType=self.atmType, inBand=iBand, \
        yLog=self.yLog)
      tmpPDF = '%s_%02d.pdf' % (self.profPrefix, iBand+1)
      os.rename(tmpPDF, '%s/%s' % (self.outDir, tmpPDF))
    # end iBand loop
  # end plotProf()
# end e2e

if __name__ == '__main__':
  parser = argparse.ArgumentParser(\
    description='End-to-end script for modifying the ' + \
    'coefficients to the angle diffusivity optimization.')
  parser.add_argument('--config_file', '-i', type=str, \
    default='rrtmgp_lblrtm_config_garand_opt_ang.ini', \
    help='Configuration file that is used with ' + \
    'LBLRTM_RRTMGP_compare.py')
  args = parser.parse_args()

  e2eObj = e2e(args.config_file)
  e2eObj.readConfig()
  e2eObj.getFilesNC()
  e2eObj.readCoeffs()
  e2eObj.reFit()
  e2eObj.bandmerge()
  e2eObj.plotProf()
# end main()

