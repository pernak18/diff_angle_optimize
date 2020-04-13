#!/usr/bin/env python

import os, sys, glob, shutil, argparse
import configparser as cp
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count, Process

# Git submodule
sys.path.append('common')
import utils

# local module
from e2e_modify_optimization import *

class newConfig:
  def __init__(self, inDict):
    """
    See main()
    """

    self.iniTemp = inDict['template_ini']
    self.csvDir = inDict['CSV_dir']
    self.iniDir = inDict['ini_dir']
    self.nCores = inDict['num_cores']

    paths = [self.iniTemp, self.csvDir]
    for path in paths: utils.file_check(path)

    if not os.path.exists(self.iniDir): os.makedirs(self.iniDir)
  # end constructor

  def coeffPairs(self):
    """
    Extract coefficients from filenames of CSV files generated with
    coeff_trial_error.py
    """
    csvFiles = sorted(glob.glob('%s/*.csv' % self.csvDir))
    nCSV = len(csvFiles)

    if nCSV == 0:
      print('No coeff_trial_error.py CSV files found')
      sys.exit(1)
    # endif

    trialCoeffs = []
    for csv in csvFiles:
      coeffs = pd.read_csv(csv, header=None)

      # all bands have the same coefficients, so just grab them from
      # the first band
      trialCoeffs.append([coeffs[0][0], coeffs[1][0]])
    # end csv loop

    self.csvFiles = csvFiles
    self.coeffs = np.array(trialCoeffs)
  # end coeffPairs()

  def newNames(self):
    """
    Generate list of names for new .ini files

    Pretty ad-hoc -- i'm putting profiles (log and linear) underneath
    the same directory and giving stats their own
    """

    self.plotTypes = ['linear_profs', 'log_profs', 'stats']
    self.plotDirs = \
      ['profile_plots/linear', 'profile_plots/log', 'stat_plots']

    iniCSV = {}
    for pType in self.plotTypes:
      iniNames = []
      subDir = '%s/%s' % (self.iniDir, pType)
      if not os.path.exists(subDir): os.makedirs(subDir)

      for coeff in self.coeffs:
        posNeg = 'neg' if coeff[0] < 0 else 'pos'
        outIni = '%s/lin_coeff_%s%.2f_%.2f.ini' % \
          (subDir, posNeg, abs(coeff[0]), coeff[1])
        iniNames.append(outIni)
      # end coeff loop

      iniCSV[pType] = iniNames

    # end plot type loop

    self.iniNames = iniCSV
  # end newNames()

  def iniWrite(self):
    """
    Write new .ini files and replace fields that need to be replaced

    This is pretty ad-hoc -- we are using conventions established in
    other code (e.g., in the newNames() method)
    """

    import configparser

    # config file fields to replace and their respective sections
    # in .ini file
    fieldReplace = ['log', 'prof_plots', 'stats_plots', \
      'coefficients_file', 'output_dir', 'band_average']
    sections = ['Plot Params', 'Plot Params', 'Plot Params', \
      'Filename Params', 'Filename Params', 'Computation']

    for pType, pDir in zip(self.plotTypes, self.plotDirs):
      for iConf, iniFile in enumerate(self.iniNames[pType]):
        # start from the config file template
        shutil.copyfile(self.iniTemp, iniFile)

        # edit fields in new ini file
        cParse = configparser.ConfigParser()
        cParse.read(iniFile)
        for iField, field in enumerate(fieldReplace):
          if field == fieldReplace[0]:
            newVal = 'y' if 'log_profs' in iniFile else ''
          elif field == fieldReplace[1]:
            newVal = 'y' if 'profs' in iniFile else ''
          elif field == fieldReplace[2]:
            newVal = 'y' if 'stats' in iniFile else ''
          elif field == fieldReplace[3]:
            newVal = self.csvFiles[iConf]
          elif field == fieldReplace[4]:
            newVal = pDir
          elif field == fieldReplace[5]:
            # don't do band averagin
            newVal = ''
          # endif field

          cParse.set(sections[iField], field, newVal)
        # end field replacement loop

        # we want unique names for each plot, and that's determined
        # by coefficients (which are in the .ini file), plot type
        # (which are in the output_dir name) and band (which is
        # appended by the plotting scripts)
        # for the coefficients, we'll just use the basename of the
        # .ini file without the extension
        base = os.path.basename(iniFile)[:-4]
        if 'log_profs' in iniFile: base += '_log'
        cParse.set('Filename Params', 'profiles_prefix', base)
        cParse.set('Filename Params', 'stats_prefix', base)

        with open(iniFile, 'w') as fp: cParse.write(fp)
        print('Wrote %s' % iniFile)

      # end ini loop
    # end plot type loop
  # end iniWrite

  def pe2e(self):
    """
    Pretty much just main() from parallel_e2e.py, with all of the
    .ini files generated with this class
    """

    for t1, t2, t3 in zip(self.iniNames[self.plotTypes[0]][1:], \
      self.iniNames[self.plotTypes[1]][1:], \
      self.iniNames[self.plotTypes[2]][1:]):
      iniFiles = [t1, t2, t3]
      iniFiles = [t1, t2]

      # make e2e object (from e2e_modify_optimization module) for
      # each .ini file
      e2eObjList = []
      for iniFile in iniFiles:
        if not os.path.exists(iniFile):
          print('Could not find %s' % iniFile)
          continue
        # endif file check

        # only call constructor and "second constructor"
        # that determines attributes from config file
        e2eDict = {'config_file': iniFile, 'broadband_only': False}
        obj = e2e(e2eDict)
        obj.readConfig()
        e2eObjList.append(obj)
      # end iniFile loop

      # since computation is the same, we only need to run these
      # methods once
      obj.getFilesNC()
      obj.readCoeffs()
      obj.reFit()
      obj.bandmerge()

      nCores = self.nCores
      totCores = cpu_count()
      nCores = self.nCores if self.nCores <= 3  else totCores-1

      p = Pool(nCores)
      pe2e = p.map(e2ePool, e2eObjList)
      p.close()
    # end iniFile loop
  # end pe2e()

# end newConfig

def e2ePool(inObj):
  """
  Run methods of multiple newConfig objects in parallel
  """

  print('Processing %s' % inObj.iniFile)

  if inObj.doProfs: inObj.plotProf()
  if inObj.doStats: inObj.plotStat()
# end e2ePool()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(\
    description='Starting with a template configuration file, ' + \
    'replace fields as necessary to generate a .ini file for ' + \
    'each coefficient pair (from coeff_trial_error.py) and ' + \
    'one for log, linear, and stat plots.')
  parser.add_argument('--template_ini', '-t', type=str, \
    default='rrtmgp_lblrtm_config_garand_opt_ang.ini', \
    help='Starting point configuration file that is used in ' + \
    'e2e_modify_optimization.py.')
  parser.add_argument('--CSV_dir', '-cd', type=str, \
    default='CSV_files', \
    help='Directory with CSV files for each linear coefficient ' + \
    'pair. Generated by coeff_trial_error.py.')
  parser.add_argument('--ini_dir', '-id', type=str, \
    default='ini_files', \
    help='Directory to which new .ini files will be written.')
  parser.add_argument('--num_cores', '-c', type=int, default=3, \
    help='Number of cores over which to process')
  args = parser.parse_args()

  newObj = newConfig(vars(args))
  newObj.coeffPairs()
  newObj.newNames()
  newObj.iniWrite()
  newObj.pe2e()
# end main()
