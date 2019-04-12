#!/usr/bin/env python

from __future__ import print_function

import os, sys, argparse, glob
import numpy as np
import shutil
from multiprocessing import Pool, cpu_count, Process

# Git submodule
sys.path.append('common')
import utils

# local module
from e2e_modify_optimization import *

def e2ePool(inObj):
  """
  Run methods of multiple e2e objects in parallel
  """

  print('Processing %s' % inObj.iniFile)

  if inObj.doProfs: inObj.plotProf()
  if inObj.doStats: inObj.plotStat()
# end e2ePool()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(\
    description='')
  parser.add_argument('--ini_files', '-i', type=str, nargs='+', \
    help='Any number of configuration files to use as input ' + \
    'into e2e objects. It is assumed that the computation ' + \
    'for all configuration is the same, but the plotting ' + \
    'parameters are different.')
  parser.add_argument('--cores', '-c', type=int, default=6, \
    help='Number of cores over which to process')
  args = parser.parse_args()

  iniFiles = args.ini_files

  if iniFiles is None:
    sys.exit('No .ini files provided, cannot proceed')

  e2eObjList = []
  for iniFile in iniFiles:
    if not os.path.exists(iniFile):
      print('Could not find %s' % iniFile)
      continue
    # endif file check

    # only call constructor and "second constructor"
    # that determines attributes from config file
    obj = e2e(iniFile)
    obj.readConfig()
    e2eObjList.append(obj)
  # end iniFile loop

  # since computation is the same, we only need to run these 
  # methods once
  obj.getFilesNC()
  obj.readCoeffs()
  obj.reFit()
  obj.bandmerge()

  nCores = args.cores
  totCores = cpu_count()
  nCores = nCores if nCores < totCores else totCores-1

  p = Pool(nCores)
  pe2e = p.map(e2ePool, e2eObjList)
# end main()
