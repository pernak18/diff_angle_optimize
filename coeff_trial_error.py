#!/usr/bin/env python

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plot
from pandas import DataFrame as DF

class coeffPairs:
  def __init__(self, inDict):
    """
    See doc in main(). Coefficient pairs are uniform across all bands
    and saved into a CSV with a unique name that contains a given 
    coefficient pair
    """

    slopes = inDict['slope_range']
    ints = inDict['intercept_range']

    # have to add slopes[2] and ints[2] to include slopes[1] and 
    # ints[1], respectively
    self.slopes = np.arange(slopes[0], slopes[1]+slopes[2], slopes[2])
    self.nSlopes = self.slopes.size
    self.sRes = slopes[2]
    self.ints = np.arange(ints[0], ints[1]+ints[2], ints[2])
    self.iRes = ints[2]
    self.nInts = self.ints.size
    self.preCSV = inDict['csv_prefix']
    self.dirCSV = inDict['csv_dir']

    if not os.path.exists(self.dirCSV): os.makedirs(self.dirCSV)

    self.stride = int(self.nSlopes / self.nInts)
    self.midPoint = self.nSlopes/2

    # ad-hoc. python rounding had this guy at 1e-16
    self.slopes[self.midPoint] = 0

    # because we're only going to/from the midpoint
    self.stride /= 2
  # end constructor

  def spanCoeffs(self):
    """
    Generate array of coefficients in a trapezoidal fashion: 
      min(intercept) spans [0, max(slope)]
      max(intercept) spans [min(slope), 0]
    """

    # not sure if this is the most efficient way to do this
    coeffArr = []
    for i, intercept in enumerate(self.ints):
      # indices of initial and final slope values for given intercept
      i1 = self.midPoint-i*self.stride
      i2 = self.nSlopes-i*self.stride
      coeffArr.append(self.slopes[i1:i2])
    # end intercept loop

    self.coeffArr = np.array(coeffArr)
    self.nY = self.coeffArr.shape[1]
  # end spanCoeffs()

  def plotCoeffs(self):
    """
    Sanity check: plot each coefficient pair
    """

    for i, intercept in enumerate(self.ints):
      x = np.repeat(intercept, self.nY)
      y = self.coeffArr[i, :]
      plot.plot(x, y, 'rx')
    # end intercept loop

    plot.xlabel('Intercepts')
    plot.ylabel('Slopes')
    plot.show()
  # end plotCoeffs()

  def writeCSV(self):
    """
    Write CSV with coefficient pair for all band
    """

    # generate a 16 x 2 (nBands x nCoeff) array for each pair of 
    # linear coefficients
    for i, intercept in enumerate(self.ints):
      for s, slope in enumerate(self.coeffArr[i, :]):
        outCSV = '%s/%s_%.2f_%.2f.csv' % \
          (self.dirCSV, self.preCSV, slope, intercept)
        forCSV = {'slopes': np.repeat(slope, 16), \
          'intercepts': np.repeat(intercept, 16)}
        DF.from_dict(forCSV).to_csv(outCSV, index=False, \
          header=None, float_format='%.3f', \
          columns=['slopes', 'intercepts'])
        print('Wrote %s' % outCSV)
      # end slope loop
    # end intercept loop
  # end writeCSV()
# end coeffPairs

if __name__ == '__main__':
  parser = argparse.ArgumentParser(\
    description='Generate numerous pairs of linear coefficients ' + \
    'that span the range of each coefficient as specified by ' + \
    'the user. The coefficient pairs will then be placed into ' + \
    'a CSV file that can be used in e2e_modify_optimization.py.')
  parser.add_argument('--slope_range', '-s', nargs=3, type=float, \
    default=[-0.5, 0.5, 0.05], help='Min, max, resolution for slope.')
  parser.add_argument('--intercept_range', '-i', nargs=3, 
    type=float, default=[1.5, 1.9, 0.05], \
    help='Min, max, resolution for intercept coefficient.')
  parser.add_argument('--csv_prefix', '-p', type=str, \
    default='sec_t_fit_coeffs',\
    help='Prefix for CSV files onto which the coefficient pairs ' + \
    'will be appended.')
  parser.add_argument('--csv_dir', '-d', type=str, \
    default='CSV_files', \
    help='Directory into which CSV file will be written')
  parser.add_argument('--verify', '-v', action='store_true', \
    help='Make verification plot -- just sent to plotting window.')
  args = parser.parse_args()

  cpObj = coeffPairs(vars(args))
  cpObj.spanCoeffs()

  if args.verify: cpObj.plotCoeffs()

  cpObj.writeCSV()
# end main()
