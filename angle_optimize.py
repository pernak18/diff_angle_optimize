#!/usr/bin/env python

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
  def __init__(self, inAng, inDict):
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
      relative_err -- boolean, plot relative instead of absolute 
        flux differences
      inDict -- dictionary with the following keys:
        reference -- str, netCDF with 3-angle reference fluxes
        flux_str -- str, netCDF variable name of fluxes to compare
        executable -- str, path to flux calculation executable to run
        template_nc -- str, path to netCDF used for executable I/O
        secant_nc -- str, path to netCDF with secant array
        test_dir -- str, path to which the results from RRTMGP runs 
          will be written
    """

    self.refNC = str(inDict['reference'])
    utils.file_check(self.refNC)

    self.angle = float(inAng)
    self.secant = np.array(1/np.cos(np.radians(self.angle)))
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

    self.secNC = inDict['secant_nc']
  # end constructor

  def refExtract(self):
    """
    Extract all the information we need from the reference netCDF
    """

    with nc.Dataset(self.refNC, 'r') as ncObj:
      # grab specified flux at specified layer
      self.fluxRef = \
        np.array(ncObj.variables[self.fluxStr])[:, self.iLayer, :]
      dims = self.fluxRef.shape
      self.nG, self.nProf = dims

      # calculate the transmitance
      od = np.array(ncObj.variables['tau']).sum(axis=1)
      self.transmittance = np.exp(-od)
    # endwith

  # end refExtract

  def writeSecNC(self):
    """
    Write a netCDF that only contains a "secant" array. This file 
    will be read in by lw_solver_opt_angs (or whatever the 
    executable into main() is).
    """

    with nc.Dataset(self.secNC, 'w') as ncObj:
      ncObj.description = 'Secant values for every g-point of ' + \
        'every (Garand) profile to fluxErr class in ' + \
        'angle_optimize.py for which errors between a 1-angle ' + \
        'test RRTGMP run and a 3-angle reference RRTMGP run were ' + \
        'calcualated. These are the angles at which the errors ' + \
        'were minimized.'
      ncObj.createDimension('profiles', self.nProf)
      ncObj.createDimension('gpt', self.nG)
      secant = ncObj.createVariable(\
        'secant', float, ('gpt', 'profiles'))
      secant.units = 'None'
      secant.description = 'Optimized secants for flux calculations'
      secant[:] = np.array(self.secant)
    # endwith
  # end writeSecNC()

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
        np.array(ncObj.variables[self.fluxStr])[:, self.iLayer, :]
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
        diagnostic -- boolean, plot diagnostic secant(roots) vs. 
          transmittance to see how well first fit does
        cutoff -- float, transmittance under which no fit is done
        weight -- boolean, weight the fitAngT() fit
    """

    self.allObj = list(fluxErrList)
    self.relErr = bool(inDict['relative_err'])
    self.smooth = bool(inDict['smooth'])
    self.binning = inDict['binning']
    self.probDist = inDict['prob_dist']
    self.samplingAng = inDict['angle_range'][2]
    self.pngPrefix = str(inDict['prefix'])
    self.plotFit = bool(inDict['plot_fit'])
    self.weightFit = bool(inDict['weight'])
    if args.smooth: self.pngPrefix += '_smooth'

    # we're assuming all profiles were run over the same amount of 
    # angles and g-points
    self.nAngles = len(fluxErrList)
    fErrDims = fluxErrList[0].fluxErr.shape
    self.nProfiles = fErrDims[0]
    self.nG = fErrDims[1]
    self.iLayer = fluxErrList[0].iLayer

    self.yLab = r'$\frac{F_{1-angle}-F_{3-angle}}{F_{3-angle}}$' if \
      self.relErr else '$F_{1-angle}-F_{3-angle}$'
    self.xLab = 'Transmittance'

    if self.smooth:
      self.yLab = \
        r'($\overline{\frac{F_{1-angle}-F_{3-angle}}{F_{3-angle}}}$)'
      self.xLab += ' (Binning = %.2f)' % self.binning
    # end smooth labels

    self.diagnostic = bool(inDict['diagnostic'])
    self.tCut = float(inDict['t_cutoff'])
  # end constructor

  def makeArrays(self):
    """
    Generate and transform arrays of transmittances and flux error
    for plotting
    """

    # we wanna loop over all profiles and append their err curves
    # onto each other so we can combine all profiles' err vs. t onto 
    # same plot

    # initialize lists and arrays
    angles, profs, sigmas = [], [], []
    err = np.zeros((self.nProfiles, self.nAngles, self.nG))

    for iAng, angObj in enumerate(self.allObj):
      # these two lists will end up having nAng and nProf dim
      angles.append(angObj.angle)

      # transmittance and weights are the same for every angle run
      tran = angObj.transmittance
      weights = 1/angObj.fluxRef.flatten()
      err[:, iAng, :] = angObj.fluxErr
    # end loop over fluxErr objects

    self.angles = np.unique(angles)

    # combine transmittances from all profiles together
    self.transmittance = np.array(tran).flatten()

    # this conditional really should never fail
    if self.angles.size != self.nAngles:
      print('Cannot continue, angles are not consistent')
      sys.exit(1)
    # endif angles

    # want a [nAng x (nProf * nG)] 2-D array of errors where we 
    # combine all of the errors from profiles
    temp = np.transpose(np.array(err), axes=(1, 0, 2))
    self.err = temp.reshape(\
      (self.nAngles, self.nG * self.nProfiles))

    # now we need to filter out low transmittances, since the err vs.
    # angle behavior is not well-behaved
    iTran = np.where(self.transmittance >= self.tCut)[0]
    if iTran.size == 0: sys.exit('No viable transmittances, exiting')

    self.transmittance = self.transmittance[iTran]
    self.err = self.err[:,iTran]
    self.weights = np.array(weights)[iTran]

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

  def calcStats(self, sums=False):
    """
    Calculate statistics for the errors for each angle

    Keywords
      sums -- boolean, print sums of diff and err squares instead of
        the moments of the error array
    """

    err = np.array(self.err)
    self.errAvg = err.mean(axis=1)
    self.errAbsAvg = np.abs(err).mean(axis=1)
    self.errSpread = err.std(ddof=1, axis=1)

    for iAng, ang in enumerate(self.angles):
      if sums:
        # are the sum of squares equal magnitudes? they should be 
        # with the fits we're doing
        diff = err[iAng, :]
        iNeg = np.where(diff < 0)[0]
        iPos = np.where(diff > 0)[0]
        if iAng == 0:
          print('%10s%10s%10s%10s' % ('neg2', 'pos2', 'neg', 'pos'))

        print('%10.3f%10.3f%10.3f%10.3f' % \
          ((diff[iNeg]**2).sum(), (diff[iPos]**2).sum(), \
           diff[iNeg].sum(), diff[iPos].sum()))
      else:
        # print statistics to standard output
        if iAng == 0:
          print('%10s%15s%15s%10s' % \
            ('Ang', 'Mean Err', 'Mean |Err|', 'SD Err'))

        print('%10.2d%15.4f%15.4f%10.4f' % \
          (self.angles[iAng], self.errAvg[iAng], \
           self.errAbsAvg[iAng], self.errSpread[iAng]) )
      # endif sums
    # end angle loop
  # end calcStats()

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
    Fit a cubic polynomial to the error-t curve for each angle for 
    better determination of the root
    """

    leg, fits, roots, newAng = [], [], [], []

    # well keep the data (for plotHist) as well, but we do need to 
    # rebuild the array in case some of the angles don't have any 
    # roots
    origDat = []

    tran = np.array(self.transmittance)
    for iErr, errAng in enumerate(self.err):
      # don't plot curves for all angles
      #if iErr % self.samplingAng != 0: continue

      iSort = np.argsort(tran)

      # fit a cubic spline to the curve for this angle
      # some a posteriori hand waving when there are multiple 
      # viable (0 <= t <= 1) roots...typically one of the three roots
      # for the cubics was negative and another was > 1, so i just 
      # use the "middle" one. this is fine given an appropriate 
      # range of angles
      fit = np.polyfit(tran[iSort], errAng[iSort], 3)
      fitDat = np.poly1d(fit)(tran[iSort])
      angRoots = np.roots(fit)
      if len(angRoots) == 0: continue
      root = angRoots[1]

      fits.append(fitDat)
      roots.append(root)
      newAng.append(self.angles[iErr])
      origDat.append(errAng)

      # legend string -- ?? degrees
      leg.append('%d' % self.angles[iErr] + r'$^{\circ}$')
    # end angle loop

    self.fitsErrTran = np.array(fits)
    self.rootsErrTran = np.array(roots)
    self.angles = np.array(newAng)
    self.err = np.array(origDat)
    self.secants = 1/np.cos(np.radians(self.angles))

  # end fitErrT()

  def fitErrAng(self):
    """
    Fit a cubic polynomial to the curve (err vs. angle) for each 
    transmittance for better determination of the root
    """

    tran = np.array(self.transmittance)
    iSort = np.argsort(tran)
    origTran = tran[iSort]
    err = self.err.T[iSort, :]

    fits, roots, newTran, newWeights = [], [], [], []

    self.secants = 1/np.cos(np.radians(self.angles))

    for iTran, errTran in enumerate(err):
      # np.polyfit produces a very noisy roots vs. t plot
      """
      fit = np.polyfit(self.angles, errTran, 3)
      fitDat = np.poly1d(fit)(self.angles)

      # convert roots to real array (there is no imaginary part, but 
      # np.roots returns an imaginary root)
      angRoots = np.abs(np.roots(fit))
      angRoots = np.roots(fit)
      fits.append(fitDat)

      # what's the correct root? this is a little ad hoc since we know
      # to be doing this analysis between 48 and 58 degrees
      iRoot = np.where((angRoots >= 20) & (angRoots <= 60) & \
        np.isreal(angRoots))[0]
      roots.append(float(np.abs(angRoots[iRoot])))
      """

      # UnivariateSpline produces a much nicer roots vs. t plot
      # but this really should be the same thing as np.polyfit to 
      # degree 3 -- maybe this root finder is better? it's simpler...
      fit = INTER.UnivariateSpline(self.secants, errTran)
      tRoots = fit.roots()
      if len(tRoots) == 0: continue

      roots.append(float(tRoots))
      fits.append(fit)
      newTran.append(origTran[iTran])
      newWeights.append(self.weights[iTran])
    # end tranErr loop

    self.fitsErrAng = np.array(fits)
    self.rootsErrAng = np.array(roots)
    self.transmittance = np.array(newTran)
    self.weights = np.array(newWeights)
  # end fitErrAng()

  def fitAngT(self, rootsStr='rootsErrAng'):
    """
    Using the roots determined in fitErrT, now generate a function of
    optimal angle dependent on transmission (i.e., roots)

    Keywords
      NEITHER OF THESE HAVE BEEN IMPLEMENTED INTO THE MAIN(), SO THERE
      IS NO REAL FLEXIBILITY HERE UNLESS THE USER CHANGES THE VALUES 
      OF THESE KEYWORDS MANUALLY

      rootsStr -- string, which roots array to use in the fitting
        should either be "rootsErrAng" (fit of error to angles for 
        every transmittance) or "rootsErrT" (fit of error to 
        transmittance for every angle)
      weights -- boolean, calculate weights for each bin
    """

    tran = np.array(self.transmittance)
    roots = np.array(getattr(self, rootsStr))

    if self.weightFit:
      coeffs = np.polyfit(tran, roots, 3, w=self.weights)
    else:
      coeffs = np.polyfit(tran, roots, 3)
    # endif weights

    self.secTFit = np.poly1d(coeffs)

    if self.diagnostic:
      plot.plot(tran, self.rootsErrAng, 'bo', \
        tran, self.secTFit(tran), 'r')
      plot.xlabel('Transmittance')
      plot.ylabel('secant(roots)')
      outPNG = 'preliminary_sec_T.png'
      plot.savefig(outPNG)
      plot.close()

      print('Wrote %s' % outPNG)
    # end diagnostic
  # end fitAngT()

  def plotDist(self):
    """
    Generate probability (mass) distributions of the test-ref 
    residuals
    """

    binning = np.arange(-0.15, 0.15, 0.01)
    for iAng, angErr in enumerate(self.err):
      # calculate histogram, then normalize bins
      # plot.hist() only does probability *density* histograms, so we
      # have to make our own probability *mass* plots
      heights, bins = np.histogram(angErr, bins=binning)
      heights = heights.astype(float)/sum(heights)

      plot.bar(bins[:-1], heights, \
        width=(max(bins)-min(bins))/len(bins))
      plot.xlabel(self.yLab)
      plot.ylabel('% in Bin')
      errStr = 'Rel' if self.relErr else 'Abs'
      outPNG = '%s_Err_distribution_ang%2d.png' % \
        (errStr, self.angles[iAng])
      plot.savefig(outPNG)
      plot.close()
      print('Wrote %s' % outPNG)
    # end angle loop
  # end plotHist()
# end combineErr

class secantRecalc(fluxErr):
  def __init__(self, inDict, inFluxErr, inCombineErr):
    """
    Combine attributes from fluxErr and combineErr objects to 
    generate a netCDF with modeled optimized secants

    Input
      inFluxErr -- list of fluxErr objects; should be one object per 
        profile per experimental angle (with nGpt elements)
      inCombineErr -- combineErr object generated from inFluxErrr

    Keywords
      outFile -- string, name of output netCDF that this class 
        produces
    """

    # first inherit some fluxErr attributes, then replace with atts 
    # for this class
    fluxErr.__init__(self, 0, inDict)

    self.secModel = inCombineErr.secTFit

    # transmittances differ by profile and g-point
    # the nProf x nGpt arrays are the same for each angle, so just 
    # grab the array for the first angle
    tran = np.array(inFluxErr[0].transmittance)
    self.secant = self.secModel(tran)

    secDims = self.secant.shape
    self.nG = secDims[0]
    self.nProf = secDims[1]

    # for the rest of the analysis, we only need a t vector
    self.transmittance = tran.flatten()

    self.template = 'rrtmgp-inputs-outputs.nc'
    split = self.template.split('.')
    self.outNC = '%s_opt_ang.nc' % split[0]

    self.pngPrefix = str(inDict['prefix'])

    self.relErr = bool(inDict['relative_err'])
    self.yLab = r'$\frac{F_{1-angle}-F_{3-angle}}{F_{3-angle}}$' if \
      self.relErr else '$F_{1-angle}-F_{3-angle}$'
  # constructor

  def calcStats(self, sums=False):
    """
    Calculate statistics for the errors for each angle

    Lame...pretty much just a c/p of combineErr.calcStats()...but 
    there are some reasons we have to do this...new err definition, 
    for instance

    Keywords
      sums -- boolean, print sums of diff and err squares instead of
        the moments of the error array
    """

    err = self.fluxErr.flatten()
    tran = np.array(self.transmittance)

    self.err = np.array(err)
    self.errAvg = err.mean()
    self.errAbsAvg = np.abs(err).mean()
    self.errSpread = err.std(ddof=1)

    if sums:
      # are the sum of squares equal magnitudes? they should be with
      # the fits we're doing
      iNeg = np.where(err < 0)[0]
      iPos = np.where(err > 0)[0]
      print('%10s%10s%10s%10s' % \
        ('neg2', 'pos2', 'neg', 'pos'))
      print('%10.3f%10.3f%10.3f%10.3f' % \
        ((err[iNeg]**2).sum(), (err[iPos]**2).sum(), \
         err[iNeg].sum(), err[iPos].sum()))
    else:
      # print statistics to standard output
      print('%10s%15s%15s%10s' % \
        ('Ang', 'Mean Err', 'Mean |Err|', 'SD Err'))
      print('%10s%15.4f%15.4f%10.4f' % \
        ('Opt', self.errAvg, self.errAbsAvg, self.errSpread) )
    # endif sums
  # end calcStats()

  def plotErrT(self):
    """
    Plot flux error as a function of transmittance for optimized angle
    """

    outPNG = '%s_flux_errors_transmittance_opt_ang.png' % \
      self.pngPrefix
    tran = np.array(self.transmittance.flatten())
    err = np.array(self.err)
    iSort = np.argsort(tran)
    plot.plot(tran[iSort], err[iSort], 'o')

    # aesthetics
    plot.ylabel(self.yLab)
    plot.xlabel('Transmittance')
    plot.title('Flux Error, Optimized')
    plot.gca().axhline(0, linestyle='--', color='k')
    plot.savefig(outPNG)
    plot.close()
    print('Wrote %s' % outPNG)

  # end plotErrT()

  def plotDist(self, tBinning=False):
    """
    Generate probability (mass) distributions of the test-ref 
    residuals

    Keywords
      tBinning -- boolean, plot separate histograms of errors for 
        different transmittance bins (0 to 1 in bins of 0.1)
    """

    err = np.array(self.err)

    binning = np.arange(-0.10, 0.10, 0.01)

    # calculate histogram, then normalize bins
    # plot.hist() only does probability *density* histograms, so we
    # have to make our own probability *mass* plots
    heights, bins = np.histogram(err, bins=binning)
    heights = heights.astype(float)/sum(heights)

    plot.bar(bins[:-1], heights, \
      width=(max(bins)-min(bins))/len(bins))
    plot.xlabel(self.yLab)
    plot.ylabel('% in Bin')
    errStr = 'Rel' if self.relErr else 'Abs'
    outPNG = '%s_Err_distribution_optAng.png' % errStr
    plot.savefig(outPNG)
    plot.close()
    print('Wrote %s' % outPNG)

    errStr = 'Rel' if self.relErr else 'Abs'
    if tBinning:
      tBins1 = np.arange(0, 1, 0.1)
      tBins2 = np.arange(0.1, 1.1, 0.1)
      tran = np.array(self.transmittance.flatten())

      print('%-15s%15s%15s%15s' % \
        ('t Bin Start', 'Mean Err', 'Mean |Err|', 'Sigma Err'))

      for t1, t2 in zip(tBins1, tBins2):
        iBin = np.where((tran >= t1) & (tran < t2))[0]
        if iBin.size == 0: continue
        outPNG = '%s_Err_distribution_optAng_t%3.1f.png' % \
          (errStr, t1)
        binErr = err[iBin]

        print('%-15.1f%15.4e%15.4e%15.4e' % \
          (t1, binErr.mean(), np.abs(binErr).mean(), \
           binErr.std(ddof=1)))

        # now just do what we did for the entire sample
        heights, bins = np.histogram(binErr, bins=binning)
        heights = heights.astype(float)/sum(heights)

        plot.bar(bins[:-1], heights, \
          width=(max(bins)-min(bins))/len(bins))
        plot.xlabel(self.yLab)
        plot.ylabel('% in Bin')
        plot.ylim([0, 1])
        plot.title('%3.1f ' % t1 + r'$\leq$ ' + 't < %3.1f' % t2)
        plot.savefig(outPNG)
        plot.close()
        #print('Wrote %s' % outPNG)
      # end bin loop
    # end tBinning
  # end plotHist()
# end secantRecalc

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
    default='lw_solver_opt_angs', \
    help='RRTMGP flux solver executable for fluxErr and ' + \
    'secantRecalc classes, respectively.')
  parser.add_argument('--template_nc', '-temp', type=str, \
    default='rrtmgp-lw-inputs-outputs-clear.nc', \
    help='netCDF that is used as input into executable. The ' + \
    'code will copy it and use a naming convention that the ' + \
    'executable expects.')
  parser.add_argument('--secant_nc', '-snc', type=str, \
    default='optimized_secants.nc', \
    help='Path to netCDF file to which secant array is written.')
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
  parser.add_argument('--print_sums', '-ps', action='store_true', \
    help='Print sums of positive diffs and negative diffs ' + \
    'to make sure the fit minimized the errors (so these two ' + \
    'quantities should be equal).')
  parser.add_argument('--diagnostic', '-d', action='store_true', \
    help='Plot diagnostic secant(roots) vs. transmittance to ' + \
    'see how well first fit does.')
  parser.add_argument('--t_cutoff', '-c', type=float, default=0.05, \
    help='Cutoff in t under which flux errors are ignored.')
  parser.add_argument('--weight', '-w', action='store_true', \
    help='Weight the fits with reference fluxes.')
  args = parser.parse_args()

  angles, res = args.angle_range, args.angle_resolution
  pFile = args.save_file

  # first make an nProf x nAngle x nGpoint array of flux differences
  if os.path.exists(pFile):
    fErrAll = pickle.load(open(pFile, 'rb'))
  else:
    # loop over all angles (inclusive), generating a fluxErr object 
    # for each angle and profile, which we'll combine using another 
    # class; fErrAll contains objects for all angles
    fErrAll = []
    for ang in np.arange(angles[0], angles[1]+res, res):
      print('Running calculations at %d degrees' % ang)
      fErr = fluxErr(ang, vars(args))
      fErr.refExtract()
      fErr.writeSecNC()
      fErr.runRRTMGP()
      fErrAll.append(fErr)
    # end angle loop

    # save the output for later plotting
    pickle.dump(fErrAll, open(pFile, 'wb'))
    # end profile loop
  # endif pickle file

  # combine the flux error arrays for plotting and fitting
  combObj = combineErr(fErrAll, vars(args))
  combObj.makeArrays()
  combObj.calcStats(sums=args.print_sums)
  
  if args.plot_fit:
    combObj.fitErrAng()
    combObj.fitErrT()
    combObj.fitAngT()
    #combObj.plotDist()

    # use the fit of angle vs. transmittance to model optimal angle 
    # for all g-points and profiles
    reSecObj = secantRecalc(vars(args), fErrAll, combObj)
    reSecObj.refExtract()
    reSecObj.writeSecNC()
    reSecObj.runRRTMGP()
    reSecObj.calcStats(sums=args.print_sums)
    reSecObj.plotErrT()
    reSecObj.plotDist()
    reSecObj.plotDist(tBinning=True)
  else:
    combObj.plotErrT()
    # probably should never be used anymore...
    #combObj.plotThetaOptT()
  # endif plot_fit

# end main()

