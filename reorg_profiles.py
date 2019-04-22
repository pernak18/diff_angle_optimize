#!/usr/bin/env python

import os, sys, argparse, glob
import subprocess as sub

sys.path.append('common')
import utils

parser = argparse.ArgumentParser(\
  description='Generate a subdirectory structure for the PDFs ' + \
  'written by parallel_e2e.py and extract_mean_profile.py.')
parser.add_argument('--indir', '-d', type=str, \
  default='profile_plots/linear', \
  help='Directory to restructure.')
args = parser.parse_args()

# grab all of the PDFs to use in restructuring
inDir = args.indir; utils.file_check(inDir)
inFiles = sorted(glob.glob('%s/*.pdf' % inDir))
nFiles = len(inFiles)

# grab the substrings from each file that indicate the coefficients 
# that we used for the flux calculations
# we're making some assumptions about file naming convention here, 
# namely what we use in make_config_files.py
# e.g., for Band 15: lin_coeff_pos0.50_1.50_15.pdf
coeffs = []
for pdf in inFiles:
  split = pdf.split('_')
  combCoeff = ''.join(coeffs)
  for iSub, substr in enumerate(split):
    if ('pos' in substr) or ('neg' in substr):
      coeff = '%s_%s' % (substr, split[iSub+1])
    else:
      continue
    # endif substr

    # coeff is repeated for all 16 bands, 
    # and we only need to keep it once
    if coeff not in combCoeff: coeffs.append(coeff)
  # end split loop
# ened pdf loop

# new (coefficient-dependent) subdirectories will be written 
# underneath the original directory
for coeff in coeffs:
  outDir = '%s/%s' % (inDir, coeff)
  if not os.path.exists(outDir): os.makedirs(outDir)
  matches = sorted(glob.glob('%s/*%s*.pdf' % (inDir, coeff)))

  for match in matches:
    os.rename(match, '%s/%s' % (outDir, os.path.basename(match)))
  # end match loop
# end coefficient loop

