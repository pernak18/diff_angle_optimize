# Diffusivity Angle Optimization

# Cloning <a name="cloning"></a>

```
git clone --recursive git@github.com:pernak18/diff_angle_optimize.git
```

Note the `--recursive` option. This will clone the submodules necessary for this repo (namely, https://github.com/AER-RC/common).

# Running the Software <a name="running"></a>

## Prerequisites <a name="prereq"></a>

A driver for the RRTMGP function `lw_solver_noscat` was written for the purpose of this optimization. Its source is located in the `lw_solver_noscat` subdirectory. The user will have to edit the `Makefile` so that the paths to the dependencies are correct (and likely will have to set the `RRTMGP` environment variable). The driver -- `lw_solver_opt_angs` -- can be built with a simple `make all` in the directory. `lw_solver_1ang.F90` can be ignored.

## Steps in processing <a name="steps"></a>

  1. Extract reference (i.e., 3-angle) RRTMGP fluxes and calculate transmittances given optical depths
  2. Write a netCDF (`optimized_secants.nc`) that contains only an array of angles to be used in the RRTMGP executable. This array is `ngpt x ncol` and thus is a function of spectral point and profile, but in the early stages (i.e., single-angle calculations), all of the elements in the array are equivalent.
  3. Run RRTMGP `lw_solver_opt_angs` at a given angle, given `optimized_secants.nc` and `rrtmgp-lw-inputs-outputs-clear.nc`, the latter of which is a netCDF with by-g-point fluxes in it. These fluxes will be overwritten with fluxes from the new runs at a given angle.
  4. Calculate the difference between items 3 and 1 for a given angle
  5. Combine objects for many angles together
  6. For each data point (e.g., each transmittance point), fit a cubic function to the error in 4 as a function of angle
  7. Find the root of the function fit in 6. This will be the secant of the optimized angle.
  8. Combine the roots for each data point onto a plot of `secant(theta)` as a function of transmittance
  9. Fit a curve to the data set in 8.

For optimization, the user should loop over a number of angles (as specified in the `argparse` arguments, documented in [Table 1](#Table1)), and process steps 1 through 4. A separate `fluxErr` object class needs to be instantiated for each angle, and the objects for all angles are combined in a `combineErr` object (steps 5 through 7). This new `combineErr` object is then used with a list of `fluxErr` objects in another class -- `secantRecalc` -- where steps 8 and 9 are performed.

## Using `main` in angle_optimize.py <a name="main"></a>

There are no required arguments for `angle_optimize.py` -- all arguments have default values assigned to them. However, the code has evolved substantially since its initial draft, and some default options may not work and certainly have not been tested. A typical call for downwelling fluxes is:

```
angle_optimize.py -pre Garand_rel_down -r -a 48 58 3 -d -c 0.00 -w -fit
```

To summarize the inputs:

  - **-pre**: prefix for output files (is only used with one PNG -- the errors as a function of transmittance for all specified angles)
  - **-r**: process relative errors instead of absolute errors
  - **-a**: angle range (degrees) and resolution over which to *plot* them (netCDF files for every degree between `ang1` and `ang2` are generated, and the "resolution" argument is really just for aesthetic purposes)
  - **-d**: plot a diagnostic figure of roots (secants) versus transmittance for every g-point and all profiles. also overplot the associated fit
  - **-c**: ignore all transmittance points below this cutoff
  - **-w**: use modified weights of reference flux/root uncertainty
  - **-fit**: instead of plotting flux errors as a function of transmittance, follow through with the optimization analysis and produce figures for each step (flesh this out)

And for upwelling:

```
angle_optimize.py -pre Garand_rel_up -r -a 48 58 3 -d -c 0.00 -w -fs gpt_flux_up -l 42 -fit
```

Inputs are mostly identical to the downwelling run, but we have to specify the string to use for flux extraction from the reference (3-angle) netCDF with the `-fs` argument, and the layer index `-l` is the TOA for upwelling (rather than surface for downwelling).

A full list of arguments is provided in [Table 1](#Table1). Note these arguments are also provided to the three classes in the `angle_optimize.py` module as attributes. Documentation for these keywords is also provided when calling `angle_optimize.py` with `-h`.

**`angle_optimize.py` Arguments** <a id="Table1"></a>

| Argument | Notes |
| :---: | :---: |
| flesh |  |
| this |  |
| out |  |

## Completing the Optimization <a name="complete">

In [`angle_optimize.py`](#main), we find separate optimized solutions for downwelling and upwelling so that we can quickly determine fluxes only based on a given transmittance. But we effectively develop two models with two different training data sets. For a given column, we should have one solution such that the optimal angle for flux calculations is independent of viewing geometry. We accomplish the combined optimization by applying [steps 8 and 9](#steps) with upwelling and downwelling data points together. `combine_up_down.py` was used to do this, but the results were not acceptable, so we abandoned this approach. From my notes, this is the sequence we used:

```
% angle_optimize.py -pre Garand_rel_down -s save_files/Garand_optimize_diffusivity_angle_weighted_relerr_up_48-58.p -r -a 48 58 3 -d -c 0.2 -w -ss # Garand_rel_down_flux_errors_transmittance.png
% angle_optimize.py -pre Garand_rel_up -s save_files/Garand_optimize_diffusivity_angle_weighted_relerr_up_48-58.p -r -a 48 58 3 -d -c 0.2 -w -fs gpt_flux_up -l 42 -ss # Garand_rel_up_flux_errors_transmittance.png

and added the code to combine_up_down.py to write a new optimized_secants.nc:

% ./combine_up_down.py

which is used in:

% ./lw_solver_opt_angs 

which writes rrtmgp-inputs-outputs.nc, which is the netCDF used in our plotting code.
```

## By-Band Optimization <a name="band">

The optimization analysis until now has been over all g-points for all bands and profiles, so fitting has been done for 16 (g-points per band) x 16 (bands) x 42 (profiles). Now we only weight the data points with 1/flux.

```
downwelling (in Bash because I need Python 3):
% bash
% for BAND in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do echo $BAND; angle_optimize.py -pre Garand_rel_down -r -a 48 58 3 -d -c 0.00 -w -fit -ss -bb -bn $BAND -pr 3; done
% for BAND in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do echo $BAND; angle_optimize.py -pre Garand_rel_up -r -a 48 58 3 -d -c 0.00 -w -fs gpt_flux_up -l 42 -fit -ss -bb -bn $BAND -pr 3; done

% wrapper_combine_up_down.py -u band_calculations/full_band/flux_weights/up/linear -d band_calculations/full_band/flux_weights/down/cubic -o band_calculations/full_band/flux_weights/combined
Replaced flux fields in rrtmgp-inputs-outputs_opt_ang_merged.nc
rrtmgp-inputs-outputs_opt_ang_merged.nc copied to /rd47/scratch/RRTMGP/paper/rte-rrtmgp-paper-figures/data/rrtmgp-lw-inputs-outputs-optAng.nc
```

The wrapper script also includes a process called bandmerging. Bandmerging has to happen because of how I have designed the code. It works with a single RRTMGP netCDF, so the dimensions are `ncol`, `nband`, and either `nlev` or `nlay`. `nband` is always 16, so lest we break the code, I have stuck with that convention and utilize the `band_lims_gpt` array in the netCDFs to extract only the information (e.g., fluxes) for a given band. We do this for each band, so we end up having 16 netCDFs (one for each band), and we have to piece together information for all bands into a single netCDF that can be used for the rest of the analysis. We also do the band calculations in this script, so we get from g-point fluxes to band and broadband fluxes.

The `--mv` keyword moves the bandmerged result to a directory where we can run our Jupyter notebook and generate figures for the RRTMGP paper.

## Plotting the By-Band Optimization <a name="plotting">

Since we produce an RRTMGP netCDF with the optimized result, we can use it in the plotting scripts we generated for RRTMGP band generation and validation. The code and configuration file to do this has been added to this repository:

```
% LBLRTM_RRTMGP_compare.py --config_file rrtmgp_lblrtm_config_garand_opt_ang.ini --plot_profiles --band 2 15
```

The configuration file will almost always point to the `rrtmgp-inputs-outputs_opt_ang_merged.nc` netCDF generated in [By-Band Optimization](#band). The code generates profile (vertical flux and heating rate comparisons of RRTMGP and LBLRTM, 1 page per column, 1 PDF per band) and stats plots (single PDF with flux and heating rate statistics for all columns, 1 page per band).

## Modifying the Optimization <a name="modify">

End users may not be satisfied with the "complete" solution that was found in [band optimization](#band). For now, we are only fitting lines to the data points because higher-order polynomials were even worse. It is not a simple process of finding the appropriate curve to fit, especially when combining the up and down solutions, so some scripts were created that allow the user to examine a number of different linear coefficient pairs and their influence on the RRTMGP fluxes with respect to LBLRTM. This process utilizes a streamlined, multithreaded process that uses many of the modules in this repository.

```
% echo $0
bash
% pwd
/nas/project/p2097/rpernak/diff_angle_optimize
% python -V
Python 3.6.8 :: Anaconda, Inc.
% ./e2e_modify_optimization.py 
```

`e2e_modify_optimization.py` allows the user to edit a configuration file like `rrtmgp_lblrtm_config_garand_opt_ang.ini` and provide the RRTMGP code with coefficients that differ from the optimized result, which are stored in `sec_t_fit_coeffs.csv` (original version is in version control). The user can change the coefficients for any band. After reading the `.ini` file, the optimized netCDF files for each band are gathered, then optimized angles based on the new user-provided coefficients are calculated, and then these angles are used in RRTMGP. The results are then plotted either as stats or profiles (which is specified in the input configuration file).

If stats and profiles are wanted, or if y-log profiles are more appropriate, a different configuration file should be written for each instance, and the computation can be minimized by running e2e_modify_optimization.py in parallel with:

```
% parallel_e2e.py -i rrtmgp_lblrtm_config_garand_opt_ang.ini dummy2.ini dummy.ini
```

`-i` can take any number of `.ini` files as input and will make each PDF file in a separate CPU core.

Another request: find the optimized angle extrema from the entire g-point analysis, then loop through a number of linear coefficients (same trials for each band) in 0.05 increments. This generates a "tapezoid" of linear coefficient pairs in coefficient space -- see `coeff_trial_error.py` (*for now, this only works with the default configuration*). A configuration file for each of these pairs (for each band, and all bands have the same coefficients) needs to be made, and this can be done with `make_config_files.py`, which automates the generation of all of the `.ini` files we'll need.

```
% make_config_files.py 
```

That's a lot of plots -- 99 coefficient pairs and 16 bands. We can reorganize with:

```
% reorg_profiles.py 
% reorg_profiles.py -d profile_plots/log
% reorg_profiles.py -d profile_plots/mean_linear
% reorg_profiles.py -d profile_plots/mean_log
```

