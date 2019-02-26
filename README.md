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

