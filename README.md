# Diffusivity Angle Optimization

# Cloning <a name="cloning"></a>

```
git clone --recursive git@github.com:pernak18/diff_angle_optimize.git
```

Note the `--recursive` option. This will clone the submodules necessary for this repo (namely, https://github.com/AER-RC/common).

# Running the Software <a name="running"></a>

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

Ideally, the user will loop over a number of angles (as specified in the `argparse` arguments), and processing steps. Each angle is its own object (`fluxErr` class), so the objects for all angles are combined (in the `combineErr` class). This new `combineErr` object is then used with the list of `fluxErr` objects in another class -- `secantRecalc` -- where 

## Using `main` in angle_optimize.py <a name="main"></a>


