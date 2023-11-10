[![Build Status](https://jenkins.h-its.org/buildStatus/icon?job=MBM%2FHITS-MBM%2FColBreaker%2Fmain)](https://jenkins.h-its.org/job/MBM/job/HITS-MBM/job/ColBreaker/job/main/)

# ColBreaker

Simulation of a super coarse grained collagen model, that is breakable under force. For details, we refer to the publication


## Status

Python version 1.0, contact benedikt.rennekamp@h-its.org for help. Code should be fully usuable, but main purpose of this repository is transparency in reviewing. A clean-up C++ version might become available in the future.


## Simulation parameters

Following parameters can be defined in the main file:

  - `allow_switches`: If crosslink would go outside the model, switch direction to retrieve fully crosslink network (default: True)
  - `build_up_phase`: boolean (default: True)
  - `C_path_difference_factor`: (default: 1.08) Length of C-terminal path in crosslinked phase compared to standard contourlength
  - `connectedness`: (default: 1.0)
  - `constant_force`: [N] (default: 3.5e-9)
  - `contour_factor`: (default: 1.2) Contour length is longer by this factor than initial distance
  - `crosslink_sites`: Side of the gap region crosslinks will be made [`N`, `C`, `BOTH`] (default: `BOTH`)
  - `d_phase_arr`: Array (default: [1, 4, 3, 2, 0])
  - `data_filename`: Filename (*.csv) for data (default: `data.csv`)
  - `diffusion_constant`: Diffusion constant [m^2/s] (default: 4e-10)
  - `dissociation_energy_in_kT`: Depth of morse potential [multiple of kT] (default: 119)
  - `double_crosslinks`: Use this option to double the amount of crosslinks: Connect both up and downwards (default: False)
  - `dt`: Integration time step [s] (default: 1e-15)
  - `dx_max`: Maximal integration step size to ensure accurate integration of Smoluchowski equation [m] (default: 1e-10)
  - `force_constant_in_kT`: Force constant for velocity pulling [multiple of kT] (default: 3.5e18)
  - `force_constant_pwWLC`: Force constant for enthalpic strechig in piecewise WLC [N/m] (default: 2.05e-8)
  - `gap_ratio`: (default: 0.54)
  - `k_f`: [N/m] (default: 440.5 by bondtype C-CT from amber99sb* ffbonded)
  - `layers`: Only use 1+4*x to get x full repeats of a TH (default: 5)
  - `low_up_ratio`: Use 1.0 for all crosslinks going to (standard) upper side (default: 1.0)
  - `max_extension_percentage`: Percentage of fibril length (default: 0.23)
  - `N_path_difference_factor`: (default: 1.0) Length of N-terminal path in crosslinked phase compared to standard contourlength
  - `pdb_filename`: Filename (*.pdb) for trajectory (default: `trajectory.pdb`)
  - `pdb_scaling_factor`: (default: 0.01)
  - `periodicity`: (default: 6.7e-8)
  - `persistence_length`: [m] (default: 1.45e-8 by [Sun et al. 2002](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1304183/) using optical tweezer)
  - `plot_freq`: Plot only every x-th frame (default: 100)
  - `pulltype`: [`VELOCITY`, `FORCE`, `STRAIN`] (default: `FORCE`)
  - `r_fb`: Flat bottom extra extension of crosslink (default: 1e-9)
  - `seed`: Random number seed (-1 for random seed) (default: -1)
  - `spacing`: (default: 1e-8)
  - `TH_per_Side`: Number of triple helices per side in hexagon (default: 2)
  - `time_total`: Simulation time [s] (default: 1e-11)
  - `tol`: Tolerace for identifying equal positions with floats [m] (default: 1e-11)
  - `v_pull`: [m/s] (default: 25)
  - `wlc_functional`: [`CLASSICAL`, `PIECEWISE`] (deault: `PIECEWISE`)
  - `WLC_max`: Cut-off force for WLC to mimick backbone rupture [N] (default: 7.5e-9)
  - `write_freq`: Write-out frequency in steps (default: 100)
