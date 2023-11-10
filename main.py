"""
ColBreaker
by Benedikt Rennekamp, status: 10.11.2023


 aim: simulate pulling experiments on coarse grained collagen model
      hexagonal bundle of fibers, with four crosslinks, each end two
      each polymer segment described by worm-like chain
      crosslink by different (e.g. gaussian well / flat-bottom Morse) potentials
"""

import matplotlib.pyplot as plt
import functions as func
import numpy as np
import random
from numpy.random import rand
import logging
import glob
from matplotlib import cm
import pandas as pd
import matplotlib.ticker as ticker
import plot
import time as t
import math
import os

# Workspace
dir_name = "run_test" # create a sub-directory with this name and store all stuff there
func.create_dir(dir_name)
func.change_to_dir(dir_name)

logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')


### Simulation Parameters
time_total = 5000.0 * 10**-9 # simulation time [s] #10-9 is nano seconds
nr_steps = 2500000000
dt = time_total / nr_steps  # integration time step [s]. 
# [m] maximal integration step size to ensure accurate integration of smoluchowski equation
dx_max = 1.0 * 10**-10
tol = 1.0 * 10**-11  # [m] tolerace for identifying equal positions with floats
# always uses the same random seed if e.g. random.seed(123). Change to
#seed = 124
#random.seed(seed)
random.seed() #new numbers every time

### plot parameters
plotflag = True
nbr_frames = 5000
plot_skip =  nr_steps / nbr_frames # plot only every x-th frame

#write-out frequency to (pandas) csv
write_feq = 1000 #in steps

### Topological parameters
layers = 14  # only use 1+4*x to get x full repeats of a TH
TH_per_Side = 5  # nbr of triple helices per side in hexagon
d_phase_arr = [1, 4, 3, 2, 0]

colors = [cm.viridis(i / (len(d_phase_arr) - 1))
          for i in range(len(d_phase_arr))]  # colors for the phases
spacing = 10.0 * 10**-9
periodicity = 67.0 * 10**-9
fibril_length = layers * periodicity

gap_ratio = 0.54
overlap_ratio = 1.0 - gap_ratio
overlap_length = overlap_ratio * periodicity
gap_length = gap_ratio * periodicity
N_path_difference_factor = 1.0 #length of N-terminal path in crosslinked phase compared to standard contourlength
C_path_difference_factor = 1.03 #2.0 - N_path_difference_factor
crosslink_sites = 'both' # 'N', 'C' or 'both' to determine on which side of the gap region crosslinks will be made
low_up_ratio = 1.0 #use 1.0 for all crosslinks going to (standard) upper side
allow_switches = False #if crosslink would go outside the model, switch direction to retrieve fully crosslink network 
double = False #use this option to double the amount of crosslinks: Connect both up and downwards. 
#Was called 'trivalent' in Anna's topology model, which however does not actually correspond to trivalent crosslinks. 


### Physics parameters
kT = 2.49 * 10**3 / (6.022 * 10**23) #1.381 * 10**-23 * 300.0  # [J]
D = 0.4 * 10**-9    # diffusion constant [m^2/s]
# persistence length 14.5nm [m] by Sun et al 2002 using optical tweezer 14.5nm
#pwWLC

contour_factor = 1.17 # contour length is longer by this factor than initial distance 
pl = 0.45 * 10**-9 #[nm]
K0 = 20.5 *10**-9 #[N/m] spring constant for enthalpic strechig in pwWLC

min_wlc_length = 2.0 * 10**-9
Edis = 119 * kT  # depth of morse potential #119 =kT lowest BDE in HLKNL 296kJ/mol l
Edis_bb = 137 * kT  # depth of morse potential at the end of WLC to mimik backbone rupture #CA-C BDE
#296kJ/mol lowest value in quantum calculation is about  119 kbT
#323.7kJ/mol would be (meaningless?) average in HLKNL corresponding to 134
#sig_cross = 0.4 * 10**-9  # width of gaussial cross linker potential well [m].
r_fb = 1.0 * 10**-9  #flat bottom extra extension of crosslink
k_f =  440.5 # N/m = #265265 kJ/(mol*nm**2) = C-CT bond # 10 = effective potential when pulled 
beta = np.sqrt(k_f / (2*Edis)) 

#cut-off force for WLC to mimick backbone rupture
WLC_max = 7100 * 10**-12 #Use  1333 * 10**-12 if k_f = 10   #[N] bzw in e-12 it is pN


### pulling parameters
pulltype = 'strain' #options: (constant) velocity, force or strain
if pulltype == 'velocity': 
    v_pull = 250.0  # alternative: v_pull_per_repeat * layers [m/s]
    k0 = kT * 10.0 * 10**17 #spring constant for vel pull. Note kT * 10**17 is about 0.0004 in SI
    pull_parameters = np.array([pulltype, v_pull, k0])
    pullparameters_log = pulltype + ': v_pull = ' + str(v_pull) + '. spring constant = ' + str(k0)
elif pulltype == 'strain':
    v_pull = 25.0  # use to build up until max strain
    max_extension = 0.205 * fibril_length #position of virtual spring, i.e. max extension if no counter force #0.23 is a well-balanced one
    k0 = kT * 200.0 * 10**17 #Note kT * 10**17 is about 0.0004 in SI  a typical value in velocity pulling is 500 kJ/(mol*nm) = 0.83 N/m , which is rather on the stiff upper side 
    pull_parameters = np.array([pulltype, v_pull, k0, max_extension])
    pullparameters_log = pulltype + ': max_extension = ' + str(max_extension) + '. spring constant = ' + str(k0)
elif pulltype == 'force':
    build_up_phase = True
    v_pull = 500.0  # use to build up until max strain
    k0 = kT * 10.0 * 10**17 #spring constant for vel pull. Note kT * 10**17 is about 0.0004 in SI # a typical value in velocity pulling is 500 kJ/(mol*nm) = 0.83 N/m , which is rather on the stiff upper side 
    constant_force = 7700 * 10**-12 #[N] bzw in e-12 it is pN
    pull_parameters = np.array([pulltype, constant_force])
    pullparameters_log = pulltype + ': constant force = ' + str(constant_force)

if __name__ == "__main__":

    logging.info("####---Started " + str(dir_name)+" of statistical Collagen MD ---#####")
    print('Start: ' + dir_name)
    logging.info('Using the following set of input parameters: ')
    logging.info('time_total = ' + str(time_total) + '. time step = ' + str(dt) + ' with ' + str(nr_steps) + '  steps.')
    logging.info('contour_factor = ' + str(contour_factor) + '. N_path_difference = ' + str(N_path_difference_factor) +' to ' + str( C_path_difference_factor))
    logging.info('persistence length = ' + str(pl) + '. V_cross = ' + str(Edis) + '. r_fb = ' + str(r_fb)+ '. beta = ' + str(beta) )
    logging.info('pulling options: ' + pullparameters_log)
    logging.info('Max force cut-off in WLC: ' + str(WLC_max))
    logging.info('Crosslink positions: ' + str(crosslink_sites) + ' and low-up-ratio: ' + str(low_up_ratio))
    

    ### Create initial topology
    pullatoms_left = []  
    pullatoms_right = []
    middle_layer_atoms = []
    # first layer
    coords = func.computeInitialCoordinates(TH_per_Side)
    G, pullatoms_left = func.defInitialNodes(coords, spacing, d_phase_arr)
    G, pullatoms_right, middle_layer_atoms = func.constructMiddleLayer(G,  middle_layer_atoms, overlap_length, overlap_ratio)
    # other layers
    last_layer = False
    for i in range(layers - 1):
        if i == (layers - 2):
            last_layer = True
        G, pullatoms_right, middle_layer_atoms = func.constructNextD(G, middle_layer_atoms, spacing, overlap_length, overlap_ratio,
                                                                     gap_length, gap_ratio, last_layer)

    G, cross, pullatoms_left = func.generateCrosslinks(G, pullatoms_left, middle_layer_atoms, connectedness=1.0,
        low_up_ratio=low_up_ratio, allow_switches = allow_switches, side = crosslink_sites, double = double)
    N_cross, C_cross = func.find_crosslink_position(middle_layer_atoms, cross)
    #print('N_term crosslinks:'  + str(N_cross))
    #print('C_term crosslinks:'  + str(C_cross))
    pullatoms_left = func.delete_unconnected_pullatom(G, cross, pullatoms_left)


    start_coords, dph = func.get_coords(G)

    # counters and sanity checks
    nr_points = len(start_coords)
    print ('We have this number of points: ' + str(nr_points))
    nr_pullatoms = len(pullatoms_left)
    if nr_pullatoms != len(pullatoms_right):
        print('Warning: Number of pull atoms should be equal on both sides. These are the pull atoms: ')
        print(pullatoms_left, pullatoms_right)
        logging.warning('Warning: Number of pull atoms should be equal on both sides. These are the pull atoms: ' + str(pullatoms_left) + ' ' + str( pullatoms_right))
    print('amount of pull atoms (left, right): ')
    print(len(pullatoms_left), len(pullatoms_right))
    wlc_1, wlc_2, contour_lengths = func.get_WLCs_and_contourlengths(
        G, start_coords, cross, contour_factor, N_path_difference_factor, C_path_difference_factor)
    nr_wlc = len(wlc_1)
    if nr_wlc != len(wlc_2):
        print('Error: Number of startpoints for WLCs not equal to number of end points')
        logging.warning('Error: Number of startpoints for WLCs not equal to number of end points')
    # view / plot layer by layer
    sliced_coords, sliced_dph = func.get_sliced_coords(G, 0)
    func.plot_points_phase_colored(sliced_coords, sliced_dph, colors)

    # plot starting config
    func.plot_starting_configuration(start_coords, dph,pullatoms_left, pullatoms_right, cross, G, colors)



    ###Start Dynamics###
    # init

    #for testing purpose: determine run times of blocks of code
    wallclock_start = t.time()
    wallclock_prev = t.time()
    timing1 = 0
    timing2 = 0
    timing3 = 0
    timing4 = 0
    timing5 = 0
    timing6 = 0
    timing7 = 0

    time = 0
    coords = start_coords[:]
    ctr_integration_problem = 0
    x0_pull_left = start_coords[pullatoms_left[0]][0]
    x0_pull_right = start_coords[pullatoms_right[0]][0]
    ctr_angle = 0
    reverse = False

    eq_length =  x0_pull_right - x0_pull_left 
    print ('Starting extension: ' +str(eq_length))
    breakage_points = []
    broken_crosslinks = []
    broken_WLC = []
    N_term_breaks = []

    stop = False


    # main loop
    for ti in range(0, nr_steps):
        time = ti * dt
        #times[ti]=time
        ### calculate forces
        f = np.zeros(nr_points)  # forces to be collected for all points

        # forces from pull atoms
        pull_forces_left = np.empty(len(pullatoms_left))
        pull_forces_right = np.empty(len(pullatoms_right))
        if pulltype == 'velocity':
            f, av_pull_force_left_ti, av_pull_force_right_ti = func.calculate_vel_pull_forces(f, coords, 
            pull_forces_left, pull_forces_right, pullatoms_left, pullatoms_right, time, x0_pull_left, x0_pull_right, v_pull, k0)
        elif pulltype == 'strain':
            f, av_pull_force_left_ti, av_pull_force_right_ti = func.calculate_strain_pull_forces(f, coords, 
            pull_forces_left, pull_forces_right, pullatoms_left, pullatoms_right, time, x0_pull_left, x0_pull_right, v_pull, k0, max_extension)
        elif pulltype == 'force':
            if build_up_phase:
                f, av_pull_force_left_ti, av_pull_force_right_ti = func.calculate_vel_pull_forces(f, coords, 
                pull_forces_left, pull_forces_right, pullatoms_left, pullatoms_right, time, x0_pull_left, x0_pull_right, v_pull, k0)
                if av_pull_force_right_ti >= constant_force: #right is positive so fine
                    build_up_phase = False
                    print('building-up force completed at time ' + str(time))
                    logging.info('building-up force completed at time ' + str(time))
            else:
                f, av_pull_force_left_ti, av_pull_force_right_ti = func.calculate_constant_pull_forces(f, 
            pull_forces_left, pull_forces_right, pullatoms_left, pullatoms_right, constant_force)
  
        
        timing1 += t.time() - wallclock_prev
        wallclock_prev = t.time()         
        #extensions
        right_positions = [ coords[atom][0] for atom in pullatoms_right]
        left_positions = [ coords[atom][0] for atom in pullatoms_left]
        av_extension_ti =  (np.average(right_positions) - np.average(left_positions)) / eq_length

        timing2 += t.time() - wallclock_prev
        wallclock_prev = t.time()   

        # forces from crosslinks
        f, broken_crosslinks, breakage_points, N_term_breaks = func.calculate_crosslink_forces(f, coords, 
        cross,  r_fb, Edis, beta, time, broken_crosslinks, breakage_points, N_cross, N_term_breaks)
        
        timing3 += t.time() - wallclock_prev
        wallclock_prev = t.time()   

        # forces from wlc segments
        for i in range(0, nr_wlc):
            if i in broken_WLC:
                continue
            l0 = contour_lengths[i]
            dx = coords[wlc_2[i]][0] - coords[wlc_1[i]][0]
            #df = func.ccalculate_WLC_force(dx, l0, kT, pl)
            df, broken = func.calculate_pwWLC_force_Morse(dx, l0, contour_factor, kT, pl, K0, beta, Edis_bb)

            if broken:
                broken_WLC.append(i)
                print('Backbone rupture between: ' + str((wlc_2[i], wlc_1[i])) + ' at time: ' + str(time))
                logging.info('Backbone rupture between: ' + str((wlc_2[i], wlc_1[i])) + ' at time: ' + str(time))
                breakage_points.append((ti, wlc_2[i], wlc_1[i]))
                print(df, f[wlc_1[i]], f[wlc_2[i]])
                print(av_pull_force_right_ti)
                continue
            '''
            if df > WLC_max:
                broken_WLC.append(i)
                print('Backbone rupture between: ' + str((wlc_2[i], wlc_1[i])) + ' at time: ' + str(time))
                logging.info('Backbone rupture between: ' + str((wlc_2[i], wlc_1[i])) + ' at time: ' + str(time))
                breakage_points.append((ti, wlc_2[i], wlc_1[i]))
                print(df, f[wlc_1[i]], f[wlc_2[i]])
                print(av_pull_force_right_ti)
                continue
            '''
            
            f[wlc_1[i]] = f[wlc_1[i]] + df
            f[wlc_2[i]] = f[wlc_2[i]] - df
            #print(df)
            #tension_wlc[i] = np.abs(df)

        timing4 += t.time() - wallclock_prev
        wallclock_prev = t.time()   
        

        # advance points according to Smoluchowski
        for k in range(0, nr_points):
            dx = func.get_smoluchowski_dx(D, dt, f[k], kT)
            coords[k][0] = coords[k][0] + dx

            if np.abs(dx) >= dx_max:
                print("Warning: Integration step of " + str(np.abs(dx)) +
                      " too large, should be below " + str(dx_max))
                logging.warning("Warning: Integration step of " +
                                str(np.abs(dx)) +
                                " too large, should be below " +
                                str(dx_max))
                ctr_integration_problem += 1
        if ctr_integration_problem > 20:  # allow a few times over soft limit then stop
            print('Too many too large integration steps. Gonna stop')
            logging.error('Too many too large integration steps. Gonna stop')
            df = pd.DataFrame(data_rows)
            df.to_csv('data_error.csv')
            break
        if np.abs(dx) >= 10 * dx_max:  # hard limit stop immediately
            print("Error: Integration step of " + str(np.abs(dx)) +
                    " nm more than ten times over " + str(dx_max))
            logging.error("Error: Integration step of " + str(np.abs(dx)) +
                    " nm more than ten times over " + str(dx_max))
            #df = pd.DataFrame(data_rows)
            #df.to_csv('data_error.csv')
            break

        timing5 += t.time() - wallclock_prev
        wallclock_prev = t.time()   

        # plot frames every plot_skip_th step
        if plotflag == True and (ti % plot_skip == 0):
            ctr_angle, reverse = func.plot_frame(
                coords, dph, pullatoms_left, pullatoms_right, wlc_1, wlc_2, broken_WLC, cross, broken_crosslinks,
                ti, nr_steps, fibril_length, colors, ctr_angle, reverse)

        timing6 += t.time() - wallclock_prev
        wallclock_prev = t.time()   
        
        #store to csv via pandas
        if (ti % write_feq) == 0:
            data_rows = []
            data_dict = {}
            data_dict.update({'Timestep' : ti, 'time' : ti*dt, 'crosslink_breaks' : len(broken_crosslinks), 'N-breaks' : len(N_term_breaks),'backbone_breaks' : len(broken_WLC),
                          'pull force right' : av_pull_force_right_ti, 'pull force left' : av_pull_force_left_ti, 'extension': av_extension_ti})
            data_rows.append(data_dict)
            df = pd.DataFrame(data_rows)
            #data_point += 1
            df.to_csv('data.csv', mode='a',  header=not os.path.exists('data.csv'))

        timing7 += t.time() - wallclock_prev
        wallclock_prev = t.time()   

    print("Time dynamics loop: --- %s seconds ---" % (t.time() - wallclock_start))  
    logging.info("Time dynamics loop: --- %s seconds ---" % (t.time() - wallclock_start))      
    print(timing1, timing2, timing3, timing4, timing5, timing6, timing7)  
    logging.info(str(timing1), str(timing2), str(timing3), str(timing4), str(timing5), str(timing6), str(timing7))                         
    df = pd.DataFrame(data_rows)
    df.to_csv('data.csv', mode='a',  header=not os.path.exists('data.csv'))
    print (breakage_points)
    logging.info('Breakges happened: ' + str(breakage_points))
    
