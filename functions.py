import os
import subprocess as sp
import matplotlib.pyplot as plt
import numpy as np
#import cv2
import glob
import networkx as nx
import sys
from numpy.random import rand
import logging
from numba import jit
#import pyvips
import time as t

def create_dir(dir_name):
    command=sp.Popen('mkdir -p ' + dir_name ,shell=True)
    command.wait()

def change_to_dir(dir_name):
    os.chdir(dir_name) 


def computeInitialCoordinates(th_per_side, x=0.0, shape='hexagon'):
    coor = []

    for i in range(th_per_side):

        if i != 0:
            for j in range(2 * (th_per_side) - 1 - i):
                a = (2 * (th_per_side) - 1 - i)
                if i % 2 == 0:
                    coor.append([x, i, 2 * (j - (a - 1) / 2), ])
                    # wlc_1.append(ctr_p)

                    coor.append([x, -i, 2 * (j - (a - 1) / 2), ])
                else:
                    coor.append([x, i, (2 * (j - (a / 2)) + 1)])
                    coor.append([x, -i, (2 * (j - (a / 2)) + 1)])
        else:
            for j in range(2 * th_per_side - 1):
                a = (2 * (th_per_side) - 1)
                coor.append([x, i, 2 * (j - (a - 1) / 2)])

    return(coor)


def defInitialNodes(coor, spacing, d_phase_arr):

    G = nx.Graph()
    NodeID = np.arange(len(coor)).tolist()

    node_to_coor = []
    pullatoms_left = []  # indices / IDs of left pull atoms

    G.add_nodes_from(NodeID)
    for i in NodeID:
        G.nodes[i]['coordinates'] = [x * spacing for x in coor[i]]
        G.nodes[i]['positions'] = coor[i]
        # set initial D-period phase
        if coor[i][1] % 2 == 0:
            G.nodes[i]['D-period'] = d_phase_arr[np.int(coor[i][2] / 2) % 5]
        else:
            G.nodes[i]['D-period'] = d_phase_arr[np.int(
                (coor[i][2] - 1) / 2 - 2) % 5]
        if not G.nodes[i]['D-period'] == 0: #in overlap region, don't pull twice on connected strand, but only on the side that would continue
            pullatoms_left.append(i)
        #elif not xx: #also pull on the on that is not connected due to boundary effect
        #    pullatoms_left.append(i)
        G.nodes[i]['CollID'] = i
        node_to_coor.extend(coor[i])
        node_to_coor.append(i)
    # some helpful graph attributes
    G.graph['node_to_coor'] = node_to_coor
    G.graph['num_coll'] = G.nodes[list(G.nodes)[-1]]['CollID'] + 1
    G.graph['max_x'] = 0.0
    return(G, pullatoms_left)


def constructMiddleLayer(G, middle_layer_atoms, overlap_length, overlap_ratio, last_layer = False):
    a = len(G.nodes())
    list_x = []
    list_c_id = []
    pullatoms_right = []

    for i in range(a):
        if G.nodes[i]['positions'][0] == G.graph['max_x']:
            list_x.append(i)

    for i, idx in enumerate(list_x):
        G.add_node(a + i)
        G.nodes[a +
                i]['coordinates'] = (G.nodes[idx]['coordinates'] +
                                     np.array([1 *
                                               overlap_length, 0, 0])).tolist()
        G.nodes[a +
                i]['positions'] = (G.nodes[idx]['positions'] +
                                   np.array([1 *
                                             overlap_ratio, 0, 0])).tolist()
        G.nodes[a + i]['D-period'] = G.nodes[idx]['D-period']
        G.nodes[a + i]['CollID'] = G.nodes[idx]['CollID']
        G.add_edge(idx, a + i, crosslink=False, weight=1.)
        list_c_id.extend(G.nodes[a + i]['positions'])
        list_c_id.append(a + i)
        middle_layer_atoms.append(a+i)


        if last_layer == True and not G.nodes[idx]['D-period'] == 4:
            pullatoms_right.append(a + i)

    G.graph['max_x'] = G.graph['max_x'] + overlap_ratio
    (G.graph['node_to_coor']).extend(list_c_id)
    return G, pullatoms_right, middle_layer_atoms


def constructNextD(G, middle_layer_atoms, spacing, overlap_length, overlap_ratio, gap_length, gap_ratio, last_layer):

    a = len(G.nodes())
    # collect all nodes in last layer and also a correct mapping of node id
    # and coor
    list_x = []
    list_c_id = []

    for i in range(a):
        if G.nodes[i]['positions'][0] == G.graph['max_x']:
            list_x.append(i)

    list_D4 = []
    aux = 0

    for i, idx in enumerate(list_x):
        if G.nodes[idx]['D-period'] == 0:
            G.add_node(a + i - aux)
            #list_D0.append(a+i- aux)
            G.nodes[a +
                    i -
                    aux]['coordinates'] = (G.nodes[idx]['coordinates'] +
                                           np.array([1 *
                                                     gap_length, 0, 0])).tolist()
            G.nodes[a +
                    i -
                    aux]['positions'] = (G.nodes[idx]['positions'] +
                                         np.array([1 *
                                                   gap_ratio, 0, 0])).tolist()
            G.nodes[a + i - aux]['D-period'] = G.nodes[idx]['D-period'] + 1
            G.nodes[a + i - aux]['CollID'] = G.nodes[idx]['CollID']
            G.add_edge(idx, a + i - aux, crosslink=False, weight=1.)
            list_c_id.extend(G.nodes[a + i - aux]['positions'])
            list_c_id.append(a + i - aux)
            #if last_layer == True:
            #    pullatoms_right.append(a + i - aux)
        if G.nodes[idx]['D-period'] == 1:
            G.add_node(a + i - aux)
            G.nodes[a +
                    i -
                    aux]['coordinates'] = (G.nodes[idx]['coordinates'] +
                                           np.array([1 *
                                                     gap_length, 0, 4 *
                                                     spacing])).tolist()
            G.nodes[a +
                    i -
                    aux]['positions'] = (G.nodes[idx]['positions'] +
                                         np.array([1 *
                                                   gap_ratio, 0, 4])).tolist()
            G.nodes[a + i - aux]['D-period'] = G.nodes[idx]['D-period'] + 1
            G.nodes[a + i - aux]['CollID'] = G.nodes[idx]['CollID']
            G.add_edge(idx, a + i - aux, crosslink=False, weight=1.)
            list_c_id.extend(G.nodes[a + i - aux]['positions'])
            list_c_id.append(a + i - aux)
            #if last_layer == True:
            #    pullatoms_right.append(a + i - aux)
        if G.nodes[idx]['D-period'] == 2:
            G.add_node(a + i - aux)
            G.nodes[a +
                    i -
                    aux]['coordinates'] = (G.nodes[idx]['coordinates'] +
                                           np.array([1 *
                                                     gap_length, -
                                                     3 *
                                                     spacing, 1 *
                                                     spacing])).tolist()
            G.nodes[a +
                    i -
                    aux]['positions'] = (G.nodes[idx]['positions'] +
                                         np.array([1 *
                                                   gap_ratio, -
                                                   3, 1])).tolist()
            G.nodes[a + i - aux]['D-period'] = G.nodes[idx]['D-period'] + 1
            G.nodes[a + i - aux]['CollID'] = G.nodes[idx]['CollID']
            G.add_edge(idx, a + i - aux, crosslink=False, weight=1.)
            list_c_id.extend(G.nodes[a + i - aux]['positions'])
            list_c_id.append(a + i - aux)
            #if last_layer == True:
            #    pullatoms_right.append(a + i - aux)
        if G.nodes[idx]['D-period'] == 3:
            G.add_node(a + i - aux)
            list_D4.append(a + i - aux)
            G.nodes[a +
                    i -
                    aux]['coordinates'] = (G.nodes[idx]['coordinates'] +
                                           np.array([1 *
                                                     gap_length, -
                                                     1 *
                                                     spacing, 1 *
                                                     spacing])).tolist()
            G.nodes[a +
                    i -
                    aux]['positions'] = (G.nodes[idx]['positions'] +
                                         np.array([1 *
                                                   gap_ratio, -
                                                   1, 1])).tolist()
            G.nodes[a + i - aux]['D-period'] = G.nodes[idx]['D-period'] + 1
            G.nodes[a + i - aux]['CollID'] = G.nodes[idx]['CollID']
            G.add_edge(idx, a + i - aux, crosslink=False, weight=1.)
            list_c_id.extend(G.nodes[a + i - aux]['positions'])
            list_c_id.append(a + i - aux)
            #if last_layer == True:
            #    pullatoms_right.append(a + i - aux)
        if G.nodes[idx]['D-period'] == 4:
            aux = aux + 1

    lastnode = list_c_id[3::4][-1]
    # print('lastnode: ',lastnode)
    for i, nodeid4 in enumerate(list_D4):
        #print(lastnode +1 + i)
        G.add_node(lastnode + 1 + i)
        G.nodes[lastnode +
                1 +
                i]['coordinates'] = (G.nodes[nodeid4]['coordinates'] +
                                     np.array([0, 1 *
                                               spacing, 1 *
                                               spacing])).tolist()
        G.nodes[lastnode +
                1 +
                i]['positions'] = (G.nodes[nodeid4]['positions'] +
                                   np.array([0, 1, 1])).tolist()
        G.nodes[lastnode + 1 + i]['D-period'] = 0
        G.nodes[lastnode + 1 + i]['CollID'] = G.graph['num_coll']
        G.graph['num_coll'] = G.nodes[lastnode + 1 + i]['CollID'] + 1
        list_c_id.extend(G.nodes[lastnode + 1 + i]['positions'])
        list_c_id.append(lastnode + 1 + i)
        #if last_layer == True:
        #    pullatoms_right.append(a + i - aux)
    G.graph['max_x'] = G.graph['max_x'] + gap_ratio
    (G.graph['node_to_coor']).extend(list_c_id)

    G, pullatoms_right, middle_layer_atoms = constructMiddleLayer(G, middle_layer_atoms, overlap_length, overlap_ratio, last_layer)
    return G, pullatoms_right, middle_layer_atoms


def generateCrosslinks(G, pullatoms_left, middle_layer_atoms, connectedness=1.0, low_up_ratio=1.0,
                       allow_switches = False, side = 'both', double = 'false'):

    node_to_coor = G.graph['node_to_coor']
    node_to_coor = np.asarray(node_to_coor).reshape(
        int(len(node_to_coor) / 4), 4)
    nodeid = node_to_coor[:, 3]
    crd = node_to_coor[:, :3]
    cross = []

    a = len(G.nodes)
    switch = False

    for i in range(a):
        if G.nodes[i]['D-period'] == 0:
            if rand(1) >= connectedness:
                continue
           
            # check if there is an existing direct neighbouring node
            # in the lower left (seen in Agnieszkas model) and if yes add an edge

            crd_i = crd[np.tile(nodeid == i, (3, 1)).transpose()]

            crd_nextD4 = crd_i + [0., -1., -1.]
            crd_nextD4_ = crd_i + [0., 1., -1.]
            if rand(1) <= low_up_ratio:
                switch = False
            else:
                switch = True

            if double == False and switch == False:
                if np.any(np.sum(crd == crd_nextD4, axis=1) ==3):
                    # check if all 3 coords are equal
                    crosslink = [int(nodeid[np.sum(crd == crd_nextD4, axis=1) == 3][0]), int(i)]
                elif allow_switches == True and np.any(np.sum(crd == crd_nextD4_, axis=1) == 3):
                    crosslink = [int(nodeid[np.sum(crd == crd_nextD4_, axis=1) == 3][0]), int(i)]
                    if crd_i[0] ==0: #when switched, there are now two strands connected to one so the pull atoms needs to be added again to balance the force
                        pullatoms_left.append(crosslink[0])
                        print ('added extra pull atom due to crosslink switch: ' + str(crosslink[0]))
                elif crd_i[0] ==0 : #in first layer, add non connected strand to pull atoms
                    pullatoms_left.append(i)
                    print ('added missing pull atom: ' + str(i))
            elif double == False and switch == True:
                if np.any(np.sum(crd == crd_nextD4_, axis=1) == 3):
                    crosslink = [int(nodeid[np.sum(crd == crd_nextD4_, axis=1) == 3][0]), int(i)]
                elif allow_switches == True and np.any(np.sum(crd == crd_nextD4, axis=1) ==3):
                    crosslink = [int(nodeid[np.sum(crd == crd_nextD4, axis=1) == 3][0]), int(i)]
                    if crd_i[0] ==0: #when switched, there are now two strands connected to one so the pull atoms needs to be added again to balance the force
                        pullatoms_left.append(crosslink[0])
                        print ('added extra pull atom due to crosslink switch: ' + str(crosslink[0]))
                #commented out for now: do not do this if switch is the reason?
                #elif crd_i[0] ==0 : #in first layer, add non connected strand to pull atoms
                    #pullatoms_left.append(i)
                    #print ('added missing pull atom: ' + str(i))
            elif double == True:  #do both directions
                if np.any(np.sum(crd == crd_nextD4, axis=1) ==3):
                    crosslink = [int(nodeid[np.sum(crd == crd_nextD4, axis=1) == 3][0]), int(i)]
                elif crd_i[0] ==0 : #in first layer, add non connected strand to pull atoms
                    pullatoms_left.append(i)
                    print ('added missing pull atom: ' + str(i))
                if np.any(np.sum(crd == crd_nextD4_, axis=1) == 3):
                    crosslink2 = [int(nodeid[np.sum(crd == crd_nextD4_, axis=1) == 3][0]), int(i)] 
                    if side == 'both':  
                        cross.append(crosslink2)
                        G.add_edge(crosslink2[0], crosslink2[1], crosslink=True)
                    elif side == 'N' and crosslink2[0] not in middle_layer_atoms and crosslink2[1] not in middle_layer_atoms:
                        cross.append(crosslink2)
                        G.add_edge(crosslink2[0], crosslink2[1], crosslink=True)
                    elif side == 'C' and crosslink2[0] in middle_layer_atoms and crosslink2[1] in middle_layer_atoms:
                        cross.append(crosslink2)
                        G.add_edge(crosslink2[0], crosslink2[1], crosslink=True) 


            if side == 'both':
                if crosslink in cross:
                    print ('Warning: Duplicate detected, will not add again: ' + str(crosslink))  
                    logging.warning('Duplicate detected, will not add again: ' + str(crosslink))
                    continue
                cross.append(crosslink)
                G.add_edge(crosslink[0], crosslink[1], crosslink=True)
            elif side == 'N' and crosslink[0] not in middle_layer_atoms and crosslink[1] not in middle_layer_atoms:
                if crosslink in cross:
                    print ('Warning: Duplicate detected, will not add again: ' + str(crosslink))
                    logging.warning('Duplicate detected, will not add again: ' + str(crosslink))  
                    continue
                cross.append(crosslink)
                G.add_edge(crosslink[0], crosslink[1], crosslink=True)
            elif side == 'C' and crosslink[0] in middle_layer_atoms and crosslink[1] in middle_layer_atoms:
                if crosslink in cross:
                    print ('Warning: Duplicate detected, will not add again: ' + str(crosslink))  
                    logging.warning('Duplicate detected, will not add again: ' + str(crosslink))
                    continue
                cross.append(crosslink)
                G.add_edge(crosslink[0], crosslink[1], crosslink=True) 
            #else:
            #    print('warning: wrong crosslink side parameter')
            #    logging.warning('wrong crosslink side parameter')
                
    print('Info: Number of crosslinks in this system: ' + str(len(cross)))
    logging.info('Number of crosslinks in this system: ' +
                 str(len(cross)))
    print('Crosslinks in : ' + str(cross))
    logging.info('Crosslinks in : ' + str(cross))

    return G, cross, pullatoms_left

def find_crosslink_position(middle_layer_atoms, cross):
    N_cross = []
    C_cross = []

    for pair in cross:
        if pair[0] in middle_layer_atoms and pair[1] in middle_layer_atoms:
            C_cross.append(pair)
        elif pair[0] not in middle_layer_atoms and pair[1] not in middle_layer_atoms:
            N_cross.append(pair)
        else:
            print('error: both crosslink atoms should be on same layer')
            logging.error('both crosslink atoms should be on same layer')
            
    return N_cross, C_cross

def delete_unconnected_pullatom(G, cross, pullatoms_left):

    cross_all_flattened = [j for pair in cross for j in pair]

    
    for atom in pullatoms_left:
        if (G.nodes[atom]['D-period'] == 4) and not (atom in cross_all_flattened): #look for ending strand that is not connected
            for molecule in getListCollICtoNodeID(G):   #also check if C-terminal atom is not connected (needed if C-crosslinks only)
                if atom in molecule and not (molecule[1] in cross_all_flattened):
                    print('Remove extra not connected pull atom: ' + str(atom))
                    pullatoms_left.remove(atom)
                         
    return pullatoms_left


def getListCollICtoNodeID(G):
    numcoll = G.graph['num_coll']
    collnodes = []

    # print(aux[1].append(1))
    for k in range(numcoll):
        aux = []
        for i in G.nodes():
            if G.nodes[i]['CollID'] == k:
                aux.append(i)
        collnodes.append(aux)
    return(collnodes)


def get_sliced_coords(G, x=0):
    NodeID = np.array(G.nodes).tolist()
    coords = []
    dph = []
    for i in NodeID:
        if G.nodes[i]['positions'][0] == (x):
            coords.append(np.asarray(G.nodes[i]['coordinates']))
            dph.append(G.nodes[i]['D-period'])

    #coords = np.concatenate(coords).reshape(len(coords),3).astype('float')
    dph = np.asarray(dph)

    return coords, dph


def get_coords(G):
    NodeID = np.array(G.nodes).tolist()
    coords = []
    dph = []
    for i in NodeID:
        coords.append(np.asarray(G.nodes[i]['coordinates']))
        dph.append(G.nodes[i]['D-period'])

    #coords = np.concatenate(coords).reshape(len(coords),3).astype('float')
    dph = np.asarray(dph)

    return coords, dph


def get_WLCs_and_contourlengths(G, coords, cross, contour_factor, N_path_difference_factor, C_path_difference_factor):

    wlc_1 = []  # array of wlc links one side # #array-size to be: nr_wlc
    wlc_2 = []  # other side of wlc links
    contour_lengths = []
    N_path = []
    C_path = []
    cross_all_flattened = [j for pair in cross for j in pair]
    colnodes = getListCollICtoNodeID(G)
    #print(colnodes)
    #print(G.edges)
    for molecule in colnodes:
        if len(molecule) <= 1:  # no connections if only one point in Triple Helix
            continue
        for i in range(len(molecule) - 1):
            start = coords[molecule[i]]
            wlc_1.append(molecule[i])
            end = coords[molecule[i + 1]]
            wlc_2.append(molecule[i + 1])

            L0 = contour_factor * (end[0] - start[0])
            if int(molecule[i]) in cross_all_flattened:
                #pass
                if G.nodes[molecule[i]]['D-period'] == 0: #N-terminal path
                    L0 = L0*N_path_difference_factor
                    N_path.append(molecule[i])   #to do: This is just the upper side not N-term crosslink
                elif  G.nodes[molecule[i]]['D-period'] == 4: #C-term path
                    C_path.append(molecule[i])
                    L0 = L0*C_path_difference_factor
                #print ("changeed contourlength due to shortest path hypothesis")
                
            contour_lengths.append(L0)

    return wlc_1, wlc_2, contour_lengths



#@jit(nopython=True) #not faster this way with jit. tbd if split up 
def calculate_constant_pull_forces(f, pull_forces_left, pull_forces_right, pullatoms_left, pullatoms_right, constant_force):
  
    #left    
    for j in range(0, len(pullatoms_left)):
        f[pullatoms_left[j]] = f[pullatoms_left[j]] - constant_force
        pull_forces_left[j] = -constant_force

    av_pull_force_left = -constant_force #no need to average if anyways constant

    # right
    for j in range(0, len(pullatoms_right)):     
        f[pullatoms_right[j]] = f[pullatoms_right[j]] + constant_force
        pull_forces_right[j] = constant_force

    av_pull_force_right = constant_force  #no need to average if anyways constant

    return f, av_pull_force_left, av_pull_force_right 

#@jit(nopython=True) #not faster this way with jit. tbd if split up 
def calculate_vel_pull_forces(f, coords, pull_forces_left, pull_forces_right, pullatoms_left, pullatoms_right, time, 
x0_pull_left, x0_pull_right, v_pull, k0):

    # pull on both ends with half speed
    x_pull_left = x0_pull_left - v_pull * (time / 2.0)
    x_pull_right = x0_pull_right + v_pull * (time / 2.0)

    #left    
    for j in range(0, len(pullatoms_left)):
        x_pos = coords[pullatoms_left[j]][0]
        dx = x_pos - x_pull_left
        df =  k0 * dx
        f[pullatoms_left[j]] = f[pullatoms_left[j]] - df
        pull_forces_left[j] = - df
    av_pull_force_left = np.mean(pull_forces_left)
    
    # right
    for j in range(0, len(pullatoms_right)):     
        x_pos = coords[pullatoms_right[j]][0]
        dx = x_pos - x_pull_right
        df =  k0 * dx
        f[pullatoms_right[j]] = f[pullatoms_right[j]] - df
        pull_forces_right[j] = - df

    av_pull_force_right = np.mean(pull_forces_right)

    return f, av_pull_force_left, av_pull_force_right 

#@jit(nopython=True) #not faster this way with jit. tbd if split up 
def calculate_strain_pull_forces(f, coords, pull_forces_left, pull_forces_right, pullatoms_left, pullatoms_right, time, 
x0_pull_left, x0_pull_right, v_pull, k0, max_extension):

    pull_forces_left = np.empty(len(pullatoms_left))
    pull_forces_right = np.empty(len(pullatoms_right))
    #extend virtual spring up to a constant strain
    x_pull_left = max(x0_pull_left - max_extension, x0_pull_left - v_pull * (time / 2.0))
    x_pull_right = min(x0_pull_right + max_extension, x0_pull_right + v_pull * (time / 2.0))

    #left    
    for j in range(0, len(pullatoms_left)):
        x_pos = coords[pullatoms_left[j]][0]
        dx = x_pos - x_pull_left
        df =  k0 * dx
        f[pullatoms_left[j]] = f[pullatoms_left[j]] - df
        pull_forces_left[j] = - df

    av_pull_force_left = np.mean(pull_forces_left)
    
    # right
    for j in range(0, len(pullatoms_right)):     
        x_pos = coords[pullatoms_right[j]][0]
        dx = x_pos - x_pull_right
        df =  k0 * dx
        f[pullatoms_right[j]] = f[pullatoms_right[j]] - df
        pull_forces_right[j] = - df
    av_pull_force_right = np.mean(pull_forces_right)

    return f, av_pull_force_left, av_pull_force_right 

@jit(nopython=True)
def calculate_crosslink_df_gaussian(dx, sig_cross, v_cross):
    ex1 = dx**2 / (2.0 * sig_cross**2)
    return  v_cross * dx * np.exp(-ex1) / (sig_cross**2)   

@jit(nopython=True)
def calculate_crosslink_df_morse_fb(dx, r_fb, Edis, beta):
    df = 2*beta*Edis*np.exp(-beta*(dx-r_fb)) * \
        (1-np.exp(-beta*(dx-r_fb)))
    
    return  df



def calculate_crosslink_forces(f, coords, cross, r_fb, Edis, beta, time, broken_crosslinks, breakage_points, N_cross, N_term_breaks):
    for crosslink in cross:
        if crosslink in broken_crosslinks: #no rebinding, even if closer again, so go on.
            continue

        if coords[crosslink[1]][0] > coords[crosslink[0]][0]:
            i1 = crosslink[0]
            i2 = crosslink[1]
        else:
            i1 = crosslink[1]
            i2 = crosslink[0]

        dx = coords[i2][0] - coords[i1][0]
        if dx < 0:
            print('Wrong order of crosslinks')
        elif dx < r_fb: #cut off flat bottom
            continue
        elif dx > (r_fb + (0.25* 10**-9)): #bond breaks the latest at 0.25nm after flat bottom
            df = calculate_crosslink_df_morse_fb(dx, r_fb, Edis, beta)
            print ("crosslink broken: " + str(crosslink)+ ' with force: ' +str(df) + ' at time: ' + str(time))
            logging.info("crosslink broken: " + str(crosslink)+ ' with force: ' +str(df) + ' at time: ' + str(time))
            broken_crosslinks.append(crosslink)
            breakage_points.append((time, crosslink))
            if crosslink in N_cross:
                N_term_breaks.append(crosslink)
            df = 0 #set to zero
        else:
            df = calculate_crosslink_df_morse_fb(dx, r_fb, Edis, beta)
        f[i1] = f[i1] + df
        f[i2] = f[i2] - df
        #tension_cross[k] = np.abs(df)

    return f, broken_crosslinks, breakage_points, N_term_breaks

@jit(nopython=True)
def calculate_WLC_force(dx, l0, kT, pl):
    dxl = 1.0 - dx/l0
    #dxv = 4.0 * dxl
    dxf = 4.0 * dxl**2
    #v = v + ((kT / pl) * (l0 / dxl + dx**2 / (2.0 * l0) - dx / 4.0))
    df = (kT / pl) * (1.0 / dxf + dx/l0 - 1.0 / 4.0)

    return df



@jit(nopython=True)
def calculate_pwWLC_force(dx, l0, cf, kT, pl, K0):
    x_rel = dx/(l0/cf)
    F_trans = 1/4*(kT*K0**2/pl)**(1/3)
    x_trans = cf*(1-0.5*(kT/(F_trans*pl))**0.5)
    if x_rel <= x_trans:
        df = kT/(4*pl)*(1-x_rel/cf)**(-2)
    else:
        df = K0*(x_rel - x_trans)+ F_trans
    return df

@jit(nopython=True)
def calculate_pwWLC_force_Morse(dx, l0, cf, kT, pl, K0, beta, Edis):
    broken = False
    x_rel = dx/(l0/cf)
    F_trans = 1/4*(kT*K0**2/pl)**(1/3)
    x_trans = cf*(1-0.5*(kT/(F_trans*pl))**0.5)

    F_morse = 4.0*10**(-9)

    x_morse_rel = (F_morse - F_trans) / K0 + x_trans  #switch to Morse potential when WLC reaches force F_morse 
    x_morse = x_morse_rel *(l0/cf) 
    x_morse_shift = 1.39 *10**(-11)  #shifting Morse potential such that F about is continous, i,e, F(x_morse) = F_morse from both sites
    
    if x_rel <= x_trans:
        df = kT/(4*pl)*(1-x_rel/cf)**(-2)
    elif ((x_trans < x_rel) and (x_rel <= x_morse_rel)):
        df = K0*(x_rel - x_trans)+ F_trans
        #print(df)
    elif (x_rel > x_morse_rel) and (dx <= (x_morse + 0.25*10**-9)):
            df =1.45*( 2*beta*Edis*np.exp(-beta*(dx + x_morse_shift-x_morse)) * \
        (1-np.exp(-beta*(dx + x_morse_shift-x_morse))))
    else: #breaks after 0.25nm into last Morse bit 
         print('broken')
         broken = True

    return df, broken



@jit(nopython=True)
def get_smoluchowski_dx(D, dt, f, kT):
    u = np.random.normal()  # random number form gaussian distribution
    dx = np.sqrt(2.0 * D * dt) * u + D * f* dt / kT  

    return dx


def plot_points_phase_colored(coords, dph, colors):

    fontsize = 16
    font = { 'size' : fontsize } 
    plt.rc('font', **font)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')


    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    for i in range(len(coords)):
        point = coords[i]
        color = colors[dph[i]]
        ax.scatter(point[0], point[1], point[2], color=color)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

    plt.show()
    plt.close()
      


def plot_starting_configuration(coords, dph,pullatoms_left, pullatoms_right, cross, G, colors):
    # function that plots the connections / WLCs between the different points
    # of a tropocollagen
    fontsize = 16
    font = { 'size' : fontsize } 
    plt.rc('font', **font)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    #ax.tick_params(top=False, bottom=False, left=False, right=False)



    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    for i in range(len(coords)):
        point = coords[i]
        color = colors[dph[i]]
        ax.scatter(point[0], point[1], point[2], color=color)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

    colnodes = getListCollICtoNodeID(G)
    for molecule in colnodes:
        if len(molecule) <= 1:  # no connections if only one point
            continue
        for i in range(len(molecule) - 1):
            Start = coords[molecule[i]]
            End = coords[molecule[i + 1]]
            x_values = [Start[0], End[0]]
            y_values = [Start[1], End[1]]
            z_values = [Start[2], End[2]]
            # linewidth = 0.5 + tension_wlc[i]*tension_factor #adjust linewidth
            # by tension
            ax.plot3D(x_values, y_values, z_values, color='0.3')

    # plot crosslinks
    for crosslink in cross:
        Start = coords[crosslink[0]]
        End = coords[crosslink[1]]
        x_values = [Start[0], End[0]]
        y_values = [Start[1], End[1]]
        z_values = [Start[2], End[2]]
        ax.plot3D(x_values, y_values, z_values, color='r', linewidth = 4)
        #ax.scatter(Start[0],Start[1], Start[2], color = 'r')
        #ax.scatter(End[0],End[1],End[2], color = 'm')

    '''        
    #plot pull atoms in different color
    for i in pullatoms_left:
        point = coords[i]
        ax.scatter(point[0], point[1], point[2], color='w', alpha = 0.8, s=0.8)
        #if i == 14:
        #    ax.scatter(point[0], point[1], point[2], color='w', alpha = 0.8, s=2.5)

    for i in pullatoms_right:
        point = coords[i]
        ax.scatter(point[0], point[1], point[2], color='w', alpha = 0.8, s=0.9)   
    
    '''
    # select plot_angle
    angle = 265
    ax.view_init(50, angle)

    plt.savefig('starting_configuration_compare3.png')
    plt.show()
    plt.close()



def plot_starting_configuration_black(coords, dph,pullatoms_left, pullatoms_right, cross, G, colors):
    # function that plots the connections / WLCs between the different points
    # of a tropocollagen
    fontsize = 16
    font = { 'size' : fontsize } 
    plt.rc('font', **font)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    #remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.tick_params(top=False, bottom=False, left=False, right=False)



    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    #remove grid
    ax.grid(False)


 
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    for i in range(len(coords)):
        point = coords[i]
        color = 'dimgrey' #colors[dph[i]]
        ax.scatter(point[0], point[1], point[2], color=color, s = 6)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

    colnodes = getListCollICtoNodeID(G)
    for molecule in colnodes:
        if len(molecule) <= 1:  # no connections if only one point
            continue
        for i in range(len(molecule) - 1):
            Start = coords[molecule[i]]
            End = coords[molecule[i + 1]]
            x_values = [Start[0], End[0]]
            y_values = [Start[1], End[1]]
            z_values = [Start[2], End[2]]
            # linewidth = 0.5 + tension_wlc[i]*tension_factor #adjust linewidth
            # by tension
            ax.plot3D(x_values, y_values, z_values, color='0.3')

    # plot crosslinks
    for crosslink in cross:
        Start = coords[crosslink[0]]
        End = coords[crosslink[1]]
        x_values = [Start[0], End[0]]
        y_values = [Start[1], End[1]]
        z_values = [Start[2], End[2]]
        ax.plot3D(x_values, y_values, z_values, color='r', linewidth = 3)
        #ax.scatter(Start[0],Start[1], Start[2], color = 'r')
        #ax.scatter(End[0],End[1],End[2], color = 'm')

    '''        
    #plot pull atoms in different color
    for i in pullatoms_left:
        point = coords[i]
        ax.scatter(point[0], point[1], point[2], color='w', alpha = 0.8, s=0.8)
        #if i == 14:
        #    ax.scatter(point[0], point[1], point[2], color='w', alpha = 0.8, s=2.5)

    for i in pullatoms_right:
        point = coords[i]
        ax.scatter(point[0], point[1], point[2], color='w', alpha = 0.8, s=0.9)   
    
    '''
    # select plot_angle
    angle = 265
    ax.view_init(50, angle)

    plt.savefig('starting_configuration_compare3.png')
    plt.show()
    plt.close()


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def plot_frame(coords, dph, pullatoms_left, pullatoms_right, wlc_1, wlc_2,  broken_WLC, cross, broken_crosslinks, ti, nr_steps, fibril_length, colors, ctr_angle, reverse):

    frames_length = len(str(nr_steps))
    frame_nbr = "{0:0={counter_length}d}".format(
        ti, counter_length=frames_length)
    dpi = 300
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    ax.set_xlim(0.3 * -fibril_length, 1.3 * fibril_length)
    ax.set_ylim(0.3 * -fibril_length, 1.3 * fibril_length)
    ax.set_zlim(0.01 * -fibril_length, 0.03 * fibril_length)

    # plot points
    """
    for i in range(len(coords)):
        point = coords[i]
        color = colors[dph[i]]
        ax.scatter(point[0], point[1], point[2], color=color, s=1)
    """
    xs = [coord[0] for coord in coords]
    ys =[coord[1] for coord in coords]
    zs = [coord[2] for coord in coords]
    colors_all = [ colors[dph[i]] for i in range(len(coords))]

    ax.scatter(xs, ys, zs, color=colors_all, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    # plot WLCs
    for i in range(len(wlc_1)):
        Start = coords[wlc_1[i]]
        End = coords[wlc_2[i]]
        x_values = [Start[0], End[0]]
        y_values = [Start[1], End[1]]
        z_values = [Start[2], End[2]]
        # linewidth = 0.5 + tension_wlc[i]*tension_factor
        if i in broken_WLC:
            ax.plot3D(x_values, y_values, z_values, color='0.3', linestyle='dashed', linewidth=0.4)
        else:
            ax.plot3D(x_values, y_values, z_values, color='0.3', linewidth=0.4)

    # plot crosslinks
    for crosslink in cross:
        Start = coords[crosslink[0]]
        End = coords[crosslink[1]]

        x_values = [Start[0], End[0]]
        y_values = [Start[1], End[1]]
        z_values = [Start[2], End[2]]
        # linewidth = 0.5 + tension_wlc[i]*tension_factor 
        if crosslink in broken_crosslinks:
            ax. plot(x_values, y_values, z_values,
                color='r', linestyle='dashed', linewidth=0.4)
        else:
            ax.plot3D(x_values, y_values, z_values, color='r', linewidth=0.8)

        
    #plot pull atoms in different color
    for i in pullatoms_left:
        point = coords[i]
        ax.scatter(point[0], point[1], point[2], color='m', alpha = 0.5, s=0.3)

    for i in pullatoms_right:
        point = coords[i]
        ax.scatter(point[0], point[1], point[2], color='m', alpha = 0.5, s=0.3)   
    
    

    # select plot_angle
    # rotate angle a bit back and forth
    angle1 = 30
    angle2 = 260
    angle1 = angle1 + (ctr_angle * 0.08)
    angle2 = angle2 + (ctr_angle * 0.08)
    if ctr_angle >= 150:
        reverse = True
    if ctr_angle <= -150:
        reverse = False

    if reverse == True:
        ctr_angle += -1
    elif reverse == False:
        ctr_angle += 1

    ax.view_init(angle1, angle2)
    set_axes_equal(ax)
    # ax.set_box_aspect([1,1,1])

    plt.title(
        'Crosslink ruptures: ' + str(len(broken_crosslinks)) +
        '. Backbone ruptures: ' + str(len(broken_WLC)))
    plt.tight_layout()

    plt.savefig("frame_" + str(frame_nbr) + ".svg")
    #plt.show()
    plt.close()

    return ctr_angle, reverse


def make_video():
    img_array = []
    for filename in glob.glob('./*.svg'):
        newfile = filename + '.png'
        image = pyvips.Image.new_from_file(filename,dpi=300,scale=1)
        image.write_to_file("newfile")
        img = cv2.imread(newfile)
        #height, width, layers = img.shape
        #size = (1920, 1080)
        img_array.append(img)

    out = cv2.VideoWriter(
        'collagen_video3.mp4',
        cv2.VideoWriter_fourcc(
            *'mjpg'),
        12)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



if __name__ == "__main__":
    make_video()
    pass
