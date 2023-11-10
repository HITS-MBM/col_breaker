import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib
import pandas as pd
import os
from scipy.integrate import simps
from numpy import trapz



def force_and_breaks_over_time(times, N_breaks_over_time, C_breaks_over_time,
                               backbone_breaks_over_time, av_pull_force, save_appendix = '', fontsize = 18):
    ax1 = plt.subplot(111)
    ax1.set_xlabel ('time [ns]')

    ax2 = ax1.twinx()
    ax1.set_ylabel('av. pull force [pN]')
    ax2.set_ylabel ('#breaks')
    ax2.plot(times, N_breaks_over_time, label = 'N_breaks', color='g')
    ax2.plot(times, C_breaks_over_time, label = 'C_breaks', color='b')
    ax2.plot(times, backbone_breaks_over_time, label = 'backbone breaks', color='r')
    #ax2.set_ylim(-0.5, 30)
    ax1.plot(times, av_pull_force, label = 'pull force', color='k')
    #ax1.set_ylim(-0.5, 1150)
    #get one legend for both axis
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines, labels,loc='upper right',bbox_to_anchor=(0.29,0.8), fontsize = fontsize -2,
                borderpad=1.5*legend_scale, labelspacing = legend_scale, handletextpad = legend_scale,
               handlelength=1.5, borderaxespad = 1.5*legend_scale, fancybox=True)
    ax2.legend(lines2, labels2, loc='center right', bbox_to_anchor=(0.40,0.6), fontsize = fontsize -2,
                borderpad=1.5*legend_scale, labelspacing = legend_scale, handletextpad = legend_scale,
               handlelength=1.5, borderaxespad = 1.5*legend_scale, fancybox=True)


    plt.tight_layout()
    plt.savefig('breaks_over_time'+ save_appendix + '.png', dpi = 300)
    plt.show()
    plt.close()



if __name__ == "__main__":


    equilibrium_length = 300 #[nm] use to convert back rel extension
    data_file = 'runs_goe_20230925/run_207_ext1175_k7500_Delta10_300nm_Morse142/data.csv'
    save_appendix  = '_207_ext1175_k7500_Delta10_300nm_Morse142'
    df = pd.read_csv(data_file, index_col = 1)
    cut_off = 100000 # [ns] # i.o.t. ensure same x-axis
    df = df.loc[df.time < cut_off * 1e-9]
    print(df.head())
    print(df.tail())
    times = df['time']*1e9  #scale x-axis to ns

    #print(times)
    N_breaks_over_time = df['N-breaks']
    cross_breaks_over_time = df['crosslink_breaks']
    backbone_breaks_over_time = df['backbone_breaks']
    C_breaks_over_time = cross_breaks_over_time - N_breaks_over_time
    
    av_pull_force  = 0.5*(np.abs(df['pull force right']) + np.abs(df['pull force left']))
    av_pull_force *= 1e12 #convert to pN 
    print(av_pull_force)
    av_extension = df['extension']

    #plot optics
    fontsize = 18
    font = { 'size' : fontsize } 
    matplotlib.rc('font', **font)
    legend_scale = 0.15

    flag_combined = False

    if flag_combined:
        save_appendix  = '192_Delta0_strain40_k212'
        data_file2 = 'run_192_Delta0_strain40_k212/data.csv'
        df2 = pd.read_csv(data_file2, index_col = 1)
        times2 = df2['time']*1e9  #scale x-axis to ns
        N_breaks_over_time2 = df2['N-breaks']
        cross_breaks_over_time2 = df2['crosslink_breaks']
        C_breaks_over_time2 = cross_breaks_over_time - N_breaks_over_time
        backbone_breaks_over_time2 = df2['backbone_breaks']
        av_pull_force2  = 0.5*(np.abs(df2['pull force right']) + np.abs(df2['pull force left']))
        av_pull_force2 *= 1e12 #convert to pN 
        av_extension2 = df2['extension']
        
        


    #plot breaks and pull force over time
    force_and_breaks_over_time(times, N_breaks_over_time, C_breaks_over_time, backbone_breaks_over_time,
                               av_pull_force, save_appendix)

    #plot extension and breaks over time
    #extension_and_breaks_over_time(times, N_breaks_over_time, C_breaks_over_time,
    #                           backbone_breaks_over_time, av_extension, save_appendix)


    """
    #plot force-extension
    ax1 = plt.subplot(111)
    ax1.set_xlabel ('rel. extension')
    ax1.set_ylabel('force [pN] ')
    ax1.plot(av_extension, av_pull_force)
    plt.tight_layout()
    plt.savefig('force_extension'+ save_appendix + '.png', dpi = 300)
    plt.show()
    plt.close()


    #plot force-extension until rupture
    rupture_row = df.loc[df['pull force right'].idxmax()]
    #rupture_frame = df['pull force right'].idxmax()
    backbone_frame = df[(df.backbone_breaks == 1)].first_valid_index()
    crosslink_frame = df[(df['crosslink_breaks'] > df['N-breaks'])].first_valid_index()
    if backbone_frame == None:
        rupture_frame = crosslink_frame
    elif crosslink_frame == None:
        rupture_frame = backbone_frame
    else:
        rupture_frame =  min(backbone_frame, crosslink_frame)  #one strand completely broken
    print('Max extension before rupture = ' + str(av_extension[rupture_frame]))
    print('Max force before rupture [pN] = ' + str(av_pull_force[rupture_frame]))


    # Compute the area using the composite trapezoidal rule.
    df_pos = df[df.extension > 1.0]
    rupture_frame_pos = rupture_frame

    av_pull_force_pos  = 0.5*1e12*(np.abs(df_pos['pull force right']) + np.abs(df_pos['pull force left']))
    av_extension_pos = df_pos['extension']

    area = trapz(av_pull_force_pos[:rupture_frame_pos], av_extension_pos[:rupture_frame_pos]*equilibrium_length)
    print("area (ignoring negative extensions) in [pN*nm] =", area)
    # Compute the area using the composite Simpson's rule.
    #area = simps(av_extension[:rupture_frame], av_pull_force[:rupture_frame])
    #print("Simpson's area =", area)

    ax1 = plt.subplot(111)
    ax1.set_xlabel ('rel. extension')
    ax1.set_ylabel('force [pN]')
    ax1.plot(av_extension[:rupture_frame+10000], av_pull_force[:rupture_frame_pos+10000], label = 'no path difference')
    ax1.scatter(av_extension[rupture_frame], av_pull_force[rupture_frame], s=400, facecolors='none', edgecolors='r')

    if flag_combined:
        rupture_row2 = df2.loc[df2['pull force right'].idxmax()]
        rupture_frame2 = df2[df2.nbr_breaks==2].first_valid_index() #one strand completely broken
        print('Max extension before rupture = ' + str(av_extension2[rupture_frame2]))
        print('Max force before rupture = ' + str(av_pull_force2[rupture_frame2]))
        ax1.plot(av_extension2[:rupture_frame2+10000], av_pull_force2[:rupture_frame2+10000], label = '0.96 / 1.04 path')
        ax1.scatter(av_extension2[rupture_frame2], av_pull_force2[rupture_frame2], s=400, facecolors='none', edgecolors='r')
        
        # Compute the area using the composite trapezoidal rule.
        df_pos2 = df2[df2.extension > 1.0]
        rupture_frame_pos2 = df_pos2[df_pos2.nbr_breaks==2].first_valid_index() #one strand completely broken
        av_pull_force_pos2  = 1e12*0.5*(np.abs(df_pos2['pull force right']) + np.abs(df_pos2['pull force left']))
        av_extension_pos2 = df_pos2['extension']
        
        area = trapz(av_pull_force_pos2[:rupture_frame_pos2], av_extension_pos2[:rupture_frame_pos2]*equilibrium_length)
        print("area [pN*nm] =", area)
        
    ax1.legend()
    plt.tight_layout()
    plt.savefig('force_extension'+ save_appendix + '_rupture.png', dpi = 300)
    plt.show()
    plt.close()
    """
