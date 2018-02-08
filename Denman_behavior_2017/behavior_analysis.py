import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter1d
import scipy.optimize as opt
from scipy.signal import resample
from scipy.interpolate import interp1d
from skimage import color
import os, copy, csv
import cPickle as pkl
from djd.generalephys import cleanAxes, smoothRF
import djd.jeti as jeti

from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap, cm

from skimage.color import rgb2xyz

import imaging_behavior.core.utilities as ut
import imaging_behavior.plotting.plotting_functions as pf
import imaging_behavior.plotting.utilities as pu

from matplotlib.colors import ListedColormap
cmapUV = plt.cm.Purples_r
cmapG = plt.cm.Blues
mouse_cmap = cmapUV(np.arange(cmapUV.N))
mouse_cmap[:80,:]=cmapUV(np.arange(cmapUV.N))[np.linspace(0,255,80).astype(int),:]
mouse_cmap[176:,:]=cmapG(np.arange(cmapUV.N))[np.linspace(0,255,80).astype(int),:]
mouse_cmap[80:177,-1]=0.0 #<---- needs to be adjusted by space
mouse_cmap = ListedColormap(mouse_cmap)

deltas = np.sort([-2.0,-1.8,-1.4,-1.3,-1.2,-1.1,
      -1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,
      2. ,  1.8,  1.4,  1.3,  1.2,  1.1,  1. ,  0.9,  0.8,  0.7,  0.6,
        0.5,  0.4,  0.3,  0.2,  0.1 ])
contrasts = deltas/2.

#TODO:
#   clean up imports
#   add docstrings

#for parsing behavioral session data; most written by doug ollerenshaw
def calculate_latency(df_in):
    # For each trial, calculate the difference between each stimulus change and the next lick (response lick)
    df_in = df_in.copy()
    for idx in df_in.index:
        if pd.isnull(df_in['response_latency'][idx]) and pd.isnull(df_in['change_time'][idx])==False and len(df_in['lick_times'][idx])>0:
            licks = np.array(df_in['lick_times'][idx])
            post_stimulus_licks = licks[licks>=0]

            df_in.loc[idx,'response_latency'] = post_stimulus_licks[0]-df_in['change_time'][idx]

    return df_in
def get_lickframes(df_in,t,lickframes,before=2,after=5):
    # Determine which frames the animal licked on
    lickframes = []
    for ii,idx in enumerate(df_in.index):
        

        change_frame =  df_in['change_frame'][idx]
        session_index = df_in['session_index'][idx]

        lickframes_local = lickframes[session_index][np.logical_and(lickframes[session_index]>change_frame-before*60,
                                                     lickframes[session_index]<change_frame+after*60)]

        lickframes.append(lickframes_local)
    return lickframes
def get_licktimes(df_in,t,lickframes,before=2,after=5):
    # turn lick frames into lick times
    licktimes = []
    for ii,idx in enumerate(df_in.index):
        

        change_frame =  df_in['change_frame'][idx]
        session_index = df_in['session_index'][idx]
        change_time = t[session_index][int(change_frame)]
        lickframes_local = lickframes[session_index][np.logical_and(lickframes[session_index]>change_frame-before*60,
                                                     lickframes[session_index]<change_frame+after*60)]
        licktimes_local = t[session_index][lickframes_local.astype(int)] - change_time

        licktimes.append(licktimes_local)
    return licktimes
def plot_licks(df_in,t,ax,showY=True,before=2,after=5,splitLicksOnReward=True,plotPostRewardLicks=True):
    # make a plot of lick rasters
    blue = [102/255.,153/255.,204/255.]
    
    df_temp = df_in.copy()
    
    correct,total = get_correct(df_temp)
    
    #lickframes=get_lickframes(df_in,t,before=before,after=after)
    licktimes = get_licktimes(df_temp,t,lickframes,before=before,after=after)
    
    pf.plotLickRaster(licktimes,ax=ax[0],splitLicksOnReward=splitLicksOnReward,plotPostRewardLicks=plotPostRewardLicks,
                      showXLabel=False,showYLabel=showY,
                      postlickcolor=blue,fontsize=12,
                      leftlim=-1*before,rightlim=after)
    pf.plotLickPSTH(licktimes,ax=ax[1],binwidth=0.1,showX=True,showYLabel=showY,
                splitLicksOnReward=splitLicksOnReward,plotPostRewardLicks=plotPostRewardLicks,
                postlickcolor=blue,fontsize=12,
                leftlim=-1*before,rightlim=after,ymax=0.035)
    
    ax[0].set_xlim([-before,after])
    ax[1].set_xlim([-before,after])
    
    return licktimes,correct,total
def plot_licks2(df_in,ax,showY=True,before=2,after=5,splitLicksOnReward=True,plotPostRewardLicks=True,includeprevious=False):
    # make a plot of lick rasters
    blue = [102/255.,153/255.,204/255.]
    
    df_temp = df_in.copy()
    
    #correct,total = get_correct(df_temp)
    
    #lickframes=get_lickframes(df_in,t,before=before,after=after)
    #licktimes = get_licktimes(df_temp,t,lickframes,before=before,after=after)
    licktimes = [np.array(licktimes) - np.array(df_in.change_time)[i] for i,licktimes in enumerate(df_in.lick_times)]
    if includeprevious:
        licktimes = [np.array(licktimes) - np.array(df_in.change_time)[i] for i,licktimes in enumerate(df_in.lick_times)]
        licktimes_previous = [np.array(df_in.lick_times[index-1])+df_in.change_time[index-1] - df_in.change_time[index] for i,index in enumerate(df_in.index[1:])] 
        licktimes = [licktimes_trial.tolist() for licktimes_trial in licktimes]
        licktimes_previous = [licktimes_trial.tolist() for licktimes_trial in licktimes_previous]
        b=[[0]]
        b.extend(licktimes_previous)
        for i in range(np.shape(licktimes)[0]):
            dump = [licktimes[i].extend(b[i]) for i in range(np.shape(licktimes)[0])]
    else:
        licktimes = [np.array(licktimes) - np.array(df_in.change_time)[i] for i,licktimes in enumerate(df_in.lick_times)]

    pf.plotLickRaster(licktimes,ax=ax[0],splitLicksOnReward=splitLicksOnReward,plotPostRewardLicks=plotPostRewardLicks,
                      showXLabel=False,showYLabel=showY,
                      postlickcolor=blue,fontsize=12,
                      leftlim=-1*before,rightlim=after,prelickcolor=[0.2,0.2,0.2])
    pf.plotLickPSTH(licktimes,ax=ax[1],binwidth=0.1,showX=True,showYLabel=showY,
                splitLicksOnReward=splitLicksOnReward,plotPostRewardLicks=plotPostRewardLicks,
                postlickcolor=blue,fontsize=12,
                leftlim=-1*before,rightlim=after,ymax=0.035,prelickcolor=[0.2,0.2,0.2])
    
    ax[0].set_xlim([-before,after])
    ax[1].set_xlim([-before,after])
    
    return licktimes#,correct,total
def get_correct(df_in,window=0.5):
    # calculate the response probability
    df_temp = df_in.copy()
    
    correct=len(df_temp[df_temp.response_latency<window])
    total=len(df_temp)
    
    return correct,total
def get_pkls(folder):
    # find all PKL files in a given directory
    filenames = []
    for f in os.listdir(folder):
        if f.endswith('.pkl'):
            filenames.append(f)
    return filenames
def plot_response_matrix_pooled(df_in):
    # make a matrix of response probabilities for the various stimulus configurations
    matr = np.zeros((2,2))*np.nan
    for ii,initial_image in enumerate(np.sort(df_in.initial_image.unique())):
        for jj,change_image in enumerate(np.sort(df_in.change_image.unique())):
            df_temp = df_in[(df_in.change_image==change_image)&(df_in.initial_image==initial_image)]

            correct=len(df_temp[df_temp.response_latency<1])
            total=len(df_temp)
            print ii,jj,initial_image,change_image,correct,total,float(correct)/total
            matr[ii,jj] = float(correct)/total
    print ""
    print matr
    fig,ax=plt.subplots(figsize=(5,5))
    pf.show_gray(matr,ax=ax,cmin=0,cmax=1,colorbar=True,cmap='magma',colorbarlabel='Response Probability')
    ax.set_xticks(range(len(df_in.initial_image.unique())))
    ax.set_yticks(range(len(df_in.initial_image.unique())))
    ax.set_xticklabels(np.sort(df_in.initial_image.unique()))
    ax.set_yticklabels(np.sort(df_in.initial_image.unique()))
    ax.set_ylabel('Image Before Change',fontsize=14)
    ax.set_xlabel('Image After Change',fontsize=14)
    ax.grid(False)
    return fig,ax
def calculate_reward_rate(df,window=1.0):
    #add a column called reward_rate to the input dataframe
    #the reward_rate column contains a rolling average of rewards/min
    reward_rate = np.zeros(np.shape(df.change_time));print np.shape(reward_rate)
    c=0
    for session_index in df.session_index.unique():                      # go through the dataframe by each behavior session
        df_temp = df[df.session_index==session_index]                    # get a dataframe for just this session
        trial_number = 0
        for trial in range(np.shape(df_temp)[0]):
            if trial_number <10 :                                        # if in first 10 trials of experiment
                reward_rate_on_this_lap = np.inf                         # make the reward rate infinite, so that you include the first trials automatically.
            else:                                                        # get the correct response rate around the trial
                if trial_number > np.shape(df_temp)[0] - 6:              # if in the last 5 trials, use all the remaining trials.
                    df_roll = df_temp[trial-5:]
                else:                                                    
                    df_roll = df_temp[trial-5:trial+5]                           # use a 10 trial window around the current trial
                correct = len(df_roll[df_roll.response_latency<window])    # get a rolling number of correct trials
                time_elapsed = np.array(df_roll.change_time)[-1] - np.array(df_roll.change_time)[0]  # get the time elapsed over the trials 
                reward_rate_on_this_lap= correct / time_elapsed          # calculate the reward rate

            reward_rate[c]=reward_rate_on_this_lap                       # store the rolling average
            c+=1;trial_number+=1                                         # increment some dumb counters
    df['reward_rate'] = reward_rate * 60.                                # convert to rewards/min
def calculate_date(df,window=1.0):
    date = np.zeros(np.shape(df.change_time));
    c=0
    for startdatetime in df.startdatetime.unique():                      # go through the dataframe by each behavior session
        df_temp = df[df.startdatetime==startdatetime]                    # get a dataframe for just this session
        trial_number = 0
        for trial in range(np.shape(df_temp)[0]):
            date[c]=float(startdatetime.split(' ')[0].split('-')[1:][0])+float(startdatetime.split(' ')[0].split('-')[1:][1])/31.                       # store the rolling average
            c+=1;trial_number+=1                                         # increment some dumb counters
    df['date'] = date                            # convert to rewards/min


#for dealing with color change matrices
#for computing color change matrices
def calculate_color_performance(df):
    dg = np.sort([-2.0,-1.8,-1.4,-1.3,-1.2,-1.1,
          -1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,
          2. ,  1.8,  1.4,  1.3,  1.2,  1.1,  1. ,  0.9,  0.8,  0.7,  0.6,
            0.5,  0.4,  0.3,  0.2,  0.1 ])
    duv = dg

    all_color_changes_count = np.zeros((np.shape(duv)[0],np.shape(duv)[0]))
    all_color_changes_percent = np.zeros((np.shape(duv)[0],np.shape(duv)[0]))
    all_color_changes_latency = np.zeros((np.shape(duv)[0],np.shape(duv)[0]))
    all_color_changes_latency_c = np.zeros((np.shape(duv)[0],np.shape(duv)[0]))
    for g in dg:
        for u in duv:
            ind_u = np.where(duv==u)[0]
            ind_g = np.where(dg==g)[0]
            bools = [df['change'][i] == np.array([0,g,u]) for i in df.index.values]
            bools = [bo.all() for bo in bools]
            temp = df[bools]
            correct,total = get_correct(temp,window=1.0)
            temp['response_latency'] = temp['response_latency'].replace([np.inf, -np.inf], np.nan)
            correct_latencies = temp['response_latency'][temp['response_latency']<1.0]
            if len(np.array(correct_latencies))>0:
                latency_c = np.mean(np.array(correct_latencies))
            else:
                latency_c = np.nan
            latency = np.mean(temp['response_latency'][temp['response_latency'].notnull()])
            all_color_changes_count[ind_u,ind_g] = total
            if total > 0:
                all_color_changes_percent[ind_u,ind_g] = correct/float(total)
            else:
                all_color_changes_percent[ind_u,ind_g]=np.nan
            all_color_changes_latency[ind_u,ind_g] = latency
            all_color_changes_latency_c[ind_u,ind_g] = latency_c
    return all_color_changes_count,all_color_changes_percent,all_color_changes_latency,all_color_changes_latency_c

def clean_performance_matrices(input_matrices,min_trials=3):
    """
    Args:
        input_matrix: a list of same shaped 2D matrices, in which the first is the number of trials
        min_trials:   a minimum number of trials in the trials matrix below which the data will be masked and replaced.
    Returns:
        a cleaned version of the input matrix
    """
    cleaned_matrices = input_matrices
    #mask_nan = np.isnan(input_matrices[0][1])
    mask_min = input_matrices[0][0] < min_trials
    cleaned_matrices[0][1][mask_min]=np.nan
    cleaned_matrices[0][2][mask_min]=np.nan
    cleaned_matrices[0][3][mask_min]=np.nan
    return cleaned_matrices

def plot_color_performace_matrices(inpt,title='',x=(0,32),y=(32,0),smooth=0):
    all_color_changes_count = inpt[0][0]
    if smooth == 0:
        all_color_changes_percent = inpt[0][1]
        all_color_changes_latency = inpt[0][2]
        all_color_changes_latency_c = inpt[0][3]
    else:
        mask = np.isnan(inpt[0][1]);all_color_changes_percent = inpt[0][1];all_color_changes_percent[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), all_color_changes_percent[~mask])
        mask = np.isnan(inpt[0][2]);all_color_changes_latency = inpt[0][2];all_color_changes_latency[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), all_color_changes_latency[~mask])
        mask = np.isnan(inpt[0][3]);all_color_changes_latency_c = inpt[0][3];all_color_changes_latency_c[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), all_color_changes_latency_c[~mask])

    f,ax=plt.subplots(1,4,figsize=(15,5))

    im_count = ax[0].imshow(all_color_changes_count,clim=(0,10),cmap=plt.cm.plasma,interpolation='none')
    ax[0].set_title('#of trials')
    plt.colorbar(im_count,ax=ax[0])

    im_correct =ax[1].imshow(smoothRF(all_color_changes_percent*100.,size=smooth),clim=(0,100),cmap=plt.cm.plasma,interpolation='none')
    ax[1].plot([0,32],[32,0],color='r',ls='--')
    ax[1].set_title('% correct')
    plt.colorbar(im_correct,ax=ax[1])

    mask = np.isnan(all_color_changes_latency)
    lat = all_color_changes_latency
    lat[mask] = 1.0;#np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), all_color_changes_latency[~mask])
    im_latency = ax[2].imshow(smoothRF(all_color_changes_latency,size=smooth),clim=(0.2,1.0),cmap=plt.cm.plasma_r,interpolation='none')
    ax[2].set_title('latency (sec)')
    ax[2].plot([0,32],[32,0],color='r',ls='--')
    plt.colorbar(im_latency,ax=ax[2])    

    mask = np.isnan(all_color_changes_latency_c)
    lat_c = all_color_changes_latency_c
    lat_c[mask] = 1.0;#np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), all_color_changes_latency[~mask])
    im_latency_c = ax[3].imshow(smoothRF(all_color_changes_latency_c,size=smooth),clim=(0.3,0.6),cmap=plt.cm.plasma_r,interpolation='none')
    ax[3].set_title('latency (sec)')
    ax[3].plot([0,32],[32,0],color='r',ls='--')
    plt.colorbar(im_latency_c,ax=ax[3])    

    x0,x1=x
    y0,y1=y
    for axis in ax:
        axis.grid(False)
        axis.set_xlim(0,32)
        axis.set_ylim(0,32)
        axis.set_ylabel(r'$\Delta UV$',size=14)
        axis.set_xlabel(r'$\Delta Green$',size=14)
        axis.plot([0,32],[32,0],color='r',ls='--')
        axis.plot([x0,x1],[y0,y1],color='#717171',ls='-')
    plt.gcf().suptitle('y position: '+title+' deg',fontsize=14,fontweight='bold')
    plt.tight_layout()
    return plt.gcf()

#for analysis of color change matrices
def extract_line(matrix,(x0,x1),(y0,y1),xlims=(0,32),smooth=0,color='k',metric=r'$correct\ [%]$',axes=None,normalize_fp = False,alpha=1.0,ls='-',return_line=False,missing='nearest'):
    num = 33
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    # Extract the values along the line, using cubic interpolation
    if missing == 'nearest':
        data = matrix
        mask = np.isnan(data);
        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        data = smoothRF(data,smooth)
    if missing == 'zero':
        data = matrix
        mask = np.isnan(data);
        data[mask] = 0.#np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        data = smoothRF(data,smooth)
    if missing == 'one':
        data = matrix
        mask = np.isnan(data);
        data[mask] = 1.#np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        data = smoothRF(data,smooth)
    if missing == 'replace':
        data = matrix
        mask = np.isnan(data);
        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        data = smoothRF(data,smooth)
        data[mask] = -40#np.nan
        
    #print np.vstack((x,y))
    zi = ndimage.map_coordinates(np.transpose(data), np.vstack((x,y)))
    if return_line:
        if normalize_fp:
            fp = zi[np.where(deltas==0.)[0]]
            #fp = np.min(zi[xlims[0]:xlims[1]])
            #fp = np.mean(zi[np.where((deltas>=-0.2) & (deltas<=0.2))[0]])
            zi = zi - fp
            
            # maximum = np.max(zi)
            # zi = zi / maximum
            # zi = zi * 100.
        if missing == 'replace':
            zi[np.where(zi<=-10.9)]=np.nan
        return zi
    else:
        if axes==None:
            f,ax =plt.subplots(1,1)
        else:
            ax = axes
        if normalize_fp:
            zi = zi - np.min(zi[xlims[0]:xlims[1]])
            #zi = zi - np.mean(np.transpose(ndimage.gaussian_filter(np.nan_to_num(matrix),smooth))[14:16,14:16])
        if missing == 'replace':
            zi[np.where(zi<=-10.9)]=np.nan
        ax.plot(deltas,zi,'-',color=color,alpha=alpha,ls=ls)
        ax.set_title(r'$along\ equiluminance\ line$',size=12)
        ax.set_xlabel(r'$\Delta\ color$',size=14)
        ax.set_ylabel(metric,size=14)
        #ax.set_xlim(xlims[0],xlims[1])
        #ax.set_ylim(0.4,1.0)

        plt.tight_layout()
        return plt.gcf()
    
    
    
def hyperbolicratio(C,r0,rmax,c50,n):
    """
    see docstring for fit_hyperbolicratio()
    """
    return r0 + rmax * ( C**n / (c50**n+C**n))
def fit_hyperbolicratio(xdata,ydata,r0_guess,rmax_guess,c50_guess,n_guess,bounds=False):
    """
    fit some data with a hyperbolic ratio function.  for contrast response functions, see Contreras and Palmer 2000
    Args:
        xdata
        ydata
        r0_guess
        rmax_guess
        c50_guess
        n_guess
        bounds (optional): if provided, a tuple of lists (length 4, correspoding to [r0,rmax,c50,n]) that are the upper and lower bounds of the fit. default False.
        
        
    Returns:
          the fit results
          return format: ([x1,x2],[32,0])
    """
    if bounds != False:
        bounds = bounds
    else:
        bounds = ([-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf])
    popt,pcov = opt.curve_fit(hyperbolicratio,xdata,ydata,p0=(r0_guess,rmax_guess,c50_guess,n_guess),bounds=bounds)
    r0,rmax,c50,n = popt
    reshaped = (np.linspace(xdata[0],xdata[-1],100),hyperbolicratio(np.linspace(xdata[0],xdata[-1],100),*popt))
    return  r0,rmax,c50,n,pcov, reshaped



def line_points(fit):
    """
    Args:
      2D gaussian fit results, which contain a list of either:
          fit params: [A, xo, yo, sigma_x, sigma_y, theta, offset]
          or
          fit params: [A, sigma_x, sigma_y, theta, offset]
        
    Returns:
          two pairs of Cartesian coordinates, where the line intersects y=32 and y=0
          return format: ([x1,x2],[32,0])
    """
    if len(fit[0])==5:
        xo=16
        yo=16
        angle = fit[0][3]
    else:
        xo = fit[0][1]
        yo = fit[0][2]
        angle = fit[0][5]
    x2 = xo - (32 - yo) * np.tan(angle)
    y1=32
    x1 = xo - yo * np.tan(-angle)
    y2=0
    return ([x1,x2],[y1,y2])
def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()
def fit_2Dgauss(data,center_guess,width_guess=2,height_guess=2):
    dataToFit = data.ravel()
    x=np.linspace(0,np.shape(data)[0]-1,np.shape(data)[0])
    y=np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    x, y = np.meshgrid(x, y)
    popt,pcov = opt.curve_fit(twoD_Gaussian,(x,y),dataToFit,p0=(data[center_guess[1]][center_guess[0]], center_guess[1], center_guess[0], width_guess, height_guess, 0, 0))
    reshaped_to_space=(x,y,twoD_Gaussian((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,pcov,reshaped_to_space
def fit_2Dgauss_centerFixed(data,center_guess,width_guess=2,height_guess=2):
    mask = np.isnan(data);
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    dataToFit = data.ravel()
    x=np.linspace(0,np.shape(data)[0]-1,np.shape(data)[0])
    y=np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    x, y = np.meshgrid(x, y)
    
    def twoD_Gaussian_fixed((x, y), amplitude,sigma_x, sigma_y, theta, offset):
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-center_guess[1])**2) + 2*b*(x-center_guess[1])*(y-center_guess[0]) 
                                + c*((y-center_guess[0])**2)))
        return g.ravel()
    popt,pcov = opt.curve_fit(twoD_Gaussian_fixed,(x,y),dataToFit,p0=(data[center_guess[0]][center_guess[1]], width_guess, height_guess, 0, 0))
    reshaped_to_space=(x,y,twoD_Gaussian_fixed((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,pcov,reshaped_to_space
def fit_line(xdata,ydata,deg):
    fit_params = np.polyfit(xdata,ydata,deg=deg)
    p = np.poly1d(fit_params)
    xp = np.linspace(xdata[0],xdata[-1],100)
    return np.nan,np.nan,np.nan,np.nan,np.nan,(xp,p(xp)) #returns a bunch of leading nans just to match the format of fit_hyperbolic

def fit_performance(deltas,a,fix_zero=False):
    mid = np.where(deltas==0.)[0][0]
    try:
        fit_ON = fit_hyperbolicratio(deltas[mid:],np.nan_to_num(a)[mid:],0,np.max(a[mid:]),deltas[mid:][np.where(np.array(np.nan_to_num(a)[mid:]) > (np.max(np.nan_to_num(a)[mid:]) - np.min(np.nan_to_num(a)[mid:])) / 2.)[0][0]],3,
                                     fix_zero=fix_zero)
    except:
        fit_ON = fit_line(deltas[mid:],np.nan_to_num(a)[mid:],deg=1)
    
    x=deltas[:mid+1][::-1]*-1
    try:
        fit_OFF = fit_hyperbolicratio(x,np.nan_to_num(a)[:mid+1][::-1],0,np.max(a[:mid+1][::-1]),x[np.where(np.array(np.nan_to_num(a)[:mid][::-1]) > (np.max(np.nan_to_num(a)[:mid][::-1]) - np.min(np.nan_to_num(a)[mid:])) / 2.)[0][0]],3,
                                      fix_zero=fix_zero)
    except:
        fit_OFF = fit_line(x,np.nan_to_num(a)[:mid+1][::-1],1)
    return fit_ON, fit_OFF

#for plotting projections of Cartesian coordinates into various coordinate systems, and overlaying information about the mouse visual system.
def plot_spherical(coords,labels=None,radius=15):
    # from https://github.com/healpy/healpy/issues/109
    x = np.array(coords)[:,0]
    y = np.array(coords)[:,1]
    radius = radius ** 2.
    
    # To plot the celestial equator in galactic coordinates
    degtorad = math.pi/180.
    alpha = np.arange(-180,180.,1.)
    alpha *= degtorad
    # From Meeus, Astronomical algorithms (with delta = 0)
    x1 = np.sin(192.25*degtorad - alpha)
    x2 = np.cos(192.25*degtorad - alpha)*np.sin(27.4*degtorad)
    yy = np.arctan2(x1, x2)
    longitude = 303*degtorad - yy 
    x3 = np.cos(27.4*degtorad) * np.cos(192.25*degtorad - alpha)
    latitude  = np.arcsin(x3)
    
    # We put the angles in the right direction
    for i in range(0,len(alpha)):
        if longitude[i] > 2.*math.pi:
            longitude[i] -= 2.*math.pi
        longitude[i] -= math.pi
        latitude[i] = -latitude[i]
    
    # To avoid a line in the middle of the plot (the curve must not loop)
    for i in range(0,len(longitude)-1):
        if (longitude[i] * longitude[i+1] < 0 and longitude[i] > 170*degtorad and longitude[i+1] < -170.*degtorad):
            indice = i
            break
    
    # The array is put in increasing longitude 
    longitude2 = np.zeros(len(longitude))
    latitude2 = np.zeros(len(latitude))
    longitude2[0:len(longitude)-1-indice] = longitude[indice+1:len(longitude)]
    longitude2[len(longitude)-indice-1:len(longitude)] = longitude[0:indice+1]
    latitude2[0:len(longitude)-1-indice] = latitude[indice+1:len(longitude)]
    latitude2[len(longitude)-indice-1:len(longitude)] = latitude[0:indice+1]
    
    xrad = x * degtorad
    yrad = y * degtorad
    
    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(111, projection="mollweide")
    
    ax1.scatter(xrad,yrad,s=np.ones(len(x))*radius)
    ax1.plot([-math.pi, math.pi], [0,0],'r-')
    ax1.plot([0,0],[-math.pi, math.pi], 'r-')
    
    #ax1.plot(longitude2,latitude2,'g-')
    if labels == None:
        labels = ['' for number in range(len(x))]
    for i in range(0,len(x)):
        ax1.text(xrad[i], yrad[i],labels[i])
    plt.title("Visual Space")
    plt.grid(True) 
def plot_mouse_spherical2(coords,projection='sinu',labels=None,radius=15.**2,colors=None):
    x = np.array(coords)[:,0]
    y = np.array(coords)[:,1]
    num_points = len(x)
    if colors == None:
        colors = ['k' for i in range(num_points)]
    
    m=Basemap(projection=projection,lon_0=0,llcrnrlon=-110,urcrnrlon=110,llcrnrlat=-30,urcrnrlat=90)
    x, y = m(x,y)

    m.scatter(x,y,s=np.ones(len(x))*radius,marker='o',c=colors[:num_points])
    m.drawparallels(np.arange(-90,90,15))
    m.drawmeridians(np.arange(-110,110,15))
    m.drawparallels([0],color='r',dashes=[100,1])
    m.drawmeridians([0],color='r',dashes=[100,1])

    x1,y1 = m(-100,-20)
    x2,y2 = m(-100,85)
    x3,y3 = m(100,85)
    x4,y4 = m(100,-20)
    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                           facecolor='bac',edgecolor='green',linewidth=3, alpha=0.2)
    plt.gca().add_patch(poly)
    
def plot_mouse_spherical3(coords,projection='ortho',labels=None,radius=15.**2,colors=None,plot_mouse_visual_field=False):
    x = np.array(coords)[:,0]
    y = np.array(coords)[:,1]
    num_points = len(x)
    if colors == None:
        colors = ['k' for i in range(num_points)]
    
    m=Basemap(projection=projection,lon_0=30,lat_0=20,rsphere = 337./2.)#,llcrnrx=-85,urcrnrx=85,llcrnry=-85,urcrnry=85)
    x, y = m(x,y)

    m.scatter(x,y,s=np.ones(len(x))*radius,marker='o',c=colors[:num_points])
    m.drawparallels(np.arange(-90,90,15))
    m.drawmeridians(np.arange(-110,110,15))
    m.drawparallels([0],color='r',dashes=[100,1])
    m.drawmeridians([0],color='r',dashes=[100,1])
    
    if plot_mouse_visual_field:
        x1,y1 = m(-100,-20)
        x2,y2 = m(-100,85)
        x3,y3 = m(100,85)
        x4,y4 = m(100,-20)
        poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],
                               facecolor='green',edgecolor='green',linewidth=3, alpha=0.2)
        plt.gca().add_patch(poly)
    
    
# class MouseView(image):
#     def __init__(self,**kwargs):
#         self.image = image
#         
#         #mouse cone fundamentals extracted from Joesch and Meister, 2016
#         self.wavelengths = np.arange(350,600)
#         self.green_cone = np.array([ 0.2663895 ,  0.26595151, 0.2663895 ,  0.26595151,  0.26551351,  0.26507551,  0.26463751,0.26419951,  0.26376152,  0.26332352,  0.26093815,  0.25812743,0.25531671,  0.25250599,  0.24969527,  0.24688455,  0.24474235,0.24389064,  0.24303892,  0.24218721,  0.24133549,  0.24048378,0.23963206,  0.23878035,  0.23792864,  0.23707692,  0.23514114,0.23254663,  0.22995212,  0.22735761,  0.2247631 ,  0.22216859,0.21957408,  0.21766776,  0.21621531,  0.21476285,  0.2133104 ,0.21185794,  0.21040548,  0.20895303,  0.20750057,  0.20604812,0.20459566,  0.20349159,  0.20305359,  0.20261559,  0.20217759,0.20173959,  0.2013016 ,  0.2008636 ,  0.2004256 ,  0.20073163,0.20124263,  0.20175363,  0.20226462,  0.20277562,  0.20328662,0.20414483,  0.20579596,  0.20744709,  0.20909822,  0.21074934,0.21240047,  0.2140516 ,  0.21630044,  0.22004806,  0.22379569,0.22754331,  0.23129094,  0.23532846,  0.23975741,  0.24418637,0.24861532,  0.25304428,  0.25822026,  0.26396933,  0.2697184 ,0.27546747,  0.28115997,  0.2866793 ,  0.29219863,  0.29771796,0.30323729,  0.30875662,  0.31664937,  0.32497199,  0.33329462,0.34177854,  0.35251026,  0.36324199,  0.37397371,  0.38386002,0.39356975,  0.40327948,  0.4129892 ,  0.42269893,  0.43240866,0.44300682,  0.45527151,  0.46753619,  0.4781454 ,  0.48785512,0.49756485,  0.50856792,  0.52144598,  0.53432405,  0.54521628,0.55529116,  0.56536604,  0.57544092,  0.58982784,  0.60454569,0.61856156,  0.63021323,  0.6418649 ,  0.65351658,  0.66516825,0.67681992,  0.68770861,  0.69844033,  0.70917206,  0.72116933,0.7340474 ,  0.74692547,  0.75789389,  0.76862592,  0.77935795,0.79008978,  0.8008215 ,  0.81155322,  0.82165581,  0.83085443,0.84005305,  0.84925167,  0.85872661,  0.86836336,  0.87800011,0.88763685,  0.89559606,  0.90326149,  0.91092692,  0.91859234,0.92544329,  0.93157577,  0.93770825,  0.94384072,  0.9499732 ,0.95610568,  0.96157836,  0.96659594,  0.97161351,  0.97663108,0.98164866,  0.98666623,  0.99004729,  0.99255591,  0.99506453,0.99757315,  1.00008176,  1.00259038,  1.00112561,  0.99805944,0.99499327,  0.99192709,  0.98886092,  0.98541062,  0.98111787,0.97682512,  0.97253237,  0.96823961,  0.96394686,  0.95587232,0.94744017,  0.93900801,  0.93057586,  0.92298929,  0.91598079,0.90897229,  0.90196379,  0.88942702,  0.87562916,  0.86132953,0.84676494,  0.83485438,  0.82565581,  0.81645723,  0.80491415,0.78651639,  0.76746986,  0.74817402,  0.73693146,  0.72568889,0.71444633,  0.69943893,  0.68334088,  0.67054608,  0.65981436,0.64908263,  0.63354967,  0.61004223,  0.590977  ,  0.57334596,0.5563548 ,  0.53979718,  0.52323956,  0.50961361,  0.49632675,0.48303988,  0.46639751,  0.44800011,  0.42960271,  0.4135525 ,0.39760827,  0.38182158,  0.36649036,  0.35115913,  0.33924521,0.33004659,  0.32084796,  0.31164934,  0.30040716,  0.28858029,0.27675342,  0.26492655,  0.25062274,  0.23631357,  0.2220044 ,0.21098843,  0.20102328,  0.19105813,  0.18109298,  0.17179996,0.16336781,  0.15493565,  0.1465035 ,  0.13826357,  0.13144965,0.12463573,  0.11782181,  0.11100789,  0.10452955,  0.09901041,0.09349126,  0.08797212,  0.08245297,  0.07693383,  0.07121703,0.06546796,  0.05971889,  0.05396982])
#         self.uv_cone = np.array([8.55815491e-01,   8.66480435e-01, 8.77815491e-01,   8.89480435e-01,   9.01112864e-01,9.12624519e-01,   9.23968072e-01,   9.31949583e-01,9.39931094e-01,   9.47492579e-01,   9.54245912e-01,9.60999245e-01,   9.67362065e-01,   9.73501733e-01,9.79641158e-01,   9.85780583e-01,   9.91133783e-01,9.94715207e-01,   9.98296632e-01,   1.00156798e+00,1.00003322e+00,   9.98498456e-01,   9.96963692e-01,9.92680668e-01,   9.87769054e-01,   9.82641501e-01,9.75734600e-01,   9.68827700e-01,   9.56850990e-01,9.50918464e-01,   9.46006480e-01,   9.38989651e-01,9.20571489e-01,   9.02152472e-01,   8.92102947e-01,8.82126518e-01,   8.46365647e-01,   8.22249893e-01,8.05877838e-01,   7.89505790e-01,   7.73133746e-01,7.53335902e-01,   7.25654579e-01,   7.14910306e-01,6.94059728e-01,   6.62657783e-01,   6.28890295e-01,6.00239525e-01,   5.64040311e-01,   5.42671217e-01,5.14950200e-01,   4.81541712e-01,   4.61587931e-01,4.29089342e-01,   3.92690615e-01,   3.67109357e-01,3.45441795e-01,   3.24721093e-01,   2.94658034e-01,2.67641282e-01,   2.47175918e-01,   2.30059558e-01,2.13175766e-01,   1.83281950e-01,   1.64908939e-01,1.53507043e-01,   1.42105148e-01,   1.30703253e-01,1.15747645e-01,   1.00399080e-01,   8.89231178e-02,7.88367343e-02,   6.87503509e-02,   5.86639674e-02,5.31923514e-02,   4.79735990e-02,   4.27548466e-02,3.75360942e-02,   3.23173418e-02,   2.85430013e-02,2.60150462e-02,   2.34870912e-02,   2.09591361e-02,1.84311810e-02,   1.59032259e-02,   1.33752709e-02,1.08473158e-02,   8.31936071e-03,   6.60923838e-03,6.50332886e-03,   6.39741934e-03,   6.29150981e-03,6.18560029e-03,   6.07969077e-03,   5.97378125e-03,5.86787173e-03,   5.76196220e-03,   5.65605268e-03,5.55014316e-03,   5.44423364e-03,   5.33832412e-03,5.23241459e-03,   5.12650507e-03,   5.02059555e-03,4.90380348e-03,   4.78102236e-03,   4.65824125e-03,4.53546013e-03,   4.41267902e-03,   4.28989790e-03,4.16711678e-03,   4.04433567e-03,   3.92155455e-03,3.79877344e-03,   3.67599232e-03,   3.55321121e-03,3.43043009e-03,   3.32200000e-03,   3.32200000e-03,3.32200000e-03,   3.32200000e-03,   3.32200000e-03,3.32200000e-03,   3.32200000e-03,   3.32200000e-03,3.32200000e-03,   3.32200000e-03,   3.24492968e-03,3.04029447e-03,   2.83565927e-03,   2.63102407e-03,2.42638886e-03,   2.22175366e-03,   2.01711846e-03,1.81248326e-03,   1.49015444e-03,   8.32398394e-04,1.74642344e-04,  -4.83113706e-04,  -1.14086976e-03,-1.79862581e-03,  -2.45638186e-03,  -3.11413791e-03,-2.92834283e-03,  -2.35280636e-03,  -1.77726988e-03,-1.20173340e-03,  -6.26196925e-04,  -5.06604473e-05,5.24876030e-04,   1.10041251e-03,   1.66100000e-03,1.66100000e-03,   1.66100000e-03,   1.66100000e-03,1.66100000e-03,   1.66100000e-03,   1.66100000e-03,1.66100000e-03,   1.66100000e-03,   1.52811998e-03,1.38195198e-03,   1.23578397e-03,   1.08961597e-03,9.43447964e-04,   7.97279959e-04,   6.51111954e-04,5.04943950e-04,   3.58775945e-04,   2.12607940e-04,6.64399357e-05,  -1.28791483e-04,  -3.64909006e-04,-6.01026529e-04,  -8.37144052e-04,  -1.07326157e-03,-1.30937910e-03,  -1.54549662e-03,  -1.78161414e-03,-2.01773167e-03,  -2.25384919e-03,  -2.48996671e-03,-2.72608423e-03,  -2.96220176e-03,  -3.19831928e-03,-3.25844876e-03,  -3.12499101e-03,  -2.99153327e-03,-2.85807553e-03,  -2.72461778e-03,  -2.59116004e-03,-2.45770229e-03,  -2.32424455e-03,  -2.19078681e-03,-2.05732906e-03,  -1.92387132e-03,  -1.79041357e-03,-1.66504417e-03,  -1.79850191e-03,  -1.93195966e-03,-2.06541740e-03,  -2.19887515e-03,  -2.33233289e-03,-2.46579063e-03,  -2.59924838e-03,  -2.73270612e-03,-2.86616387e-03,  -2.99962161e-03,  -3.13307935e-03,-3.26653710e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03,  -3.32200000e-03,-3.32200000e-03,  -3.32200000e-03])
#         #human color matching functions from http://cvrl.ioo.ucl.ac.uk/cmfs.htm
#         cie2006=[]
#         [cie2006.append(row) for row in csv.reader(open('lin2012xyz2e_1_7sf.csv'))];
#         wavelengths = np.array(cie2006)[:,0].astype(int)
#         X = np.array(cie2006)[:,1].astype(float)
#         Y = np.array(cie2006)[:,2].astype(float)
#         Z = np.array(cie2006)[:,3].astype(float)
#         self.X = np.zeros(250);self.X[50:]=X[:200]#convert to wavelength range 350-600
#         self.Y = np.zeros(250);self.Y[50:]=Y[:200]#convert to wavelength range 350-600
#         self.Z = np.zeros(250);self.Z[50:]=Z[:200]#convert to wavelength range 350-600
#         
#         #taken from the behavioral data in Denman et al., 2017
#         self.green_weights = None
#         self.uv_weights = None
#         
#         #options 
#         self.adjust_spatial_resolution = False
#         self.image_limits = [[-20,80],[-10,65]] #[[left,right],[bottom,top]] coordinates of displayed image in visual space, in degrees
#     
#         #initialize with defaults
#         self.apply_display_colors('sRGB')   #1. assume using a standard display calibrated with sRGB standard
#         self.convolve_cones() #2.
#         self.spatial_weight('mouse') #3.
#         self.set_color_table('mouse')#4.
#        
#     def apply_display_colors(display,**kwargs):
#         """
#         Args: 'sRGB', 'AllenDome'
#           string, specifying display spectra
#         Returns: None      
#         Sets the color properties of the display; options are 'sRGB', 'AllenDome', and 'custom'.
#             'sRGB':      assumes a display calibrated for sRGB color space, as most are.
#                          first converts sRGB to XYZ; uses XYZ tristimulus value and the CIE 2006 standard observer color matching functions to compute spectral radiance.
#             'AllenDome': uses the measured spectral output for RGB tristimulus values in the custom dome experimental setup used for some mouse work
#                          at the Allen Insitute for Brain Science. See Denman et al., 2017.
#             'custom':    allows the user to pass a numpy array (3,249) specifying the spectral output from 350 to 600nm for each tristimulus value. spectra should be passed as 'red'=np.array(),'green'=np.array(),'blue'=np.array()
#         """
#         if display =='sRGB':
#             #convert to XYZ using skimage
#             xyz_image = color.rgb2xyz(self.image)
#             self.display = [[self.X,   
#                              self.Y,
#                              self.Z]]
#             
#         if display =='AllenDome':
#             xyz_image = self.image
#             self.display = [[np.zeros(249), #R
#                              np.array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,2.15824075e-03,   6.99597750e-04,   6.18983727e-03,6.76624326e-03,   8.85627628e-03,   4.62471902e-03,1.83978565e-02,   1.20829883e-02,   1.65570517e-02,1.77922228e-02,   1.10305775e-02,   6.14487992e-03,7.88888650e-03,   2.76463534e-03,   7.51312152e-03,1.09348817e-03,   4.17783430e-03,   7.69244007e-03,3.95409617e-03,   6.09481377e-03,   3.96033428e-03,4.80750616e-03,   3.07218452e-04,   2.43447573e-05,2.92897706e-04,   3.25497166e-03,   1.60448014e-03,3.22660443e-03,   2.42002592e-03,   7.85272164e-04,3.41915205e-03,   1.73132065e-04,   3.61968551e-05,1.96178087e-04,   8.19511815e-04,   0.00000000e+00,1.13514902e-03,   5.49859105e-04,   1.74504178e-03,3.68327651e-03,   1.48470052e-03,   2.96465793e-03,3.97157361e-03,   0.00000000e+00,   1.16658690e-04,0.00000000e+00,   5.32123536e-04,   3.78158039e-03,0.00000000e+00,   8.12518150e-05,   0.00000000e+00,3.47408499e-05,   0.00000000e+00,   9.97779021e-04,2.40812782e-04,   2.78455962e-03,   3.11824430e-04,3.19874808e-03,   0.00000000e+00,   1.24535638e-03,1.01213472e-03,   1.97347193e-04,   1.10329436e-03,4.96246384e-04,   1.28097809e-03,   6.91442507e-04,1.18580939e-03,   3.02292502e-03,   3.12639147e-03,6.85938458e-04,   1.30158803e-04,   1.45002796e-03,2.86062682e-03,   2.97183712e-03,   1.35430537e-03,4.09413081e-03,   1.30524216e-03,   2.01631855e-03,5.30773740e-03,   2.31666703e-03,   4.13008056e-03,5.01863364e-03,   7.03601428e-03,   1.06322800e-02,8.05753036e-03,   6.97478946e-03,   9.79080848e-03,1.26722738e-02,   1.37475397e-02,   1.47500188e-02,1.72007056e-02,   2.14614636e-02,   2.05279800e-02,2.07039375e-02,   2.62502823e-02,   2.67032707e-02,3.09393721e-02,   3.68077589e-02,   4.28027899e-02,4.42870280e-02,   4.91285479e-02,   5.77728901e-02,6.05937490e-02,   6.70480334e-02,   7.89445239e-02,8.52280672e-02,   9.66220140e-02,   1.07453187e-01,1.20147994e-01,   1.34395065e-01,   1.50209192e-01,1.62567086e-01,   1.81972423e-01,   1.99895673e-01,2.22801338e-01,   2.45024038e-01,   2.79335212e-01,3.06185938e-01,   3.40022264e-01,   3.70072706e-01,4.06821042e-01,   4.42808436e-01,   4.91879712e-01,5.39391462e-01,   5.88279897e-01,   6.27278789e-01,6.84551018e-01,   7.27561655e-01,   7.71058972e-01,8.09974510e-01,   8.55883713e-01,   8.89698528e-01,9.19840391e-01,   9.42141605e-01,   9.61318391e-01,9.82565581e-01,   9.88911236e-01,   9.96345870e-01,1.00000000e+00,   9.88150295e-01,   9.76402766e-01,9.55238930e-01,   9.38992977e-01,   9.10765566e-01,8.80817299e-01,   8.47690827e-01,   8.05153425e-01,7.75194403e-01,   7.36039558e-01,   6.99611194e-01,6.72720135e-01,   6.38902632e-01,   6.11277520e-01,5.81654603e-01,   5.55078138e-01,   5.27971972e-01,5.02444153e-01,   4.80312335e-01,   4.48766899e-01,4.24282349e-01,   3.98940060e-01,   3.74038741e-01,3.48225905e-01,   3.27363219e-01,   3.09014595e-01,2.88294417e-01,   2.69348871e-01,   2.54345430e-01,2.37749927e-01,   2.20057918e-01,   2.13203803e-01,1.94572045e-01,   1.82891199e-01,   1.71523065e-01,1.64188993e-01,   1.47805640e-01,   1.34419265e-01,1.27905557e-01,   1.18994214e-01,   1.16159642e-01,1.07921583e-01,   9.81922949e-02,   9.23852135e-02,9.20009787e-02,   8.11372705e-02,   7.80835583e-02,7.13391484e-02,   7.03932155e-02,   6.45084268e-02,6.02818439e-02,   5.60149284e-02,   5.03482044e-02,4.85819074e-02,   4.32706476e-02,   3.83210364e-02,4.12642374e-02,   3.55722383e-02,   3.45843596e-02,3.10232638e-02,   2.69709175e-02,   2.85885219e-02,2.55510234e-02,   2.41017671e-02,   2.37799133e-02,2.16977317e-02,   2.00799122e-02,   2.13693978e-02,1.53718930e-02,   1.50047324e-02,   1.59023468e-02,1.52427213e-02,   1.11807490e-02,   1.19359627e-02,1.18455102e-02,   1.15297332e-02,   1.00395259e-02,1.03649021e-02,   1.12037923e-02,   1.27902600e-02,6.94346451e-03]),    #G
#                              np.array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,0.00000000e+00,   0.00000000e+00,   0.00000000e+00,3.46153887e-01,   4.35965214e-01,   5.29394671e-01,6.26785158e-01,   7.45869635e-01,   8.33152814e-01,9.28810318e-01,   9.87726463e-01,   1.00000000e+00,9.74021102e-01,   9.02598946e-01,   8.22163165e-01,7.16789332e-01,   6.27403325e-01,   5.43374773e-01,4.77393551e-01,   4.19099271e-01,   3.75232077e-01,3.37917990e-01,   2.99042104e-01,   2.59147562e-01,2.19891794e-01,   1.86541378e-01,   1.53954953e-01,1.30079939e-01,   1.09419926e-01,   8.68039668e-02,7.27282334e-02,   5.98698149e-02,   4.89291960e-02,4.14626273e-02,   3.31763786e-02,   2.72098442e-02,2.06869059e-02,   1.69181958e-02,   1.00485024e-02,9.03349201e-03,   7.28899080e-03,   4.86908580e-03,4.91969631e-03,   3.67037391e-03,   2.15243384e-03,3.11821799e-03,   1.68074539e-03,   2.21166699e-03,1.37978221e-03,   1.40970259e-03,   2.40997205e-03,1.11075242e-03,   1.92876277e-03,   1.32199673e-03,2.23919395e-03,   1.71346602e-03,   2.07132177e-03,8.67389800e-05,   3.15418795e-03,   1.27972885e-03,1.38518194e-03,   1.77922428e-03,   1.49385795e-03,1.42798874e-03,   1.21360603e-03,   1.25674569e-03,1.08269183e-03,   1.88167634e-03,   1.21645911e-03,2.88506834e-05,   1.21600473e-03,   6.87550523e-04,2.54949833e-03,   1.12703084e-03,   1.79701906e-03,1.10845411e-03,   1.01965457e-03,   7.50254928e-04,1.33196139e-03,   5.69919004e-04,   1.34093803e-03,1.22856886e-03,   1.74065484e-03,   1.19047493e-03,1.06028457e-03,   1.44896417e-03,   1.47570918e-03,1.47925968e-03,   5.53302443e-04,   1.27539111e-03,1.76360101e-03,   4.81477685e-04,   7.34282952e-04,8.90510332e-04,   1.24468878e-03,   9.12415662e-04,3.33309205e-07,   1.57008279e-03,   1.22945120e-03,1.09668778e-03,   6.31249703e-06,   1.00386751e-03,1.49353037e-03,   1.27634742e-03,   0.00000000e+00,1.25559917e-03,   0.00000000e+00,   3.40565544e-04,4.05211630e-04,   0.00000000e+00,   1.35676207e-03,5.05520714e-04,   9.44275079e-04,   9.16763971e-04,4.65298068e-04,   7.46070408e-04,   4.49113167e-04,7.93864817e-04,   1.35669867e-03,   1.16239849e-03,1.03308519e-03,   9.39588628e-04,   2.28831980e-04,2.43305031e-04,   1.91011735e-04,   1.10035981e-03,4.13503532e-04,   1.31677137e-03,   1.50901098e-04,1.31420888e-03,   9.29317532e-04,   7.83931864e-04,1.42632444e-03,   7.84882892e-04,   8.84936255e-06,2.44738441e-04,   4.96783942e-04,   2.10367255e-03,5.35655601e-04,   9.52844893e-04,   1.34272385e-03,1.13668377e-03,   2.01949078e-03,   1.43368962e-03,6.32242998e-04,   1.40845041e-03,   1.72858735e-03,8.80920806e-04,   1.17313453e-03,   7.18358527e-04,9.56226323e-04,   7.34489008e-04,   9.23991779e-04,1.28517084e-03,   9.63417147e-05,   5.93314278e-04,3.82366367e-05,   5.74230328e-04,   8.86970397e-04,6.82156085e-05,   1.24675462e-03,   1.44865245e-03,8.97283760e-05,   1.46410664e-03,   5.90471762e-04,0.00000000e+00,   1.01696527e-03,   2.24050426e-04,6.02423006e-04,   2.12430456e-04,   4.64677258e-04,1.72565502e-04,   0.00000000e+00,   1.16606523e-04,0.00000000e+00,   7.95962360e-04,   4.25466400e-04,6.04599802e-04,   4.18792829e-04,   2.71018497e-04,6.48241392e-04,   3.10497757e-06,   9.35388257e-04,3.38763876e-04,   1.91747724e-04,   7.55881840e-04,3.23413237e-04,   9.41839393e-04,   0.00000000e+00,8.97003735e-04,   4.47295120e-05,   1.32437430e-05,1.39098849e-04,   7.28967765e-05,   4.43591925e-04,1.00780371e-05,   6.28539275e-04,   3.64969963e-05,0.00000000e+00,   8.35065436e-04,   8.18654930e-06,5.38741157e-04,   0.00000000e+00,   3.87164829e-06,7.53277082e-05,   7.66554481e-05,   1.93374509e-04,2.16185958e-04,   0.00000000e+00,   5.75572334e-04,0.00000000e+00,   2.12249761e-04,   6.53398074e-05,1.44429357e-04,   0.00000000e+00,   2.04387935e-04,1.90554713e-04,   0.00000000e+00,   0.00000000e+00,5.66664377e-04,   5.82668054e-04,   2.19746498e-04,2.78927347e-06])]]   #B
#             
#         if display =='custom':
#             xyz_image = self.image
#             self.display = [[kwargs['red'],
#                              kwargs['green'],
#                              kwargs['blue']]]
#             
#         #apply display spectra for XYZ at each point in image; creates a matrix of dimension [image_dim1, image_dim2, n_wavelengths], here n_wavelengths = 250, from 350-600.
#         self.spectrum = xyz_to_spec(xyz_image)
#     
#     def xyz_to_spec(image):
#         x_ = np.einsum('i,jk->jki', self.display[0], image[:,:,0])
#         y_ = np.einsum('i,jk->jki', self.display[1], image[:,:,1])
#         z_ = np.einsum('i,jk->jki', self.display[2], image[:,:,2])
#         spec_image = x_+y_+z_
# 
#             
#     def convolve_cones():
#         """
#         Args: None
#         Returns: None      
#         Convolves the displayed spectra with the mouse cone fundamentals.
#         """
#         self.green_opsin_activation=[np.convolve(self.green_cone,self.display[0]),
#                                      np.convolve(self.green_cone,self.display[1]),
#                                      np.convolve(self.green_cone,self.display[2])]
#         self.uv_opsin_activation=[np.convolve(self.uv_cone,self.display[0]),
#                                   np.convolve(self.uv_cone,self.display[1]),
#                                   np.convolve(self.uv_cone,self.display[2])]
#             
#     def spatial_weight(species):
#         pass
#             
#     
#     def get_original():
#         pass
#         return self.image
# 
#     def get_adjusted():
#         return self.adjusted
# 
#     def show_original():
#         f,ax = plt.subplots(1,1)
#         return plt.gcf()
# 
#     def show_adjusted():
#         f,ax = plt.subplots(1,1)
#         return plt.gcf()
# 
#     def show():
#         f,ax = plt.subplots(1,2)
#         return plt.gcf()
# 
# class Mousitize(image):
#     def __init__(self,**kwargs):        
#        self.green_cone = np.array()
#        self.uv_cone = np.array()
#        self.green_weights = None
#        self.uv_weights = None
#     
#     def get_original():
#         return self.image
# 
#     def get_adjusted():
#         return self.adjusted