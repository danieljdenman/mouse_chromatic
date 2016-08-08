#import the required packages.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
import os, sys,glob, copy, h5py, csv
import cPickle as pkl
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.ndimage import zoom, gaussian_filter, imread
from scipy.stats import ttest_ind
from dip_test import dip

#define some colors with HEX values, for use throughout in plotting functions
UV='#4B0082'
Green='#6B8E23'

sampling_rate = 25000 # in Hz, for eCube system


#=================================================================================================
#---------plotting helper functions-------------------------------------------------
#=================================================================================================
def placeAxesOnGrid(fig,dim=[1,1],xspan=[0,1],yspan=[0,1],wspace=None,hspace=None):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec
    
    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively
        
    returns:
        subaxes handles
        
        written by doug ollerenshaw
    '''

    outer_grid = gridspec.GridSpec(100,100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0],dim[1],
                                                  subplot_spec=outer_grid[int(100*yspan[0]):int(100*yspan[1]),int(100*xspan[0]):int(100*xspan[1])],
                                                  wspace=wspace, hspace=hspace)
    

    #NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0]*[dim[1]*[fig]] #filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx])
            fig.add_subplot(inner_ax[row,col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist() #remove redundant dimension
    return inner_ax
            
def cleanAxes(ax,bottomLabels=False,leftLabels=False,rightLabels=False,topLabels=False,total=False):
    ax.tick_params(axis='both',labelsize=10)
    ax.spines['top'].set_visible(False);
    ax.yaxis.set_ticks_position('left');
    ax.spines['right'].set_visible(False);
    ax.xaxis.set_ticks_position('bottom')
    if not bottomLabels or topLabels:
        ax.set_xticklabels([])
    if not leftLabels or rightLabels:
        ax.set_yticklabels([])
    if rightLabels:
        ax.spines['right'].set_visible(True);
        #ax.spines['left'].set_visible(False);
        ax.yaxis.set_ticks_position('right');
    if topLabels:
        ax.spines['top'].set_visible(True);
    if total:
        ax.set_frame_on(False);
        ax.set_xticklabels('',visible=False);
        ax.set_xticks([]);
        ax.set_yticklabels('',visible=False);
        ax.set_yticks([])
        
def scatter_withcirclesize(ax,x,y,s,alpha=1.0,c='k',cmap=plt.cm.PRGn,colorbar=False,**kwargs):
    if c != 'k':
        if type(c)==str:
            c = [c for dump in range(len(s))]
            cmap=None
        if type(c)==list:
            if len(c) == len(s):
                c = c
            else:
                print 'incorrect number of colors specified.';return None
    else:
        c = [c for dump in range(len(s))]
    
    points=[]    
    for (x_i,y_i,r_i,c_i) in zip(x,y,s,c):
        points.append(patches.Circle((x_i,y_i),radius=r_i))
    if cmap is not None:
        p = PatchCollection(points,cmap=cmap,alpha=alpha,clim=(-1,1))
        p.set_array(np.array(c))
        ax.add_collection(p)
    else:
        p = PatchCollection(points,color=c,alpha=alpha)
        ax.add_collection(p)
    if colorbar:
        plt.colorbar(p)

def scatter_withellipsesize(ax,x,y,w,h,alpha=1.0,c='k',cmap=plt.cm.PRGn,colorbar=False,**kwargs):
    if c != 'k':
        if type(c)==str:
            c = [c for dump in range(len(s))]
            cmap=None
        if type(c)==list:
            if len(c) == len(w):
                c = c
            else:
                print 'incorrect number of colors specified.';return None
    else:
        c = [c for dump in range(len(w))]
    
    points=[]    
    for (x_i,y_i,w_i,h_i,c_i) in zip(x,y,w,h,c):
        #points.append(patches.Circle((x_i,y_i),radius=r_i))
        points.append(patches.Ellipse((x_i,y_i),width=w_i,height=h_i))
    if cmap is not None:
        p = PatchCollection(points,cmap=cmap,alpha=alpha,clim=(-1,1))
        p.set_array(np.array(c))
        ax.add_collection(p)
    else:
        p = PatchCollection(points,color=c,alpha=alpha)
        ax.add_collection(p)
    if colorbar:
        plt.colorbar(p)
        
def plotsta(sta,taus=(np.linspace(-10,280,30).astype(int)),colorrange=(-0.15,0.15),title='',taulabels=False,nrows=3,cmap=plt.cm.seismic,smooth=None,colorbar=False):
#show the space-space plots of an already computed STRF for a range of taus.
    ncols = np.ceil(len(taus) / nrows ).astype(int)#+1
    fig,ax = plt.subplots(nrows,ncols,figsize=(10,6))
    titleset=False
    m=np.mean(sta[str(taus[3])])
    for i,tau in enumerate(taus):
        if nrows>1:
            axis = ax[int(np.floor(i/ncols))][i%ncols]
        else:
            axis = ax[i]
        if smooth is None:
            img = sta[str(tau)].T 
        else:
            img = smoothRF(sta[str(tau)].T,smooth)

        img = axis.imshow(img,cmap=cmap,vmin=colorrange[0],vmax=colorrange[1],interpolation='none')
        axis.set_frame_on(False);
        axis.set_xticklabels('',visible=False);
        axis.set_xticks([]);
        axis.set_yticklabels('',visible=False);
        axis.set_yticks([])
        axis.set_aspect(1.0)
        if taulabels:
            axis.set_title('tau = '+str(tau),fontsize=8)
        if titleset is not True:
            axis.set_title(title,fontsize=12)
            titleset=True
        else:
			if tau == 0:
				axis.set_title('tau = '+str(tau),fontsize=12)
    plt.tight_layout()
    if colorbar:
        plt.colorbar(img)

#=================================================================================================




#=================================================================================================
#---------custom-made analysis functions-------------------------------------------------
#---------includes some plotting capabilities as well------------------------------------
#=================================================================================================
#returns the F1 frequency of a response. requires the frequency to specified.
#computed in the Fourier domain.
def f1(inp,freq):
    ps = np.fft.fft(inp)**2 / np.sqrt(len(inp))
    return np.abs(ps[freq])

def psth_line(times,triggers,pre=0.5,timeDomain=False,post=1,binsize=0.05,ymax=75,yoffset=0,output='fig',name='',color='#00cc00',linewidth=0.5,axes=None,labels=True,sparse=False,labelsize=18,axis_labelsize=20,error='shaded',alpha=0.5,**kwargs):
    post = post + 1
    peris=[]#np.zeros(len(triggers),len(times))
    p=[]
    if timeDomain:
        samplingRate = 1.0
    else:
        samplingRate = sampling_rate
        
    times = np.array(times).astype(float) / samplingRate + pre
    triggers = np.array(triggers).astype(float) / samplingRate

    numbins = (post+pre) / binsize 
    bytrial = np.zeros((len(triggers),numbins))
    for i,t in enumerate(triggers):
        
        if len(np.where(times >= t - pre)[0]) > 0 and len(np.where(times >= t + post)[0]) > 0:
            start = np.where(times >= t - pre)[0][0]
            end = np.where(times >= t + post)[0][0]
            for trial_spike in times[start:end-1]:
                if float(trial_spike-t)/float(binsize) < float(numbins):
                    bytrial[i][(trial_spike-t)/binsize-1] +=1   
        else:
        	 pass
             #bytrial[i][:]=0
        #print 'start: ' + str(start)+'   end: ' + str(end)

    variance = np.std(bytrial,axis=0)/binsize/np.sqrt((len(triggers)))
    hist = np.mean(bytrial,axis=0)/binsize
    edges = np.linspace(-pre,post,numbins)

    if output == 'fig':
        dump=plt.locator_params(axis='y',nbins=4)
        if error == 'shaded':
            if 'shade_color' in kwargs.keys():
                shade_color=kwargs['shade_color']
            else:
                shade_color=color    
            if axes == None:
                plt.figure()
                axes=plt.gca()
            upper = hist+variance
            lower = hist-variance
            axes.fill_between(edges[2:-1],upper[2:-1]+yoffset,hist[2:-1]+yoffset,alpha=alpha,color='white',facecolor=shade_color)
            axes.fill_between(edges[2:-1],hist[2:-1]+yoffset,lower[2:-1]+yoffset,alpha=alpha,color='white',facecolor=shade_color)
            axes.plot(edges[2:-1],hist[2:-1]+yoffset,color=color,linewidth=linewidth)
            axes.set_xlim(-pre,post-1)
            axes.set_ylim(0,ymax);
            if sparse:
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            else:
                if labels:
                    axes.set_xlabel(r'$time \/ [s]$',fontsize=axis_labelsize)
                    axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=axis_labelsize)
                    axes.tick_params(axis='both',labelsize=labelsize)
            axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
            axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')   
            axes.set_title(name,y=0.5)
            return axes 
        else:
            if axes == None:
                plt.figure()
                axes=plt.gca()
            f=axes.errorbar(edges,hist,yerr=variance,color=color)
            axes.set_xlim(-pre,post - 1)
            axes.set_ylim(0,ymax)
            if sparse:
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            else:
                if labels:
                    axes.set_xlabel(r'$time \/ [s]$',fontsize=axis_labelsize)
                    axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=axis_labelsize)
                    axes.tick_params(axis='both',labelsize=labelsize)
            axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
            axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')   
            axes.set_title(name)
            return axes
    if output == 'hist':
        return (hist,edges)    
    if output == 'p':
        return (edges,hist,variance)
    
def psth_area((data,bins),pre=None,binsize=None, sd = 3,time=0.2):
    if pre is None:
        pre = bins[0]
    if binsize == None:
        binsize = bins[1]-bins[0]
    startbin = np.where(bins>0)[0][0]
    baseline = np.mean(data[:startbin])
    threshold = baseline + np.std(data[:startbin])*sd +0.2
    crossings = plt.mlab.cross_from_below(data[startbin:],threshold)
    if len(crossings)>0:
        try:
            area = np.trapz(np.abs(data[startbin:startbin+np.ceil(time/binsize)]) - baseline)
            return area
        except: return None 
        print 'response did not exceed threshold: '+str(threshold)+', no area returned'
        return None
    
#compute the latency to first reseponse from a PSTH
def psth_latency(data,bins,pre=None,binsize=None, sd = 2.5,smooth=False,offset=0):
    if smooth:
        data = savgol_filter(data,5,3)
    if pre is None:
        pre = bins[0]
    if binsize == None:
        binsize = bins[1]-bins[0]
    startbin = np.where(bins>0)[0][0]
    baseline = np.mean(data[:startbin])
    threshold = baseline + np.std(data[:startbin])*sd +0.2
    crossings = plt.mlab.cross_from_below(data[startbin:],threshold)
    if len(crossings)>0:
        crossing = crossings[0]#the first bin above the threshold
        chunk = np.linspace(data[crossing+startbin-1],data[crossing+startbin],100)
        bin_crossing = plt.mlab.cross_from_below(chunk,threshold)
        latency =(crossing-1)*(1000*binsize)+bin_crossing/100.0 * (1000*binsize) 
    else:
        #print 'response did not exceed threshold: '+str(threshold)+', no latency returned'
        return None
    return latency[0] - offset

#compute a spike-triggered average on three dimensional data. this is typically a movie of the stimulus
#TODO: should be modified to take data of any shape (for example, an LFP trace) to average.
def sta(spiketimes,data,datatimes,taus=(np.linspace(-10,280,30)),exclusion=None,samplingRateInkHz=25):
    output = {}
    for tau in taus:
        avg = np.zeros(np.shape(data[:,:,0]))
        count = 0
        for spiketime in spiketimes:
            if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                if exclusion is not None: #check to see if there is a period we are supposed to be ignoring, because of eye closing or otherwise
                    if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                         index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                         avg += data[:,:,index]
                else:
                	index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                	avg += data[:,:,index]
                count+=1
        output[str(int(tau))]=avg/count
    return output

def impulse(sta,center,taus = np.arange(-10,580,10).astype(int)):
	impulse = [sta[str(tau)][center[0]][center[1]] for tau in taus]
	return (taus,impulse)
            
            
#try to fit an already computed STRF by finding the maximum in space-time, fitting with a 2D gaussian in space-space, and pulling out the temporal kernel at the maximum space-space pixel.
#is very, very finicky right now, requires lots of manual tweaking. 
def fitRF(RF,threshold=None,fit_type='gaussian_2D',verbose=False,rfsizeguess=1.2,flipSpace=False,backup_center=None,zoom_int=10,zoom_order=5,centerfixed=False):
#takes a dictinary containing:
#   
# returns a dictionary containing:
#   the averaged spatial RF, 
#   the centroid of the fit [max fit]
#   a 2D gaussian fit of that spatial RF
#   the impulse response at the center of the fit
#   TODO: a fit of that impulse response with: ?? currently not defined.
    if np.isnan(RF[RF.keys()[0]][0][0]):#check to make sure there is any data in the STRF to try to fit. if not, return the correct data structure filled with None
        fit={};fit['avg_space_fit']=None;fit['params'] = None;fit['cov']=None ;fit['amplitude']=None ;fit['x']=None ;fit['y']=None ;fit['s_x']=None ;fit['s_y']=None ;fit['theta']=None ;fit['offset']=None;fit['center']=None;fit['peakTau']=None;fit['impulse']=None;fit['roughquality']=None
        return fit
    else:
        if 'fit' in RF.keys():
            trash = RF.pop('fit') # get rid of the dictionary entry 'fit'; we only want sta tau frames in this dictionary
        taus = [int(i) for i in RF.keys()]
        taus.sort()
        fit={}
        
        #========================================================================
        #find the taus to average over for the spatial RF
        #first, define the threshold; above this means there is a non-noise pixel somwhere in the space-space
        if threshold == None:
            #set to z sd above mean
            blank = (RF['-10']+RF['0']+RF['10'])/3. # define a blank, noise-only image
            threshold = np.mean(blank)+np.std(blank)*3.
            if verbose:
                print 'threshold: '+str(threshold)
            
        #find the average space-space over only the range of non-noise good 
        avgRF = np.zeros(np.shape(RF[str(int(taus[0]))]))#initialize the average to blank.
        goodTaus = [40,50,60,70,80,90,100]#
        for tau in goodTaus:
            avgRF += RF[str(int(tau))]
        avgRF = avgRF / float(len(goodTaus))
        fit['avg_space']=avgRF
        #========================================================================   
        
        #====fit==================================================================
        maximum_deviation = 0;best_center = (0,0)
        for i in np.linspace(24,63,40):
            for j in np.linspace(10,49,40):
                imp_temp = impulse(RF,(i,j))
                if np.max(np.abs(imp_temp[1])) > maximum_deviation:
                    best_center = (i,j)
                    maximum_deviation = np.max(np.abs(imp_temp[1]))
        center = best_center
        imp_temp = impulse(RF,center)
        if verbose:
            print 'peak frame tau: '+str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]]))
            print 'peak center   : '+str(center)
            print 'peak value    : '+str(RF[str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]]))][center[0],center[1]])
        peak_frame = RF[str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]]))]
        peak = peak_frame[center[0],center[1]]
        #center = (np.where(np.abs(smoothRF(peak_frame,1)) == np.max(np.abs(smoothRF(peak_frame,1))))[0][0],np.where(np.abs(smoothRF(peak_frame,1)) == np.max(np.abs(smoothRF(peak_frame,1))))[1][0])
        
        if verbose:
            print 'peak amp: '+str(peak)+'  threshold: '+str(threshold)
        if np.abs(peak) > threshold * 1.0:
            peak_frame = smoothRF(zoom(peak_frame,zoom_int,order=zoom_order),0)
            fit['roughquality']='good'
        else:
            center = backup_center
            imp_temp = impulse(RF,center)
            peak_frame = RF[str(int(100))]
            peak = peak_frame[center[0],center[1]]
            peak_frame = smoothRF(zoom(peak_frame,zoom_int,order=zoom_order),0)
            print 'could not find a peak in the RF, using center: '+str(center)
            print 'peak amplitude: '+str(peak)+', threshold: '+str(threshold)
            fit['roughquality']='bad'
        fit['center']=center
        fit['center_guess']=center
        fit['fit_image']=peak_frame
        if verbose:
            print 'center guess: '+str(center)
        
        #initialize some empty parameters
        fitsuccess=False;retry_fit = False
        best_fit = 10000000#initialize impossibly high
        fit['avg_space_fit']=None;fit['params']=None
        best_fit_output = ((None,None,None,None,None,None,None),600,None)
        try:
            if centerfixed:
                popt,pcov,space_fit = fit_rf_2Dgauss_centerFixed(peak_frame,(center[0]*zoom_int,center[1]*zoom_int),width_guess=rfsizeguess*zoom_int,height_guess=rfsizeguess*zoom_int)
            else:
                popt,pcov,space_fit = fit_rf_2Dgauss(peak_frame,(center[0]*zoom_int,center[1]*zoom_int),width_guess=rfsizeguess*zoom_int,height_guess=rfsizeguess*zoom_int)
        except:
            popt,pcov,space_fit=((None,0,0,0,0,0,0),600,np.zeros((64,64)))
        
        fit['avg_space_fit']=np.array(space_fit)/float(zoom_int)
        fit['params'] = popt    
        fit['cov']=pcov
        fit['amplitude']=popt[0]
        if centerfixed:
            fit['x']=center[1]
            fit['y']=center[0]
            fit['s_x']=popt[1] / float(zoom_int)
            fit['s_y']=popt[2] / float(zoom_int)
            fit['theta']=popt[3]
            fit['offset']=popt[4] / float(zoom_int)
        else:
            fit['x']=popt[1] / float(zoom_int)
            fit['y']=popt[2] / float(zoom_int)
            fit['s_x']=popt[3] / float(zoom_int)
            fit['s_y']=popt[4] / float(zoom_int)
            fit['theta']=popt[5]
            fit['offset']=popt[6] / float(zoom_int)
        #======================================================================== 
    
    #        
        #============get impulse======================================================================== 
        if verbose:
            print 'center: '+str(center[0])+' '+str(center[1])
            
        if fit['avg_space_fit'] is not None:
            center_h = (np.ceil(fit['y']),np.ceil(fit['x']))
            center_r = (np.round(fit['y']),np.round(fit['x']))
            center_l = (np.floor(fit['y']),np.floor(fit['x']))
        try:
            impuls_h = impulse(RF,center_h,taus)[1]
            impuls_r = impulse(RF,center_r,taus)[1]
            impuls_l = impulse(RF,center_l,taus)[1]
            if np.max(np.abs(impuls_h)) > np.max(np.abs(impuls_r)):
                if np.max(np.abs(impuls_h)) > np.max(np.abs(impuls_l)):
                    impuls = impuls_h
                    center = center_h
                else:
                    impuls = impuls_l
                    center= center_l
            else:
                if np.max(np.abs(impuls_r)) > np.max(np.abs(impuls_l)):
                    impuls= impuls_r
                    center= center_r
                else:
                    impuls= impuls_l
                    center= center_l
        except:
            impuls = np.zeros(len(taus))
        
        if fit_type == 'gaussian_2D':
            #get impulse at the 'center'
            if verbose:
                print 'center from fit: '+str(center[0])+' '+str(center[1])
            #impuls = [RF[str(tau)][center[0]][center[1]] for tau in taus]
            fit['impulse']=(np.array(taus),np.array(impuls))
            peakTau = taus[np.abs(np.array(impuls)).argmax()]
            peakTau = 80
            fit['peakTau']=peakTau
        
        fit['center_usedforfit'] = fit['center']
        fit['center_usedforimp'] = center
        fit['impulse']=(np.array(taus),np.array(impuls))
        #======================================================================== 
        
        return fit
    
#function for fitting with a 2-dimensional gaussian.
def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()
def fit_rf_2Dgauss(data,center_guess,width_guess=2,height_guess=2):
    dataToFit = data.ravel()
    x=np.linspace(0,np.shape(data)[0]-1,np.shape(data)[0])
    y=np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    x, y = np.meshgrid(x, y)
    popt,pcov = opt.curve_fit(twoD_Gaussian,(x,y),dataToFit,p0=(data[center_guess[1]][center_guess[0]], center_guess[1], center_guess[0], width_guess, height_guess, 0, 0))
    reshaped_to_space=(x,y,twoD_Gaussian((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,pcov,reshaped_to_space

def fit_rf_2Dgauss_centerFixed(data,center_guess,width_guess=2,height_guess=2):
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

#smooth a 2D image, meant to be space-space of a receptive field
#size = number of pixels to smooth over
def smoothRF(img,size=3):
    smooth = gaussian_filter(img,(size,size))
    return smooth


def hyperbolicratio(C,r0,rmax,c50,n):
#hyperbolic ratio fit function
    return r0 + rmax * ( C**n / (c50**n+C**n))

def fit_hyperbolicratio(xdata,ydata,r0_guess,rmax_guess,c50_guess,n_guess):
#fit contrast response with the hyperbolic ratio function 
    popt,pcov = opt.curve_fit(hyperbolicratio,xdata,ydata,p0=(r0_guess,rmax_guess,c50_guess,n_guess))
    r0,rmax,c50,n = popt
    return  r0,rmax,c50,n,pcov

def Welchs(a,b,equal_var=False,output=False):
    print 'a: '+str(np.mean(a))+u' \u00B1 '+str(np.std(a))
    print 'b: '+str(np.mean(b))+u' \u00B1 '+str(np.std(b))

    t_stat, p_stat = ttest_ind(a,b,equal_var=False)
    
    print 't: '+str(t_stat)+'  p: '+str(p_stat)
    if output:
        return (t_stat, p_stat)
#=================================================================================================


 
#propagate cell ID numbers from the sorting process
cell_numbers={}
for expt in ['M186118','M186100','M180417','M186098','M179401','M192079','M181423']:
    cell_numbers[expt]={}
cell_numbers['M186118']['lgn_list_ex']=[365,366,75,457,451,115,449,187,371,374,189,155,191,401,394,
 395,121,240,406,463,430,243,88,442,166,247,52]
cell_numbers['M186100']['lgn_list_ex']=[503,505,
 519,549,547,559,557,567,138,715,709,811]
cell_numbers['M180417']['lgn_list_ex']=[405,402,407,357,412,361,371,268,664,662,665,449,451,152,
 457,458,35,463,470,667,493,489,147,196,85,26,
            546,540,550,81,83,298,189,567,581,580,
            16,586,596,593,590,184,597,599,601,608,12,610,232,619,182,624,623,627,
            68,631,630,229,286,636,642,647,659,648,6,649,657,658,654,284]
cell_numbers['M186098']['lgn_list_ex']=[318,256,
 269,376,427,431,437,458,342,137,8,465,391,467,
 7,346,83,58]
cell_numbers['M179401']['lgn_list_ex']=[129,177,176,222,262,127,175,84,46,261,220,174,82,125,472,272,271,172,121,37,80,287,293,294,475,311,295,478,316,320,321,
 33,323,347,325,310,167,327,28,331,209,251,252,480,479,22,351,353,70,357,363,110,361,356,338,339,154,155,473,474,202,65,66,492,482,
 366,484,343,414,426,428,461,9,444,453,192,193,451,382,392,437,55,397,233,236,470,463,454,490,185,440,471,94,229,231,181,91,467,
             468][:44]
cell_numbers['M192079']['lgn_list_ex']=[227,295,107,48,294,226,360,374,376,379,375,380,372,370,153,156,222,41,285,340,152,221,99,40,338,216,388,389,336,390,394,395,
 392,211,400,399,401,94,402,32,406,271,410,407,417,419,409,434,429,431,31,30,435,437,87,136,140,445,439,440,444,28,206,201,264,
 447,449,263,455,462,457,459,84,463,465,467,469,473,477,478,124,479,252,481,191,192,485,70,487,305,248,489,491,493,495,520,245,122,
10,12,14,304,497,13,67,499,179,501,66,503,301,507,116,117,64,65,509,62,115,513,57,230,515,518,162,351,104,159,361][27:101]
cell_numbers['M181423']['lgn_list_ex']=lgn_list = [290,467,471,186,324,44,304,328,325,469,470,134,184,329,318,40,
 77,394,384,263,267,393,117,171,28,73,401,428,425,429,415,438,419,432,436,437,439,440,24,441,206,207,163,
 108,162,252,254,448,22,447,158,202,450,250,16,17,452,201,104,454,453,61,249,456,153,101,102,458,8,195,460,
 462,464,194,3,241,143,1,188,472,431,321,323,281,293,295,289,299,303,45,307,236,88,82,342,368,233,177,176,32,
 ][80:108]
   
#add in manual annotation of color exchange plots. these classification were done by djd based on visual inspection
#of each cell. 
exchange_annotation={'M186118':{},
                    'M186100':{},
                    'M186098':{},
                    'M180417':{},
                    'M181423_5':{}}

exchange_annotation['M186118']['ONOFF_list']=[115,189,247,442,449,463]
exchange_annotation['M186100']['ONOFF_list']=[505]
exchange_annotation['M186098']['ONOFF_list']=[346]
exchange_annotation['M180417']['ONOFF_list']=[12,35,68,81,85,182,189,196,229,232,357,407,567,580,599,619,623,627,636,659]
exchange_annotation['M181423_5']['ONOFF_list']=[17,153,464]
        
exchange_annotation['M186118']['ON_list']=[121,240,366,374,401]
exchange_annotation['M186100']['ON_list']=[709]
exchange_annotation['M186098']['ON_list']=[83,269,318,391,458]
exchange_annotation['M180417']['ON_list']=[152,184,361,405,463,489,493,546,550,596,597,601,610,624,654,658,667]
exchange_annotation['M181423_5']['ON_list']=[201,1]

exchange_annotation['M186118']['OFF_list']=[88,155,187,243,394,451,371]
exchange_annotation['M186100']['OFF_list']=[503,519,547,549,559]
exchange_annotation['M186098']['OFF_list']=[7,8,58,137,342,467]
exchange_annotation['M180417']['OFF_list']=[16,26,83,147,284,286,298,402,581,593,630,642,647,648,649,657,664]
exchange_annotation['M181423_5']['OFF_list']=[250,452,104,456,8,3,188]

exchange_annotation['M186118']['color_list']=[75,430]
exchange_annotation['M186100']['color_list']=[557]
exchange_annotation['M186098']['color_list']=[256,376]
exchange_annotation['M180417']['color_list']=[371,449,590,662,665]
exchange_annotation['M181423_5']['color_list']=[241]
