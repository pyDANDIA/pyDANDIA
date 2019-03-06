# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:09:36 2019

@author: rstreet
Based on pyLIMA and additional code from E. Bachelet
"""

from sys import argv
from os import path
import numpy as np
import copy
from pyLIMA import microltoolbox
import matplotlib.pyplot as plt
import data_handling_utils
from pyLIMA import event
from pyLIMA import microlfits
from pyLIMA import microlmodels
from pyLIMA import microloutputs
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid.inset_locator import inset_axes


MARKER_SYMBOLS = np.array([['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])
MARKER_COLOURS = np.array([['#1b8dd3', '#d2921a', '#0a9331', 
                            '#b20a0a', '#9c43d3', '#5b3c01', '#cf59d1', 
                            '#867987', '#9ea342', '#41a3a8'] * 10])
MARKER_SIZE = 4

def plot_event_lightcurve():
    """Function to plot an event lightcurve with a model overplotted from
    pyLIMA"""
    
    params = get_params()
    
    rescaling_coeffs = read_error_rescaling_factors(params['error_scaling_file'])
    
    params = read_data_files(params, rescaling_coeffs)
    
    current_event = create_event(params)
    
    for d in params['data']:
        
        current_event.telescopes.append(d.tel)
    
    current_event.find_survey(params['survey'])
    current_event.check_event()

    (current_event,params) = create_model(current_event,params)
    
    plot_lcs(current_event,params)
    
def get_params():
    
    params = {}
    
    if len(argv) == 1:
        
        input_file = input('Please enter the path to the parameter file: ')

    else:

        input_file = argv[1]
    
    if path.isfile(input_file) == False:
        
        print('ERROR: Cannot find input parameter file')
        exit()
        
    flines = open(input_file,'r').readlines()
    
    datasets = []
    phot_params = []
    
    for line in flines:
        key = line.replace('\n','').split()[0]
        key = str(key).lower()
        
        if 'dataset' in key:
            entries = line.replace('DATASET','').replace('\n','').split()
            ddict = {'name': entries[0], 'filter': entries[1], 
                     'data_file': entries[2], 'gamma': float(entries[3])}
            datasets.append(ddict)
            phot_params.append(float(entries[4]))
            phot_params.append(float(entries[5]))
        
        if key in ['name','output','error_scaling_file','survey','model_type',\
                    'binary_origin']:
            entries = line.replace(key.upper(),'').replace('\n','').split()
            params[key] = entries[0]
        
        if key in ['ra', 'dec', 'to', 'uo', 'te', 'rho', 'logs', 'logq', \
                    'alpha', 'pien', 'piee', 'sdsdt', 'dalphadt','chisq','topar']:
            entries = line.replace(key.upper(),'').replace('\n','').split()
            params[key] = float(entries[0])
            
    params['datasets'] = datasets
    params['phot_params'] = phot_params
    
    return params

def read_data_files(params, rescaling_coeffs):
    
    datasets = []
    
    for ddict in params['datasets']:
        
        d = data_handling_utils.Dataset(ddict)
        
        if 'gamma' in ddict.keys():
            d.gamma = ddict['gamma']
        
        rescale_factors = fetch_rescaling_factors(d.name,rescaling_coeffs)
        
        d.read_dataset_to_telescope(params['model_type'], 
                                    rescaling=rescale_factors)
        
        datasets.append( d )
        
        print(d.summary())
        
    params['data'] = datasets
    
    return params
    
def read_error_rescaling_factors(coeffs_file):
    
    rescaling_coeffs = {}
    
    if path.isfile(coeffs_file):
        print('Read error rescaling factors:')
        
        flines = open(coeffs_file,'r').readlines()
        
        for line in flines:
            if line[0:1] != '#' and len(line.replace('\n','')) > 0:
                
                
                (tel_code, a0, a1) = line.replace('\n','').split()
                
                rescaling_coeffs[tel_code] = [float(a0), float(a1)]
                print(tel_code+' a0 = '+a0+', a1='+a1)
        
    elif path.isfile(coeffs_file) == False:
        print('ERROR: Error rescaling is switched on but no '+coeffs_file+' found')
        exit()
        
    return rescaling_coeffs

def fetch_rescaling_factors(name,rescaling_coeffs):
    
    if name in rescaling_coeffs.keys():
        factors = rescaling_coeffs[name]
    else:
        factors = []

    return factors
        
def create_event(params):
    """Function to initialise a pyLIMA event object"""
    
    current_event = event.Event()
    current_event.name = params['name']
    
    current_event.ra = params['ra']
    current_event.dec = params['dec']
    
    return current_event

def create_model(current_event,params,diagnostics=False):
    
    f = microlfits.MLFits(current_event)
    
    parallax_params = ['None', params['to']]
    orbital_params = ['None', params['to']]
    model_params = ['to', 'uo', 'tE']
    
    if params['rho'] != None:
        
        model_params.append('rho')
    
    if params['logs'] != None and params['logq'] != None and params['alpha'] != None:
        
        model_params.append('logs')
        model_params.append('logq')
        model_params.append('alpha')
        
    if params['pien'] != None and params['piee'] != None:
        
        parallax_params = ['Full', params['topar']]
        model_params.append('piEN')
        model_params.append('piEE')
        
    if 'sdsdt' in params.keys() and 'dalphadt' in params.keys() and 'sdszdt' in params.keys():
        
        orbital_params = ['3D', params['topar']]
        model_params.append('sdsdt')
        model_params.append('dalphadt')
        model_params.append('sdszdt')
    
    elif 'sdsdt' in params.keys() and 'dalphadt' in params.keys():
        
        orbital_params = ['2D', params['topar']]
        model_params.append('sdsdt')
        model_params.append('dalphadt')
        
    params['model_params'] = model_params
    
    model = microlmodels.create_model(params['model_type'], current_event,
                                  parallax=parallax_params,
                                  orbital_motion=orbital_params,
                                  blend_flux_ratio = False)
    
    if 'binary_origin' in params.keys():
        model.binary_origin  = params['binary_origin']
        
    model.define_model_parameters()
    
    f.model = model
    
    results = []
    for key in model_params:
        results.append(params[str(key).lower()])
    
    params['fitted_model_params'] = results
    
    results = results + params['phot_params']
    results.append(params['chisq'])
    
    f.fit_results = results
    
    current_event.fits.append(f)
    
    if diagnostics:
        fig = microloutputs.LM_plot_lightcurves(f)
    
        plt.show()
        
        plt.close()
    
    return current_event,params

def generate_residual_lcs(current_event,params):
    
    f = current_event.fits[-1]
    model = f.model
    
    pylima_params = model.compute_pyLIMA_parameters(params['fitted_model_params'])
    
    residual_lcs = []
    
    for tel in current_event.telescopes:
        
        flux = tel.lightcurve_flux[:, 1]
        flux_model = model.compute_the_microlensing_model(tel, pylima_params)[0]
        
        dmag = 2.5 * np.log10(flux_model/flux)
        
        res_lc = np.copy(tel.lightcurve_magnitude)
        res_lc[:,1] = dmag
        
        residual_lcs.append(res_lc)
    
    return residual_lcs

def plot_aligned_data(ax,f,current_event):
    
    norm_lcs = microltoolbox.align_the_data_to_the_reference_telescope(f)
    
    for i,lc in enumerate(norm_lcs):
        ax.errorbar(lc[:,0],lc[:,1],yerr=lc[:,2],ls='None', 
                            markersize=MARKER_SIZE,
                            marker=str(MARKER_SYMBOLS[0][i]), capsize=0.0,
                            markerfacecolor=MARKER_COLOURS[0][i],
                            markeredgecolor=MARKER_COLOURS[0][i],
                            label=current_event.telescopes[i].name,
                            alpha=0.5)

        ax.plot(lc[:,0], lc[:,1], 'k.', markersize=1)
        
def plot_lcs(current_event,params):
    
    f = current_event.fits[-1]
    
    norm_lcs = microltoolbox.align_the_data_to_the_reference_telescope(f)
    
    (fig,fig_axes) = initialize_plot_lightcurve(f,len(norm_lcs),title=params['name'])
    
    xmin = 1e9
    xmax = -1e9
    ymin = 1e6
    ymax = -1.6
    for i,lc in enumerate(norm_lcs):
        xmin = min(xmin, lc[:,0].min())
        xmax = max(xmax, lc[:,0].max())
        ymin = min(ymin, lc[:,1].min())
        ymax = max(ymax, lc[:,1].max())
    xmax = xmax * (1.0 + 5e-5)
    ymin = ymin * 0.90
    
    plot_aligned_data(fig_axes[0],f,current_event)
    
    microloutputs.LM_plot_model(f, fig_axes[0])
    
    set_ticks = True
    if set_ticks:
        xticks = np.arange(xmin,xmax,20)
        yticks = np.arange(ymin,ymax,0.2)
        fig_axes[0].set_xticks(xticks, minor=True)
        fig_axes[0].set_yticks(yticks, minor=True)
    
    plt.axis([xmin,xmax,ymax,ymin])
    
    use_legend = True
    if use_legend:
        fig_axes[0].legend(loc='upper left', 
               fontsize = 12,
               bbox_to_anchor=(0.025, 0.95))
               
    add_inset_box(current_event,fig_axes[0])
    add_inset_box2(current_event,fig_axes[0])
    
    plot_residuals(f, fig_axes)

    set_ticks = False
    if set_ticks:
        [xmin,xmax,ymin,ymax] = plt.axis()
        xticks = np.arange(xmin,xmax,20)
        yticks = np.arange(ymin,ymax,2)
        fig_axes[1].set_xticks(xticks, minor=True)
        fig_axes[1].set_yticks(yticks, minor=True)

    plt.tight_layout()
    
    plt.savefig(path.join(params['output'],'lightcurve.pdf'),
                bbox_inches='tight')
    

def initialize_plot_lightcurve(fit,n_datasets,title=None):
    """Function to initialize the lightcurve plot, based on the function 
    from pyLIMA.microloutputs by E. Bachelet

    :param object fit: a fit object. See the microlfits for more details.

    :return: a matplotlib figure  and the corresponding matplotlib axes
    :rtype: matplotlib_figure,matplotlib_axes

    """
    font_size = 18
    
    n_datasets_per_plot = 3
    
    if (float(n_datasets)%float(n_datasets_per_plot)) == 0:
        n_subplots = int(1 + (n_datasets/n_datasets_per_plot))
    else:
        n_subplots = int(1 + (n_datasets/n_datasets_per_plot)) + 1
        
    height_ratios = [3] + [1]*(n_subplots-1)

    fig_size = [10,(5+2*n_subplots)]
    figure, figure_axes = plt.subplots(n_subplots, 1, sharex=True, gridspec_kw={'height_ratios': height_ratios},
                                       figsize=(fig_size[0], fig_size[1]), dpi=75)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.99, wspace=0.2, hspace=0.1)
    figure_axes[0].grid()
    # fig_size = plt.rcParams["figure.figsize"]
    if title != None:
        figure.suptitle(fit.event.name, fontsize=30 * fig_size[0] / len(fit.event.name))

    #figure_axes[0].set_ylabel('Mag', fontsize=5 * fig_size[1] * 3 / 4.0)
    figure_axes[0].set_ylabel('Mag', fontsize=font_size)
    figure_axes[0].yaxis.set_major_locator(MaxNLocator(4))
    #figure_axes[0].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)
    figure_axes[0].tick_params(axis='y', labelsize=font_size)

    #figure_axes[1].set_xlabel('HJD', fontsize=5 * fig_size[0] * 3 / 4.0)
    figure_axes[-1].set_xlabel('HJD', fontsize=font_size)

    for i in range(1,n_subplots,1):
        figure_axes[i].xaxis.set_major_locator(MaxNLocator(6))
        figure_axes[i].yaxis.set_major_locator(MaxNLocator(4))
        figure_axes[i].xaxis.get_major_ticks()[0].draw = lambda *args: None
        figure_axes[i].ticklabel_format(useOffset=False, style='plain')
        #figure_axes[i].set_ylabel('Residuals', fontsize=5 * fig_size[1] * 3 / 4.0)
        figure_axes[i].tick_params(axis='x', labelsize=3.5 * fig_size[0] * 3 / 4.0)
        figure_axes[i].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)
        figure_axes[i].set_ylabel('Residuals', fontsize=font_size)
        figure_axes[i].tick_params(axis='x', labelsize=font_size)
        figure_axes[i].tick_params(axis='y', labelsize=font_size)
        figure_axes[i].grid()

    return figure, figure_axes

def add_inset_box(current_event,ax):
    """Add an inset box to the lightcurve box giving a zoom-in around a 
    selected feature. 
    Based on code by E. Bachelet
    """
    
    inset_axfig1 = inset_axes(ax, width="25%", height="40%",
                              bbox_to_anchor=(0.07, -0.07, 0.97, 1.07),
                              borderpad=5,
                              bbox_transform=ax.transAxes)
                              
    microloutputs.LM_plot_model(current_event.fits[0],inset_axfig1)
    #microloutputs.LM_plot_align_data(current_event.fits[0],inset_axfig1)
    plot_aligned_data(inset_axfig1,current_event.fits[0],current_event)
    
    inset_axfig1.legend_.remove()
    inset_axfig1.texts[0].set_visible(False)

    x1, x2, y1, y2 = 2458225.0,2458250.0,13.25,11.0 # specify the limits
    inset_axfig1.set_xlim(x1, x2) # apply the x-limits
    inset_axfig1.set_ylim(y1, y2) # apply the y-limits
    
    # loc=1 upper right
    # loc=2 upper left
    # loc=3 lower left
    # loc=4 lower right
    patch,pp1,pp2 = mark_inset(ax, inset_axfig1, 
                               loc1=1, loc2=4, fc="none", ec="0.5")
    pp1.loc1 = 3    # Patch on main box
    pp1.loc2 = 1
    pp2.loc1 = 2    # Patch on inset
    pp2.loc2 = 4
    inset_axfig1.get_xaxis().get_major_formatter().set_useOffset(False)
    #inset_axfig1.set_xticks([2457084,2457085.2])
    inset_axfig1.tick_params(axis='both',labelsize=10)
    plt.xticks(rotation=30)
    plt.grid()
    
def add_inset_box2(current_event,ax):
    """Add an inset box to the lightcurve box giving a zoom-in around a 
    selected feature. 
    Based on code by E. Bachelet
    """
    
    inset_axfig2 = inset_axes(ax, width="25%", height="40%",
                              bbox_to_anchor=(-0.4, -0.25, 1.0, 1.2),
                              borderpad=5,
                              bbox_transform=ax.transAxes)
                              
    microloutputs.LM_plot_model(current_event.fits[0],inset_axfig2)
    #microloutputs.LM_plot_align_data(current_event.fits[0],inset_axfig2)
    plot_aligned_data(inset_axfig2,current_event.fits[0],current_event)
    
    inset_axfig2.legend_.remove()
    inset_axfig2.texts[0].set_visible(False)

    x1, x2, y1, y2 = 2458220.0,2458235.0,14.1,13.9 # specify the limits
    inset_axfig2.set_xlim(x1, x2) # apply the x-limits
    inset_axfig2.set_ylim(y1, y2) # apply the y-limits
    
    # loc=1 upper right
    # loc=2 upper left
    # loc=3 lower left
    # loc=4 lower right
    patch,pp1,pp2 = mark_inset(ax, inset_axfig2, 
                               loc1=2, loc2=4, fc="none", ec="0.5")
    pp1.loc1 = 1    # Patch on main plot
    pp1.loc2 = 3
    pp2.loc1 = 4    # Patch on inset
    pp2.loc2 = 2
    
    inset_axfig2.get_xaxis().get_major_formatter().set_useOffset(False)
    #inset_axfig1.set_xticks([2457084,2457085.2])
    inset_axfig2.tick_params(axis='both',labelsize=10)
    
    plt.xticks(rotation=30)
    plt.grid()

def plot_residuals(fit, figure_axes):
    """Plot the residuals from the fit.

    :param object fit: a fit object. See the microlfits for more details.
    :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
    """
    
    def plot_dataset_residuals(fit,pyLIMA_parameters,ax,index):

        telescope = fit.event.telescopes[index]
        time = telescope.lightcurve_flux[:, 0]
        flux = telescope.lightcurve_flux[:, 1]
        error_flux = telescope.lightcurve_flux[:, 2]
        err_mag = microltoolbox.error_flux_to_error_magnitude(error_flux, flux)

        flux_model = fit.model.compute_the_microlensing_model(telescope, 
                                                              pyLIMA_parameters)[0]

        residuals = 2.5 * np.log10(flux_model / flux)

        ax.errorbar(time, residuals, yerr=err_mag, ls='None', 
                    markersize=MARKER_SIZE,
                            marker=str(MARKER_SYMBOLS[0][index]), capsize=0.0,
                            markerfacecolor=MARKER_COLOURS[0][index],
                            markeredgecolor=MARKER_COLOURS[0][index],
                            ecolor=MARKER_COLOURS[0][index],
                            alpha=0.5)
                            
        ax.plot(time, residuals, 'k.', markersize=1)
                            
    plot_residuals_windows = 0.2
    MAX_PLOT_TICKS = 2

    n_plot = 1
    
    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)
        
    for index in range(0,len(fit.event.telescopes),3):
        
        ax = figure_axes[n_plot]
        
        plot_dataset_residuals(fit,pyLIMA_parameters,ax,index)
        
        if index+1 < len(fit.event.telescopes):
            plot_dataset_residuals(fit,pyLIMA_parameters,ax,index+1)
            
        if index+2 < len(fit.event.telescopes):
            plot_dataset_residuals(fit,pyLIMA_parameters,ax,index+2)
            
        ax.set_ylim([-plot_residuals_windows, plot_residuals_windows])
        ax.invert_yaxis()
        ax.yaxis.get_major_ticks()[-1].draw = lambda *args: None

        n_plot += 1
        
    # xticks_labels = figure_axe.get_xticks()
    # figure_axe.set_xticklabels(xticks_labels, rotation=45)

def generate_model_lightcurve(e,ts=None,diagnostics=False):
    """Function to produce a model lightcurve based on a parameter set
    fitted by pyLIMA
    
    Inputs:
    e  Event object, with attributed lightcurve data and model fit(s)
    """
    
    lc = e.telescopes[0].lightcurve_magnitude
    
    fit_params = e.fits[-1].model.compute_pyLIMA_parameters(e.fits[-1].fit_results)
    
    if type(ts) != type(np.zeros(1)):
        ts = np.linspace(lc[:,0].min(), lc[:,0].max(), len(lc[:,0]))

    reference_telescope = copy.copy(e.fits[-1].event.telescopes[0])
    
    reference_telescope.lightcurve_magnitude = np.array([ts, [0] * len(ts), [0] * len(ts)]).T
    
    reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()

    if e.fits[-1].model.parallax_model[0] != 'None':
        
        reference_telescope.compute_parallax(e.fits[-1].event, e.fits[-1].model.parallax_model)

    flux_model = e.fits[-1].model.compute_the_microlensing_model(reference_telescope, fit_params)[0]
    
    mag_model = microltoolbox.flux_to_magnitude(flux_model)

    if diagnostics:
        fig = plt.figure(1,(10,10))
    
        plt.plot(ts,mag_model,'r-')
    
        plt.xlabel('HJD')
        plt.ylabel('Magnitude')
        
        rev_yaxis = True
        if rev_yaxis:
            [xmin,xmax,ymin,ymax] = plt.axis()
            plt.axis([xmin,xmax,ymax,ymin])
        
        plt.grid()
        plt.savefig('lc_model_test.png')
        
        plt.close(1)
        
    return mag_model
    

if __name__ == '__main__':
    
    plot_event_lightcurve()
    