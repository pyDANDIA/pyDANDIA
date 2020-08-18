# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 17:15:16 2019

@author: rstreet
Based on pyLIMA and additional code from E. Bachelet
"""

from sys import argv
from os import path
import numpy as np
import copy
from pyLIMA import microltoolbox
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import data_handling_utils
import pylima_lightcurve_tools
from pyLIMA import event
from pyLIMA import microlfits
from pyLIMA import microlmodels
from pyLIMA import microloutputs
from pyLIMA import microlcaustics
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid.inset_locator import inset_axes

def plot_event_caustic():
    """Function to plot an event lens plane and caustic structures from
    pyLIMA"""

    params = pylima_lightcurve_tools.get_params()

    params = pylima_lightcurve_tools.read_data_files(params, {})


    current_event = pylima_lightcurve_tools.create_event(params)

    for k,d in enumerate(params['data']):

        if k == 0:
            mint = d.tel.lightcurve_flux[:,0].min()
            maxt = d.tel.lightcurve_flux[:,0].max()

            ts = np.arange(mint,maxt,(1.0/24.0))

            flux_lc = np.zeros([len(ts),3])
            flux_lc[:,0] = ts

            d.tel.lightcurve_flux = flux_lc

        current_event.telescopes.append(d.tel)

    current_event.find_survey(params['survey'])
    current_event.check_event()

    (current_event,params) = pylima_lightcurve_tools.create_model(current_event,params)

    plot_lens_plane(current_event,params)

def plot_lens_plane(current_event,params):
    """Based on code from E. Bachelet"""

    f = current_event.fits[-1]

    ref_tel = copy.copy(f.event.telescopes[0])

    pyLIMA_parameters = f.model.compute_pyLIMA_parameters(f.fit_results)

    (to, uo) = f.model.uo_to_from_uc_tc(pyLIMA_parameters)

    (trajectory_x, trajectory_y, sep) = f.model.source_trajectory(ref_tel, to, uo,
                                                                pyLIMA_parameters.tE,
                                                                pyLIMA_parameters)

    (regime, caustics, cc) = microlcaustics.find_2_lenses_caustics_and_critical_curves(10**pyLIMA_parameters.logs,
                                                                                        10** pyLIMA_parameters.logq,
                                                                                        resolution=5000)

    fig = plt.figure(2,(10,10))
    ax = plt.axes()
    ax.axis('equal')

    for count, caustic in enumerate(caustics):

                try:
                    ax.plot(caustic.real, caustic.imag,lw=3,c='r')
                    ax.plot(cc[count].real, cc[count].imag, '--k')

                except AttributeError:
                    pass

    ax.plot(trajectory_x, trajectory_y,'b',lw=2)

    ax.axis([-1.5,1.5,-1.5,1.5])

    ax.tick_params(axis='both',labelsize=18)

    ticks = np.arange(-1.5,1.5,0.1)

    ax.set_xticks(ticks, minor=True)
    ax.set_yticks(ticks, minor=True)

    ax.set_xlabel(r'X ${\rm  [\theta_E]}$',fontsize=20+5)
    ax.set_ylabel(r'Y ${\rm  [\theta_E]}$',fontsize=20+5)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    #inset_ax.arrow(0.09,0.009+0.0015,-0.01,-0.001,
    ##                         head_width=0.01, head_length=0.01, color='b')

    #handles, labels = aaa.axes[0].get_legend_handles_labels()
    tel = current_event.telescopes[0]

    (trajectory_x, trajectory_y, sep) = f.model.source_trajectory(tel, to,
                                                            uo, pyLIMA_parameters.tE,
                                                            pyLIMA_parameters)

    if 'rho' in pyLIMA_parameters._fields:
        # index_source = np.where((trajectory_x ** 2 + trajectory_y ** 2) ** 0.5 < max(1, pyLIMA_parameters.uo + 0.1))[0][
        #   0]
        index_source = np.argmin((trajectory_x ** 2 + trajectory_y ** 2) ** 0.5)
        source_disk = plt.Circle((trajectory_x[index_source], trajectory_y[index_source]), pyLIMA_parameters.rho,
                                 color='y')
        ax.add_artist(source_disk)

    plt.grid()
    plt.tight_layout()
    #plt.show()
    fig.savefig(path.join(params['output'],'lens_plane_caustics.pdf'),
                dpi=fig.dpi,
                bbox_inches='tight')


if __name__ == '__main__':

    plot_event_caustic()
