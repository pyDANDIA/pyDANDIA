# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:04:25 2018

@author: rstreet
"""

from os import path
from sys import argv
import numpy as np
from scipy import optimize, odr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_bicolour_flux_curves(lc1,lc2,a,f1,f2,plot_file):
    """Function to plot the flux lightcurves in two bandpasses"""

    idx = extract_valid_data(lc1, lc2)

    fig = plt.figure(1,(10,10))

    plt.rcParams.update({'font.size': 18})

    plt.errorbar(lc2['flux'][idx],lc1['flux'][idx],
                 xerr=lc2['flux_err'][idx], yerr=lc1['flux_err'][idx],
                color='k',fmt='+',marker='+',markersize=10)

    ydata = straight_line_function(a,lc2['flux'][idx])

    plt.plot(lc2['flux'][idx],ydata,'b-')

    plt.xlabel('SDSS-'+f2+' flux [ADU]')
    plt.ylabel('SDSS-'+f1+' flux [ADU]')

    plt.xticks(rotation=30)

    plt.savefig(plot_file)

    plt.close(1)

def extract_valid_data(lc1, lc2):
    """Function to identify the indices of points where valid measurements
    are available in both colours"""

    idx1 = np.where(lc1['flux_err'] > 0.0)[0]
    idx2 = np.where(lc2['flux_err'] > 0.0)[0]
    idx = list(set(idx1).intersection(set(idx2)))

    return idx

def measure_source_colour(lc1,lc2):
    """Function to determine the source colour from a straight line fit to the
    curve:

    flux_total(f1) = a0 + a1*flux_total(f2)

    where:
    a0 = source colour i.e. flux(source,f1)/flux(source,f2)
    flux_total includes the blended flux
    a1 = blend flux in filter 2

    Note that this function implicitly assumes that the flux measurements in
    multiple passbands are made effectively simultaneously - i.e., it can
    only be applied to trios of images.
    """

    idx = extract_valid_data(lc1, lc2)

    init_pars = [0.0, 1.0]

    (fit,flag) = optimize.leastsq(straight_line_residuals, init_pars, args=(
        lc2['flux'][idx].data,lc1['flux'][idx].data))

    colour = -2.5 * np.log10(fit[1])

    blend_flux = fit[0]

    return colour, blend_flux, fit


def measure_source_colour_odr(lc1,lc2):
    """Function to determine the source colour from a straight line fit to the
    curve, using SciPy's ODR package:

    flux_total(f1) = a0 + a1*flux_total(f2)

    where:
    a0 = source colour i.e. flux(source,f1)/flux(source,f2)
    flux_total includes the blended flux
    a1 = blend flux in filter 2

    Note that this function implicitly assumes that the flux measurements in
    multiple passbands are made effectively simultaneously - i.e., it can
    only be applied to trios of images.
    """

    idx = extract_valid_data(lc1, lc2)

    init_pars = [0.0, 1.0]

    linear_model = odr.Model(straight_line_function)

    data = odr.RealData(lc2['flux'][idx].data, lc1['flux'][idx].data,
                        sx=lc2['flux_err'][idx].data, sy=lc1['flux_err'][idx].data)

    odr_fitter = odr.ODR(data, linear_model, beta0=init_pars)

    results = odr_fitter.run()

    fit = results.beta
    sig_fit = results.sd_beta

    colour = -2.5 * np.log10(fit[1])

    sig_colour = (sig_fit[1]/fit[1])*colour

    blend_flux = fit[0]

    sig_blend_flux = (sig_fit[0]/fit[0]) * blend_flux

    return colour, sig_colour, blend_flux, sig_blend_flux, fit

def straight_line_residuals(a,x,y):
    """Function to calculate the residuals to a fit of a straight line"""

    residuals = y - straight_line_function(a,x)

    return residuals

def straight_line_function(a,x):

    y = a[0] + a[1] * x

    return y
