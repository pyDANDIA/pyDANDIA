# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:42:26 2017

@author: rstreet
"""

import os
import sys
import numpy as np
import logs
import metadata
import matplotlib.pyplot as plt
from astropy.io import fits
import starfind
import psf

def run_psf_photometry(setup,reduction_metadata,log,ref_star_catalog,
                       image_path,psf_model,sky_model,centroiding=True):
    """Function to perform PSF fitting photometry on all stars for a single
    image.
    
    :param SetUp object setup: Essential reduction parameters
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param array ref_star_catalog: catalog of objects detected in the image
    :param str image_path: Path to image to be photometered
    :param PSFModel object psf_model: PSF to be fitted to each star
    :param BackgroundModel object sky_model: Model for the image sky background
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    
    Returns:
    
    :param array ref_star_catalog: catalog of objects detected in the image
    """
    
    log.info('Starting photometry of '+os.path.basename(image_path))
    
    data = fits.getdata(image_path)
    residuals = np.copy(data)
    
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    half_psf = int(float(psf_size)/2.0)
    
    logs.ifverbose(log,setup,'Applying '+psf_model.psf_type()+\
                    ' PSF of diameter='+str(psf_size))
    
    Y_data, X_data = np.indices((int(psf_size),int(psf_size)))
    
    for j in range(0,len(ref_star_catalog),1):
        
        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        
        X_grid = X_data + (int(xstar) - half_psf)
        Y_grid = Y_data + (int(ystar) - half_psf)
        
        logs.ifverbose(log,setup,' -> Star '+str(j)+' at position ('+\
        str(xstar)+', '+str(ystar)+')')
        
        (fitted_model,good_fit) = psf.fit_star_existing_model(setup, data, 
                                               xstar, ystar, psf_size, 
                                               psf_model, sky_model, 
                                               centroiding=centroiding,
                                               diagnostics=True)
        
        logs.ifverbose(log, setup,' -> Star '+str(j)+
        ' fitted model parameters = '+repr(fitted_model.get_parameters())+
        ' good fit? '+repr(good_fit))
        
        if good_fit == True:
            
            sub_psf_model = psf.get_psf_object('Moffat2D')
            
            pars = fitted_model.get_parameters()
            pars[1] = (psf_size/2.0) + (ystar-int(ystar))
            pars[2] = (psf_size/2.0) + (xstar-int(xstar))
            
            sub_psf_model.update_psf_parameters(pars)
            
            (res_image,corners) = psf.subtract_psf_from_image(data,sub_psf_model,
                                                    xstar,ystar,
                                                    psf_size, psf_size)
            
            residuals[corners[2]:corners[3],corners[0]:corners[1]] = res_image[corners[2]:corners[3],corners[0]:corners[1]]
            
            logs.ifverbose(log, setup,' -> Star '+str(j)+
            ' subtracted PSF from the residuals')
                    
            (flux, flux_err) = fitted_model.calc_flux(Y_grid, X_grid)
            
            (mag, mag_err) = convert_flux_to_mag(flux, flux_err)
            
            ref_star_catalog[j,5] = mag
            ref_star_catalog[j,6] = mag_err
            
            logs.ifverbose(log,setup,' -> Star '+str(j)+
            ' flux='+str(flux)+' +/- '+str(flux_err)+' ADU, '
            'mag='+str(mag)+' +/- '+str(mag_err)+' mag')
    
        else:
            
            logs.ifverbose(log,setup,' -> Star '+str(j)+
            ' No photometry possible from poor PSF fit')
            
    res_image_path = image_path.replace('.fits','_res.fits')
    
    hdu = fits.PrimaryHDU(residuals)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(res_image_path, overwrite=True)
    
    logs.ifverbose(log, setup, 'Output residuals image '+res_image_path)

    plot_ref_mag_errors(setup,ref_star_catalog)

    log.info('Completed photometry')
    
    return ref_star_catalog
    
def run_psf_photometry_on_difference_image(setup,reduction_metadata,log,ref_star_catalog,
                       difference_image,psf_model,sky_model,kernel,centroiding=True):
    """Function to perform PSF fitting photometry on all stars for a single difference image.
    
    :param SetUp object setup: Essential reduction parameters
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param array ref_star_catalog: catalog of objects detected in the image
    :param array_like difference_image: the array of data on which performs photometry
    :param PSFModel object psf_model: PSF to be fitted to each star
    :param BackgroundModel object sky_model: Model for the image sky background
    :param array_like kernel: the kernel data in 2d np.array
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    
    Returns:
    
    :param array ref_star_catalog: catalog of objects detected in the image
    """
    
    log.info('Starting photometry of '+os.path.basename(image_path))
 
    
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    half_psf = int(float(psf_size)/2.0)
    
    logs.ifverbose(log,setup,'Applying '+psf_model.psf_type()+\
                    ' PSF of diameter='+str(psf_size))
    
    Y_data, X_data = np.indices((int(psf_size),int(psf_size)))

    list_image_id = []
    list_star_id = []

    list_ref_mag = []
    list_ref_mag_error = []
    list_ref_flux = []
    list_ref_flux_error  []
    
    list_delta_flux = []
    list_delta_flux_error = []
    list_mag = []
    list_mag_error = []

    list_phot_scale_factor = []
    list_phot_scale_factor_error = []
    list_background = []
    list_background_error = []

    list_align_x = []
    list_align_y = []
	

    phot_scale_factor = np.sum(kernel)
    error_phot_scale_factor = 0

    for j in range(0,len(ref_star_catalog),1):
        
	list_image_id.append(ref_star_catalog[j,0])
	list_star_id.append(0)

	ref_flux = 0
	error_ref_flux = 0

        list_ref_mag.append(ref_star_catalog[j,5])
        list_ref_mag_error.append(ref_star_catalog[j,6])
        list_ref_flux.append(ref_flux)
        list_ref_flux_error.append(error_ref_flux)
    	

        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        
        X_grid = X_data + (int(xstar) - half_psf)
        Y_grid = Y_data + (int(ystar) - half_psf)
        
        logs.ifverbose(log,setup,' -> Star '+str(j)+' at position ('+\
        str(xstar)+', '+str(ystar)+')')
        
        (fitted_model,good_fit) = psf.fit_star_existing_model_with_kernel(setup, data, 
                                               xstar, ystar, psf_size, 
                                               psf_model, sky_model, kernel,
                                               centroiding=centroiding,
                                               diagnostics=True)
        
        logs.ifverbose(log, setup,' -> Star '+str(j)+
        ' fitted model parameters = '+repr(fitted_model.get_parameters())+
        ' good fit? '+repr(good_fit))
        
	

        if good_fit == True:
            
            sub_psf_model = psf.get_psf_object('Moffat2D')
            
            pars = fitted_model.get_parameters()
            pars[1] = (psf_size/2.0) + (ystar-int(ystar))
            pars[2] = (psf_size/2.0) + (xstar-int(xstar))
            
            sub_psf_model.update_psf_parameters(pars)
            
            (res_image,corners) = psf.subtract_psf_from_image_with_kernel(data,sub_psf_model,
                                                    xstar,ystar,
                                                    psf_size, psf_size)
            
            residuals[corners[2]:corners[3],corners[0]:corners[1]] = res_image[corners[2]:corners[3],corners[0]:corners[1]]
            
            logs.ifverbose(log, setup,' -> Star '+str(j)+
            ' subtracted PSF from the residuals')
                    
            (flux, flux_err) = fitted_model.calc_flux_with_kernel(Y_grid, X_grid, kernel)
	
            list_delta_flux.append(flux)
            list_delta_flux_error.append(flux_err)        
	        
            flux = flux+ref_flux/phot_scale_factor
	    flux_err = (error_ref_flux**2+flux_err**2/phot_scale_fator**2+(flux*error_phot_scale_factor/phot_scale_factor**2)**2)**0.5
            
	    (mag, mag_err) = convert_flux_to_mag(flux, flux_err)
            
    	
            list_mag.append(mag)
            list_mag_error.append(mag_err)
	    list_phot_scale_factor.append(phot_scale_factor)
    	    list_phot_scale_factor_error.append(0)
            list_background.append(0)
            list_background_error.append(0)

            list_align_x.append(0)
            list_align_y.append(0)



        else:
            
            logs.ifverbose(log,setup,' -> Star '+str(j)+
            ' No photometry possible from poor PSF fit')

	    list_flux.append(-10**30)
            list_flux_error.append(-10**30)
            list_mag.append(-10**30)
            list_mag_error.append(-10**30)
            list_phot_scale_factor.append(np.sum(kernel))
    	    list_phot_scale_factor_error.append(0)
            list_background.append(0)
            list_background_error.append(0)

            list_align_x.append(0)
            list_align_y.append(0)

    list_image_id = []
    list_star_id = []

    list_ref_mag = []
    list_ref_mag_error = []
    list_ref_flux = []
    list_ref_flux_error  []
    
    list_delta_flux = []
    list_delta_flux_error = []
    list_mag = []
    list_mag_error = []

    list_phot_scale_factor = []
    list_phot_scale_factor_error = []
    list_background = []
    list_background_error = []

    list_align_x = []
    list_align_y = []
    difference_image_photometry = [list_image_id,list_star_id,list_ref_mag,list_ref_mag_error,list_ref_flux,
                                            list_ref_flux_erlist_delta_flux,list_delta_flux_error,list_mag,list_mag_error,
                                            list_phot_scale_factor, list_phot_scale_factor_error, list_background,
                                            list_background_error, list_align_x, list_align_y ]


    log.info('Completed photometry on difference image')
    
    return  difference_image_photometry


def convert_flux_to_mag(flux, flux_err):
    """Function to convert the flux of a star from its fitted PSF model 
    and its uncertainty onto the magnitude scale.
    
    :param float flux: Total star flux
    :param float flux_err: Uncertainty in star flux
    
    Returns:
    
    :param float mag: Measured star magnitude
    :param float flux_mag: Uncertainty in measured magnitude
    """
    def flux2mag(ZP, flux):
        
        return ZP - 2.5 * np.log10(flux)
    
    if flux < 0.0 or flux_err < 0.0:
        
        mag = 0.0
        mag_err = 0.0

    else:        

        ZP = 25.0
        
        mag = flux2mag(ZP, flux)
        
        m1 = flux2mag(ZP, (flux + flux_err))
        m2 = flux2mag(ZP, (flux - flux_err))
        
        mag_err = (m2 - m1)/2.0
    
    return mag, mag_err
    
def plot_ref_mag_errors(setup,ref_star_catalog):
    """Function to output a diagnostic plot of the fitted PSF magnitudes
    against photometric error"""
    
    file_path = os.path.join(setup.red_dir,'ref','ref_image_phot_errors.png')
    
    fig = plt.figure(1)
    
    idx = np.where(ref_star_catalog[:,5] > 0.0)
    
    plt.plot(ref_star_catalog[idx,5], ref_star_catalog[idx,6],'k.')

    plt.yscale('log')    
    
    plt.xlabel('Instrumental magnitude')

    plt.ylabel('Photometric uncertainty [mag]')
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    plt.axis([xmax,xmin,ymin,ymax])
    
    plt.savefig(file_path)

    plt.close(1)
    
