"""
Created on Wed Oct 18 15:42:26 2017

@author: rstreet
"""

import os
import sys
import numpy as np
from photutils import aperture_photometry
from photutils import CircularAperture
from photutils.utils import calc_total_error

from pyDANDIA import logs
from pyDANDIA import metadata
import matplotlib.pyplot as plt
from astropy.io import fits
from pyDANDIA import starfind
from pyDANDIA import psf
from pyDANDIA import convolution
from scipy.odr import *
import scipy.optimize as so
import scipy.ndimage as sndi


def linear_func(p, x):
    a, b = p
    return a * x + b


def run_psf_photometry(setup,reduction_metadata,log,ref_star_catalog,
                       image_path,psf_model,sky_model,
                       centroiding=True, diagnostics=True, psf_size=None):
    """Function to perform PSF fitting photometry on all stars for a single
    image.
    
    :param SetUp object setup: Essential reduction parameters
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param array ref_star_catalog: catalog of objects detected in the image
    :param str image_path: Path to image to be photometered
    :param PSFModel object psf_model: PSF to be fitted to each star
    :param BackgroundModel object sky_model: Model for the image sky background
    :param float ref_flux: Reference flux value for optimized PSF measurement
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    
    Returns:
    
    :param array ref_star_catalog: catalog of objects detected in the image
    """

    log.info('Starting photometry of ' + os.path.basename(image_path))

    data = fits.getdata(image_path)
    if psf_size == None:
        psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]

    half_psf = int(float(psf_size)/2.0)
    
    exp_time = reduction_metadata.extract_exptime(os.path.basename(image_path))
    
    gain = reduction_metadata.get_gain()
    
    logs.ifverbose(log,setup,'Applying '+psf_model.psf_type()+\
                    ' PSF of diameter='+str(psf_size))
    logs.ifverbose(log,setup,'Scaling fluxes by exposure time '+str(exp_time)+'s')
    
    Y_data, X_data = np.indices((int(psf_size),int(psf_size)))
    
    Y_image, X_image = np.indices(data.shape)
    
    sky_bkgd = sky_model.background_model(data.shape,sky_model.get_parameters())
    
    residuals = np.copy(data)
    
    jincr = int(float(len(ref_star_catalog))*0.01)
    
    for j in range(0,len(ref_star_catalog),1):
        
        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        
        X_grid = X_data + (int(xstar) - half_psf)
        Y_grid = Y_data + (int(ystar) - half_psf)
        
        corners = psf.calc_stamp_corners(xstar, ystar, psf_size, psf_size, 
                                         data.shape[1], data.shape[0],
                                         over_edge=True)
        
        if diagnostics:
            logs.ifverbose(log,setup,' -> Star '+str(j)+' at position ('+\
                            str(xstar)+', '+str(ystar)+')')
        
            logs.ifverbose(log,setup,' -> Corners of PSF stamp '+repr(corners))
        
        (data_section, sec_xstar, sec_ystar) = psf.extract_image_section(data,
                                                            xstar,ystar,corners)
        
        (sky_section, sky_x, sky_y) = psf.extract_image_section(sky_bkgd,
                                                            xstar,ystar,corners)
        
        (fitted_model,good_fit) = psf.fit_star_existing_model(setup, data_section, 
                                               sec_xstar, sec_ystar, 
                                               psf_size, psf_model, 
                                               sky_section, 
                                               centroiding=centroiding,
                                               diagnostics=False)
        
        if diagnostics:
            logs.ifverbose(log, setup,' -> Star '+str(j)+
                            ' fitted model parameters = '+
                            repr(fitted_model.get_parameters()))
        
        fitted_pars = fitted_model.get_parameters()
        fitted_pars[1] = ystar
        fitted_pars[2] = xstar
        fitted_model.update_psf_parameters(fitted_pars)
    
        if diagnostics:
            logs.ifverbose(log, setup,' -> Star '+str(j)+
                            ' updated centroid parameters = '+\
                            repr(fitted_model.get_parameters())+
                            ' good fit? '+repr(good_fit))
        
        if good_fit == True:
            
            sub_psf_model = psf.get_psf_object('Moffat2D')
            
            pars = fitted_model.get_parameters()
            pars[1] = (psf_size/2.0) + (sec_ystar-int(sec_ystar))
            pars[2] = (psf_size/2.0) + (sec_xstar-int(sec_xstar))
            
            pars[1] = sec_ystar
            pars[2] = sec_xstar
            
            sub_psf_model.update_psf_parameters(pars)
            
            sub_corners = psf.calc_stamp_corners(sec_xstar, sec_ystar, 
                                                 psf_size, psf_size, 
                                                 data_section.shape[1], 
                                                 data_section.shape[0],
                                                 over_edge=True)
                                         
            psf_image = psf.model_psf_in_image(data_section,sub_psf_model,
                                                    [sec_xstar,sec_ystar],
                                                    sub_corners)
            
            residuals[corners[2]:corners[3],corners[0]:corners[1]] -= psf_image
            
            if diagnostics:
                logs.ifverbose(log, setup,' -> Star '+str(j)+
                                ' subtracted PSF from the residuals')
                    
            (flux, flux_err) = fitted_model.calc_flux(Y_grid, X_grid)
            
            (mag, mag_err, flux_scaled, flux_err_scaled) = convert_flux_to_mag(flux, flux_err, exp_time=exp_time)
            
            ref_star_catalog[j,5] = flux_scaled
            ref_star_catalog[j,6] = flux_err_scaled
            ref_star_catalog[j,7] = mag
            ref_star_catalog[j,8] = mag_err
            
            if diagnostics:
                logs.ifverbose(log,setup,' -> Star '+str(j)+
                                ' flux='+str(flux_scaled)+' +/- '+\
                                str(flux_err_scaled)+' ADU, '
                                'mag='+str(mag)+' +/- '+str(mag_err)+' mag')
    
        else:
            
            if diagnostics:
                logs.ifverbose(log,setup,' -> Star '+str(j)+
                                ' No photometry possible from poor PSF fit')
        
        if j%jincr == 0:
            percentage = round((float(j)/float(len(ref_star_catalog)))*100.0,0)
            logs.ifverbose(log,setup,' -> Photometry '+str(percentage)+\
                            '% complete ('+str(j)+' stars out of '+\
                            str(len(ref_star_catalog))+')')
                            
    res_image_path = os.path.join(setup.red_dir,'ref',os.path.basename(image_path).replace('.fits','_res.fits'))
    
    hdu = fits.PrimaryHDU(residuals)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(res_image_path, overwrite=True)
    
    logs.ifverbose(log, setup, 'Output residuals image '+res_image_path)

    plot_ref_mag_errors(setup,ref_star_catalog)

    log.info('Completed photometry')

    return ref_star_catalog

def run_psf_photometry_naylor(setup,reduction_metadata,log,ref_star_catalog,
                       image_path,psf_model,sky_model,ref_flux,
                       centroiding=True,diagnostics=True, psf_size=None):
    """Function to perform PSF fitting photometry on all stars for a single
    image.
    
    Updated to implement the optimized photometry algorithm by 
    Naylor, T., 1998, MNRAS, 296, 339.
    
    :param SetUp object setup: Essential reduction parameters
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param array ref_star_catalog: catalog of objects detected in the image
    :param str image_path: Path to image to be photometered
    :param PSFModel object psf_model: PSF to be fitted to each star
    :param BackgroundModel object sky_model: Model for the image sky background
    :param float ref_flux: Reference flux value for optimized PSF measurement
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    
    Returns:
    
    :param array ref_star_catalog: catalog of objects detected in the image
    """

    log.info('Starting photometry of ' + os.path.basename(image_path))

    data = fits.getdata(image_path)
    if psf_size == None:
        psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]

    half_psf = int(float(psf_size)/2.0)
    
    exp_time = reduction_metadata.extract_exptime(os.path.basename(image_path))
    
    gain = reduction_metadata.get_gain()
    
    logs.ifverbose(log,setup,'Applying '+psf_model.psf_type()+\
                    ' PSF of diameter='+str(psf_size))
    logs.ifverbose(log,setup,'Scaling fluxes by exposure time '+str(exp_time)+'s')
    
    Y_data, X_data = np.indices((int(psf_size),int(psf_size)))
    
    Y_image, X_image = np.indices(data.shape)
    
    sky_bkgd = sky_model.background_model(data.shape,sky_model.get_parameters())
    
    for j in range(0,len(ref_star_catalog),1):
        
        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        
        logs.ifverbose(log,setup,' -> Star '+str(j)+' at position ('+\
        str(xstar)+', '+str(ystar)+')')
                
        corners = psf.calc_stamp_corners(xstar, ystar, psf_size, psf_size, 
                                    data.shape[1], data.shape[0],
                                    over_edge=True)
        
        dx = corners[1] - corners[0]
        dy = corners[3] - corners[2]
        
        logs.ifverbose(log,setup,' -> Corners of PSF stamp '+repr(corners)+\
                        ', dimensions dx='+str(dx)+', dy='+str(dy))
            
        if dx >= psf_size-1 and dy >= psf_size-1:
            Y_grid, X_grid = np.indices((int(dy),int(dx)))
            
            xstar_psf = xstar - corners[0]
            ystar_psf = ystar - corners[2]
                    
            psf_data = data[corners[2]:corners[3],corners[0]:corners[1]]
            
            logs.ifverbose(log,setup,' -> Shape of PSF stamp '+repr(psf_data.shape))
            logs.ifverbose(log,setup,' -> Centroid of star in stamp '+\
                                            repr(xstar_psf)+', '+repr(ystar_psf))
            
            (flux, flux_err, Fij) = psf_model.calc_optimized_flux(ref_flux,
                                                             sky_model.varience,
                                                             ystar_psf, xstar_psf,
                                                             Y_grid,X_grid,
                                                             gain,psf_data)
            
            logs.ifverbose(log, setup,' -> Star '+str(j)+
            ' measured optimized flux = '+repr(flux)+' +/- '+str(flux_err))
            
            (mag, mag_err, flux_scaled, flux_err_scaled) = convert_flux_to_mag(flux, flux_err, exp_time=exp_time)
            
            ref_star_catalog[j,5] = flux_scaled
            ref_star_catalog[j,6] = flux_err_scaled
            ref_star_catalog[j,7] = mag
            ref_star_catalog[j,8] = mag_err
            
            logs.ifverbose(log,setup,' -> Star '+str(j)+
            ' flux='+str(flux)+' +/- '+str(flux_err)+' ADU, '
            'mag='+str(mag)+' +/- '+str(mag_err)+' mag')
        
        else:
            logs.ifverbose(log, setup,' -> Star stamp smaller than PSF diameter, '+str(psf_size))
            
    plot_ref_mag_errors(setup, ref_star_catalog)

    log.info('Completed photometry')

    return ref_star_catalog

def quick_offset_fit(params, psf_model, psf_parameters,X_grid, Y_grid, min_x, max_x, min_y, max_y, kernel, data,weight):
   
    if np.abs(params[0])>1 or np.abs(params[1])>1:
        return [np.inf]*len(data.ravel())

    papa = np.copy(psf_parameters)
    #papa[0] = params[0]
    papa[1] = psf_parameters[2]+params[0]
    papa[2] = psf_parameters[1]+params[1]


    model = quick_model(papa.tolist(), psf_model, X_grid, Y_grid, min_x, max_x, min_y, max_y, kernel)
    intensities = np.polyfit(model.ravel(), data.ravel(), 1, w=1/weight.ravel())
    #import pdb;
    #pdb.set_trace()
    residus = data - model*intensities[0]-intensities[1]
    residus /= weight
    return residus.ravel()

def quick_offset_fit2(params, data,weight):
    model = params.reshape(data.shape)

    residus = data - model
    residus /= weight
    return residus.ravel()


def quick_intensities_fit(params, psf_model, psf_parameters,X_grid, Y_grid, min_x, max_x, min_y, max_y, kernel, data,weight):
    
    model = quick_model(params, psf_model, X_grid, Y_grid, min_x, max_x, min_y, max_y, kernel)
    #print(params)
    residus = data.ravel()-model.ravel()
  
    return residus/weight.ravel()

def quick_model(params, psf_model, X_grid, Y_grid, min_x, max_x, min_y, max_y, kernel):

    psf_params = np.copy(params)
  
    #psf_params[-1] = 0

    psf_image = psf_model.psf_model(Y_grid, X_grid, psf_params)
    #psf_convolve = convolution.convolve_image_with_a_psf(psf_image, kernel, fourrier_transform_psf=None,
    #                                                     fourrier_transform_image=None,
    #                                                     correlate=None, auto_correlation=None)

    psf_convolve = sndi.filters.convolve(psf_image, kernel)
    #psf_convolve /= np.sum(psf_convolve)

    return psf_convolve[min_y:max_y, min_x:max_x] 
def iterate_on_background(data,psf_fit):

    weight1 = (0.5 + np.abs(data + 0.25) ** 0.5)
    weight2 = (-0.5 + np.abs(data + 0.25) ** 0.5)
    weight = (weight1 ** 2 + weight2 ** 2) ** 0.5
    mask = data != True


    back_old=10
    back = 0

    while np.abs(back_old-back)>1:

        back_old = back
        true = data - back
        flux = np.sum(true[mask] * psf_fit[mask] * 1 / weight[mask] ** 2) / np.sum(
            psf_fit[mask] ** 2 / weight[mask] ** 2)
        back = np.median(data - flux * psf_fit)
        back_err = np.abs(back) ** 0.5
        model = flux * psf_fit + back
        residus = data - model
        SNR = (flux ** 2 * np.sum(psf_fit[mask] ** 2 / (np.abs(back) + np.abs(flux * psf_fit[mask])))) ** 0.5
        flux_err = ((np.abs(flux) / SNR) ** 2 + np.sum(np.abs(residus)[mask])) ** 0.5

    return flux, flux_err, back, back_err
def sigma_clip_the_flux(data,psf_fit):

    outliers = 1
    model = data
    mask = data != True
    back = 0
    weight1 = (0.5 + np.abs(data + 0.25) ** 0.5)
    weight2 = (-0.5 + np.abs(data + 0.25) ** 0.5)
    weight = (weight1 ** 2 + weight2 ** 2) ** 0.5

    while outliers !=0:


        #intensities, cov = np.polyfit(psf_fit[mask], data[mask], 1, w=1 / weight[mask], cov=True)
        #(flux, flux_err) = (intensities[0], cov[0][0] ** 0.5)

        #(back, back_err) = (intensities[1], cov[1][1] ** 0.5)
        true = data-back
        flux = np.sum(true[mask] * psf_fit[mask] * 1 / weight[mask] ** 2) / np.sum(psf_fit[mask] ** 2 / weight[mask] ** 2)
        back = np.median(data-flux*psf_fit)
        back_err = np.abs(back)**0.5
        model = flux*psf_fit+back
        residus = data-model
        SNR = (flux ** 2 * np.sum(psf_fit[mask] ** 2 / (np.abs(back) + np.abs(flux * psf_fit[mask])))) ** 0.5
        flux_err = ((np.abs(flux) / SNR)**2+np.sum(np.abs(residus)[mask]))**0.5

        max_deviant = np.max(np.abs(data[mask]-model[mask])/weight[mask])

        if max_deviant>5:
            index = np.where(np.abs(data-model)/weight==max_deviant)
            mask[index] = False
            outliers = 1
        else:
            outliers = 0
        print(outliers)
    #import pdb;
    #pdb.set_trace()
    return flux,flux_err,back,back_err

def quick_polyfit(params,data,weight,psf_fit):


    model = params[0]*psf_fit+params[1]

    return ((data-model)/weight).ravel()




def run_psf_photometry_on_difference_image(setup, reduction_metadata, log, ref_star_catalog,
                                           difference_image, psf_model, kernel, kernel_error, ref_exposure_time,image_id):
    """Function to perform PSF fitting photometry on all stars for a single difference image.
    
    :param SetUp object setup: Essential reduction parameters
    :param MetaData reduction_metadata: pipeline metadata for this dataset
    :param logging log: Open reduction log object
    :param array ref_star_catalog: catalog of objects detected in the image
    :param array_like difference_image: the array of data on which performs photometry
    :param array_like psf_model: PSF to be fitted to each star
   
    
    Returns:
    
    :param array ref_star_catalog: catalog of objects detected in the image
    """
    # import matplotlib.pyplot as plt
    # plt.imshow(np.log10(difference_image))
    # plt.show()

    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    half_psf = int(psf_size)

    size_stamp = int(2 * half_psf) + 1
    if size_stamp % 2 == 0:
        size_stamp += 1

    Y_data, X_data = np.indices((size_stamp, size_stamp))

    list_image_id = []
    list_star_id = []

    list_ref_mag = []
    list_ref_mag_error = []
    list_ref_flux = []
    list_ref_flux_error = []

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

    phot_scale_factor = (np.sum(kernel))
    error_phot_scale_factor = (phot_scale_factor*0.1)


    control_size = 50
    control_count = 0
    #psf_parameters = psf_model.get_parameters()

    #radius = half_psf + 1

    positions = ref_star_catalog[:, [1, 2]]
    pixscale = reduction_metadata.reduction_parameters[1]['PIX_SCALE'].data[0]
    radius = np.max((reduction_metadata.images_stats[1]['FWHM_X'][image_id],reduction_metadata.images_stats[1]['FWHM_Y'][image_id]))
    radius *= 1.5852*2
    apertures = CircularAperture(positions, r=radius)
    error = calc_total_error(difference_image, np.std(difference_image.ravel()), 1)
    phot_table = aperture_photometry(difference_image, apertures, method='subpixel',
                                     error=error)

    #import pdb;
    #pdb.set_trace()
    for j in range(0, len(ref_star_catalog), 1)[:]:
     
        #j = 59716
        list_image_id.append(0)
        list_star_id.append(ref_star_catalog[j, 0])

        ref_flux = ref_star_catalog[j, 5]
        error_ref_flux = ref_star_catalog[j, 6]

        list_ref_mag.append(ref_star_catalog[j, 5])
        list_ref_mag_error.append(ref_star_catalog[j, 6])
        list_ref_flux.append(ref_flux)
        list_ref_flux_error.append(error_ref_flux)

        #### comented for swithcing from psf fitting to aperture photometry

        #xstar = ref_star_catalog[j, 1]
        #ystar = ref_star_catalog[j, 2]

        #X_grid = X_data + (int(np.round(xstar)) - half_psf)
        #Y_grid = Y_data + (int(np.round(ystar)) - half_psf)

        # logs.ifverbose(log, setup, ' -> Star ' + str(j) + ' at position (' + \
        #               str(xstar) + ', ' + str(ystar) + ')')

        #psf_parameters[1] = xstar
        #psf_parameters[2] = ystar

        #psf_image = psf_model.psf_model(X_grid, Y_grid, psf_parameters)


        ##psf_convolve = sndi.filters.convolve(psf_image, kernel,mode='constant')

        #try:

        #    max_x = int(np.min([difference_image.shape[0], np.max(X_data + (int(np.round(xstar)) - half_psf)) + 1]))
        #    min_x = int(np.max([0, np.min(X_data + (int(np.round(xstar)) - half_psf))]))
        #    max_y = int(np.min([difference_image.shape[1], np.max(Y_data + (int(np.round(ystar)) - half_psf)) + 1]))
        #    min_y = int(np.max([0, np.min(Y_data + (int(np.round(ystar)) - half_psf))]))

        #    data = difference_image[min_y:max_y, min_x:max_x]

        #    max_x = int(max_x - (int(np.round(xstar)) - half_psf))
        #    min_x = int(min_x - (int(np.round(xstar)) - half_psf))
        #    max_y = int(max_y - (int(np.round(ystar)) - half_psf))
        #    min_y = int(min_y - (int(np.round(ystar)) - half_psf))

        #    PSF = psf_convolve[min_y:max_y, min_x:max_x]


        #    psf_fit = PSF

        #    good_fit = True


        #except:
        #    good_fit = False

        good_fit = True

        if good_fit == True:

            # logs.ifverbose(log, setup, ' -> Star ' + str(j) +
            #              ' subtracted from the residuals')

         

            #weight1 = (0.5 + np.abs(data + 0.25) ** 0.5)
            #weight2 = (-0.5 + np.abs(data + 0.25) ** 0.5)
            #weight = (weight1 ** 2 + weight2 ** 2)
            #poids = weight**0.5



            #intensities, cov = np.polyfit(psf_fit.ravel(), data.ravel(), 1, w=1/poids.ravel(), cov=True)

            #(flux,flux_err) = (intensities[0], cov[0][0] ** 0.5)
            #(back, back_err) = (intensities[1], cov[1][1] ** 0.5)

            #flux = np.sum(flux*psf_image)

            #flux2 = np.sum(data.ravel())



            #SNR = flux2/len(data)**2
            #flux = flux2/phot_scale_factor
            #flux_err = flux/SNR


            #residus = data - psf_fit * intensities[0]-back


            #flux_err = (flux_err**2+np.mean(residus**2))**0.5


            flux = phot_table[j][3]/phot_scale_factor
            flux_err = phot_table[j][4]


            flux_tot = ref_flux*ref_exposure_time - flux
        
            flux_err_tot = (error_ref_flux ** 2*ref_exposure_time + flux_err**2/phot_scale_factor**2) ** 0.5

            list_delta_flux.append(flux)
            list_delta_flux_error.append(flux_err)

            (mag, mag_err,ftmp,fetemp) = convert_flux_to_mag(flux_tot, flux_err_tot,ref_exposure_time)
       

            list_mag.append(mag)
            list_mag_error.append(mag_err)
            list_phot_scale_factor.append(phot_scale_factor)
            list_phot_scale_factor_error.append(error_phot_scale_factor)
            #list_background.append(back)
            #list_background_error.append(back_err)

            #list_align_x.append(xstar)
            #list_align_y.append(ystar)

            list_background.append(0)
            list_background_error.append(0)

            list_align_x.append(positions[j][0])
            list_align_y.append(positions[j][1])



        else:

            logs.ifverbose(log, setup, ' -> Star ' + str(j) +
                           ' No photometry possible from poor fit')

            list_delta_flux.append(-10 ** 30)
            list_delta_flux_error.append(-10 ** 30)
            list_mag.append(-10 ** 30)
            list_mag_error.append(-10 ** 30)
            list_phot_scale_factor.append(np.sum(kernel))
            list_phot_scale_factor_error.append(-10 ** 30)
            list_background.append(-10 ** 30)
            list_background_error.append(-10 ** 30)

            list_align_x.append(-10 ** 30)
            list_align_y.append(-10 ** 30)
    # import pdb; pdb.set_trace()

    difference_image_photometry = [list_image_id, list_star_id, list_ref_mag, list_ref_mag_error, list_ref_flux,
                                   list_ref_flux_error, list_delta_flux, list_delta_flux_error, list_mag,
                                   list_mag_error,
                                   list_phot_scale_factor, list_phot_scale_factor_error, list_background,
                                   list_background_error, list_align_x, list_align_y]

    log.info('Completed photometry on difference image')

    # return  difference_image_photometry, control_zone
    return np.array(difference_image_photometry).T, 1


	
def convert_flux_to_mag(flux, flux_err, exp_time=None):
    """Function to convert the flux of a star from its fitted PSF model 
    and its uncertainty onto the magnitude scale.
    
    :param float flux: Total star flux
    :param float flux_err: Uncertainty in star flux
    
    Returns:
    
    :param float mag: Measured star magnitude
    :param float flux_mag: Uncertainty in measured magnitude
    :param float flux: Total flux, scaled by the exposure time if given
    :param float flux_err: Uncertainty on total flux, scaled by the exposure 
                            time, if given
    """

    def flux2mag(ZP, flux):

        return ZP - 2.5 * np.log10(flux)

    
    if exp_time != None:
        
        frac_err = flux_err / flux
        
        flux = flux / exp_time
        
        flux_err = flux * frac_err
        
    if flux < 0.0 or flux_err < 0.0:

        mag = 0.0
        mag_err = 0.0

    else:

        ZP = 25.0

        mag = flux2mag(ZP, flux)


        mag_err = (2.5 / np.log(10.0)) * flux_err / flux

    return mag, mag_err, flux, flux_err

def convert_mag_to_flux(mag, mag_err):
    """Function to convert the flux of a star from its fitted PSF model 
    and its uncertainty onto the magnitude scale.
    
    :param float flux: Total star flux
    :param float flux_err: Uncertainty in star flux
    
    Returns:
    
    :param float mag: Measured star magnitude
    :param float flux_mag: Uncertainty in measured magnitude
    :param float flux: Total flux, scaled by the exposure time if given
    :param float flux_err: Uncertainty on total flux, scaled by the exposure 
                            time, if given
    """

    ZP = 25.0
    
    flux = 10**( (mag - ZP) / -2.5 )
    
    ferr = mag_err/(2.5*np.log10(np.e)) * flux
    
    return flux, ferr
    

def plot_ref_mag_errors(setup,ref_star_catalog):
    """Function to output a diagnostic plot of the fitted PSF magnitudes
    against photometric error"""

    ref_path = setup.red_dir + '/ref/'
    file_path = os.path.join(ref_path, 'ref_image_phot_errors.png')

    fig = plt.figure(1)

    idx = np.where(ref_star_catalog[:, 7] > 0.0)

    plt.plot(ref_star_catalog[idx, 7], ref_star_catalog[idx, 8], 'k.')

    plt.yscale('log')

    plt.xlabel('Instrumental magnitude')

    plt.ylabel('Photometric uncertainty [mag]')

    [xmin, xmax, ymin, ymax] = plt.axis()

    plt.axis([xmax, xmin, ymin, ymax])

    plt.grid()
    
    plt.savefig(file_path)

    plt.close(1)
    
