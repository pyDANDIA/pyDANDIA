"""
Created on Wed Oct 18 15:42:26 2017

@author: rstreet
"""

import os
import sys
import numpy as np
from pyDANDIA import logs
from pyDANDIA import  metadata
import matplotlib.pyplot as plt
from astropy.io import fits
from pyDANDIA import  starfind
from pyDANDIA import  psf
from pyDANDIA import  convolution
from scipy.odr import *


def linear_func(p,x):
    a, b = p
    return a*x+b



def run_psf_photometry(setup,reduction_metadata,log,ref_star_catalog,
                       image_path,psf_model,sky_model,centroiding=True,
                       diagnostics=True, psf_size=None):
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
    
    if psf_size == None:
        psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]

    half_psf = int(float(psf_size)/2.0)
    
    exp_time = extract_exptime(reduction_metadata,
                               os.path.basename(image_path))
                               
    logs.ifverbose(log,setup,'Applying '+psf_model.psf_type()+\
                    ' PSF of diameter='+str(psf_size))
    logs.ifverbose(log,setup,'Scaling fluxes by exposure time '+str(exp_time)+'s')
    
    Y_data, X_data = np.indices((int(psf_size),int(psf_size)))
    
    for j in range(0,len(ref_star_catalog),1):
        
        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]
        
        X_grid = X_data + (int(xstar) - half_psf)
        Y_grid = Y_data + (int(ystar) - half_psf)
        
        logs.ifverbose(log,setup,' -> Star '+str(j)+' at position ('+\
        str(xstar)+', '+str(ystar)+')')
        
        (fitted_model,good_fit) = psf.fit_star_existing_model(setup, residuals, 
                                               xstar, ystar, psf_size, 
                                               psf_model, sky_model, 
                                               centroiding=centroiding,
                                               diagnostics=diagnostics)
        
        logs.ifverbose(log, setup,' -> Star '+str(j)+
        ' fitted model parameters = '+repr(fitted_model.get_parameters())+
        ' good fit? '+repr(good_fit))
        
        if good_fit == True:
            
            sub_psf_model = psf.get_psf_object('Moffat2D')
            
            pars = fitted_model.get_parameters()
            pars[1] = half_psf + (ystar-int(ystar))
            pars[2] = half_psf + (xstar-int(xstar))
            
            sub_psf_model.update_psf_parameters(pars)
            
            (res_image,corners) = psf.subtract_psf_from_image(residuals,sub_psf_model,
                                                    xstar,ystar,
                                                    psf_size, psf_size)
            
            residuals[corners[2]:corners[3],corners[0]:corners[1]] = res_image[corners[2]:corners[3],corners[0]:corners[1]]
            
            hdu = fits.PrimaryHDU(res_image[corners[2]:corners[3],corners[0]:corners[1]])
            hdulist = fits.HDUList([hdu])
            file_path = os.path.join(setup.red_dir,'ref',\
            'psf_star_stamp_residuals'+str(round(xstar,0))+'_'+str(round(ystar,0))+'.fits')
            hdulist.writeto(file_path,overwrite=True)
        
            logs.ifverbose(log, setup,' -> Star '+str(j)+
            ' subtracted PSF from the residuals')
            
            (flux, flux_err) = fitted_model.calc_flux(Y_grid, X_grid)
            
            (mag, mag_err, flux_scaled, flux_err_scaled) = convert_flux_to_mag(flux, flux_err, exp_time=exp_time)
            
            ref_star_catalog[j,5] = flux_scaled
            ref_star_catalog[j,6] = flux_err_scaled
            ref_star_catalog[j,7] = mag
            ref_star_catalog[j,8] = mag_err
            
            logs.ifverbose(log,setup,' -> Star '+str(j)+
            ' flux='+str(flux)+' +/- '+str(flux_err)+' ADU, '
            'mag='+str(mag)+' +/- '+str(mag_err)+' mag')
    
        else:
            
            logs.ifverbose(log,setup,' -> Star '+str(j)+
            ' No photometry possible from poor PSF fit')

    res_image_path = os.path.join(setup.red_dir,'ref',os.path.basename(image_path).replace('.fits','_res.fits'))
    
    hdu = fits.PrimaryHDU(residuals)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(res_image_path, overwrite=True)
    
    logs.ifverbose(log, setup, 'Output residuals image '+res_image_path)

    plot_ref_mag_errors(setup,ref_star_catalog)

    log.info('Completed photometry')
    
    return ref_star_catalog
    
def run_psf_photometry_on_difference_image(setup,reduction_metadata,log,ref_star_catalog,
                       difference_image,psf_model,kernel,kernel_error, ref_exposure_time):
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
    #import matplotlib.pyplot as plt
    #plt.imshow(np.log10(difference_image))
    #plt.show()

    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    half_psf = psf_size/2

    size_stamp = int(psf_size)+1
    if size_stamp%2 == 0 :
       size_stamp += 1

    Y_data, X_data = np.indices((size_stamp+1,size_stamp+1))

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
     

    phot_scale_factor = np.sum(kernel)
    error_phot_scale_factor = (np.sum(kernel_error))**0.5

    #kernel /=phot_scale_factor

    control_size = 50
    control_count = 0
    psf_parameters = psf_model.get_parameters()
    psf_parameters[0] = 1
    radius = psf_size/2

    for j in range(0,len(ref_star_catalog),1):
    
        list_image_id.append(0)
        list_star_id.append(ref_star_catalog[j,0])

        ref_flux = ref_star_catalog[j,5]
        error_ref_flux = ref_star_catalog[j,6]

        list_ref_mag.append(ref_star_catalog[j,5])
        list_ref_mag_error.append(ref_star_catalog[j,6])
        list_ref_flux.append(ref_flux)
        list_ref_flux_error.append(error_ref_flux)


        xstar = ref_star_catalog[j,1]
        ystar = ref_star_catalog[j,2]

        X_grid = X_data + (int(np.round(xstar)) - half_psf)
        Y_grid = Y_data + (int(np.round(ystar)) - half_psf)

        logs.ifverbose(log,setup,' -> Star '+str(j)+' at position ('+\
        str(xstar)+', '+str(ystar)+')')

        psf_parameters[1] = xstar
        psf_parameters[2] = ystar

        psf_image = psf_model.psf_model(X_grid, Y_grid, psf_parameters)
        
        psf_convolve = convolution.convolve_image_with_a_psf(psf_image, kernel, fourrier_transform_psf=None, fourrier_transform_image=None,
                        correlate=None, auto_correlation=None)
        try:

          max_x = int(np.min([difference_image.shape[0], np.max(X_data + (int(np.round(xstar)) - half_psf))]))
          min_x = int(np.max([0, np.min(X_data + (int(np.round(xstar)) - half_psf))]))
          max_y = int(np.min([difference_image.shape[1], np.max(Y_data + (int(np.round(ystar)) - half_psf))]))
          min_y = int(np.max([0, np.min(Y_data + (int(np.round(ystar)) - half_psf))]))

          data = difference_image[min_y:max_y,min_x:max_x]
          
          max_x = int(max_x- (int(np.round(xstar)) - half_psf))
          min_x = int(min_x- (int(np.round(xstar)) - half_psf))
          max_y = int(max_y- (int(np.round(ystar)) - half_psf))
          min_y = int(min_y- (int(np.round(ystar)) - half_psf))

          psf_fit = psf_convolve[min_y:max_y,min_x:max_x]
          psf_fit /= np.max(psf_fit)
         
          residuals = np.copy(data)
          
        except:
             import pdb; pdb.set_trace()
             
        good_fit = True
        if good_fit == True:

           

          logs.ifverbose(log, setup,' -> Star '+str(j)+
          ' subtracted from the residuals')



          center = (psf_fit.shape)
          xx,yy=np.indices(psf_fit.shape)
          mask = ((xx-center[0]/2)**2+(yy-center[1]/2)**2)<radius**2
          mask2 = ((xx-center[0]/2)**2+(yy-center[1]/2)**2)<2*radius**2
	  
          weight1= 0.5+np.abs(data+0.25)**0.5
          weight2= -0.5+np.abs(data+0.25)**0.5
          weight = (weight1**2+weight2**2)**0.5
          
         
          rejected_points = 0
        
          intensities,cov = np.polyfit(psf_fit[mask2],data[mask2],1,w=1/weight[mask2],cov=True)
       	   
          (flux, flux_err) =  (intensities[0],(cov[0][0])**0.5)




         
          (back,back_err) = (intensities[1],cov[1][1]**0.5)

          flux_tot = ref_flux-flux/phot_scale_factor
          flux_err_tot = (error_ref_flux**2+(flux_err/phot_scale_factor)**2)**0.5
          
          
          list_delta_flux.append(flux)
          list_delta_flux_error.append(flux_err)        


          (mag, mag_err,flux_scaled,flux_err_scaled) = convert_flux_to_mag(flux_tot, flux_err_tot, exp_time=ref_exposure_time)


          list_mag.append(mag)
          list_mag_error.append(mag_err)
          list_phot_scale_factor.append(phot_scale_factor)
          list_phot_scale_factor_error.append(error_phot_scale_factor)
          list_background.append(back)
          list_background_error.append(back_err)

          list_align_x.append(0)
          list_align_y.append(0)



        else:

          logs.ifverbose(log,setup,' -> Star '+str(j)+
          ' No photometry possible from poor fit')

          list_delta_flux.append(-10**30)
          list_delta_flux_error.append(-10**30)
          list_mag.append(-10**30)
          list_mag_error.append(-10**30)
          list_phot_scale_factor.append(np.sum(kernel))
          list_phot_scale_factor_error.append(0)
          list_background.append(0)
          list_background_error.append(0)

          list_align_x.append(0)
          list_align_y.append(0)
    #import pdb; pdb.set_trace()
 
    difference_image_photometry = [list_image_id,list_star_id,list_ref_mag,list_ref_mag_error,list_ref_flux,
                                            list_ref_flux_error,list_delta_flux,list_delta_flux_error,list_mag,list_mag_error,
                                            list_phot_scale_factor, list_phot_scale_factor_error, list_background,
                                            list_background_error, list_align_x, list_align_y ]


    log.info('Completed photometry on difference image')
    
    #return  difference_image_photometry, control_zone
    return  np.array(difference_image_photometry).T     ,1

def convert_flux_to_mag(flux, flux_err, exp_time=None):
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
        
        mag_err = (2.5/np.log(10.0))*flux_err/flux
        
    return mag, mag_err, flux, flux_err
    
def plot_ref_mag_errors(setup,ref_star_catalog):
    """Function to output a diagnostic plot of the fitted PSF magnitudes
    against photometric error"""

    ref_path =  setup.red_dir+'/ref/'
    file_path = os.path.join(ref_path,'ref_image_phot_errors.png')
    
    fig = plt.figure(1)
    
    idx = np.where(ref_star_catalog[:,7] > 0.0)
    
    plt.plot(ref_star_catalog[idx,7], ref_star_catalog[idx,8],'k.')

    plt.yscale('log')    
    
    plt.xlabel('Instrumental magnitude')

    plt.ylabel('Photometric uncertainty [mag]')
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    plt.axis([xmax,xmin,ymin,ymax])
    
    plt.savefig(file_path)

    plt.close(1)
    
def extract_exptime(reduction_metadata,image_name):
    """Function to look up the exposure time for the indicated image from
    the headers_summary in the reduction metadata"""
    
    idx = list(reduction_metadata.headers_summary[1]['IMAGES']).index(image_name)
    
    exp_time = float(reduction_metadata.headers_summary[1]['EXPKEY'][idx])
    
    return exp_time