######################################################################
#
# stage6.py - Sixth stage of the pipeline. Subtract and make photometry
# on residuals

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

import numpy as np
import os
import sys
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from scipy.ndimage.interpolation import shift
from sky_background import mask_saturated_pixels, generate_sky_model
from sky_background import fit_sky_background, generate_sky_model_image



import config_utils

import metadata
import logs
import convolution
#import db.astropy_interface as db_ai
#import db.phot_db as db_phot
import sky_background
import psf
import photometry
import phot_db


def run_stage6(setup):
    """Main driver function to run stage 6: image substraction and photometry.
    This stage align the images to the reference frame!
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata], the stage4 status, the report, the metadata file
    :rtype: array_like

    """

    stage6_version = 'stage6 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage6', version=stage6_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    # find the images needed to treat
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=6, rerun_all=None, log=log)

    # find the starlist
    starlist =  reduction_metadata.star_catalog[1]	

    max_x = np.max(starlist['x_pixel'].data)
    max_y = np.max(starlist['y_pixel'].data)
    mask  = (starlist['psf_star'].data == 1) & (starlist['x_pixel'].data<max_x-25)  & (starlist['x_pixel'].data>25) & (starlist['y_pixel'].data<max_y-25)  & (starlist['y_pixel'].data>25)

    control_stars = starlist[mask][:10]
    star_coordinates = np.c_[control_stars['star_index'].data,
                             control_stars['x_pixel'].data,
                             control_stars['y_pixel'].data]

    for index,key in enumerate(starlist.columns.keys()):

	if index != 0:

	
	    ref_star_catalog = np.c_[ref_star_catalog,starlist[key].data]

	else:
		
	    ref_star_catalog = starlist[key].data



    psf_model = fits.open(reduction_metadata.data_architecture[1]['REF_PATH'].data[0]+'/psf_model.fits')

    psf_type = psf_model[0].header['PSFTYPE']
    psf_parameters = [0, psf_model[0].header['Y_CENTER'],
                      psf_model[0].header['X_CENTER'],
                      psf_model[0].header['GAMMA'],
                      psf_model[0].header['ALPHA']]  	
    
 
    sky_model = sky_background.model_sky_background(setup,
                                        reduction_metadata,log,ref_star_catalog)


    psf_model = psf.get_psf_object( psf_type )
    psf_model.update_psf_parameters( psf_parameters)

    ind = ((starlist['x_pixel']-150)**2<1) & ((starlist['y_pixel']-150)**2<1)
    print np.argmin(((starlist['x_pixel']-150)**2) + ((starlist['y_pixel']-150)**2))
    if len(new_images) > 0:

        # find the reference image
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            reference_image_directory = reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
            reference_image,date = open_an_image(setup, reference_image_directory, reference_image_name, image_index=0,
                                            log=None)

            row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == reference_image_name)[0][0]
            ref_exposure_time = float(reduction_metadata.headers_summary[1][row_index]['EXPKEY'])
		
	    print ref_exposure_time
   
            logs.ifverbose(log, setup,
                           'I found the reference frame:' + reference_image_name)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find any reference image! Aboard stage6')

            status = 'KO'
            report = 'No reference frame found!'

            return status, report

        # find the kernels directory
        try:

            kernels_directory = reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'].data[0]+'kernel/'

            logs.ifverbose(log, setup,
                           'I found the kernels directory:' + kernels_directory)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find the kernels directory! Aboard stage6')

            status = 'KO'
            report = 'No kernels directory found!'

            return status, report
       
        # turn on the db
        try:

            kernels_directory = reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'].data[0]+'kernel/'

            logs.ifverbose(log, setup,
                           'I found the kernels directory:' + kernels_directory)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find the kernels directory! Aboard stage6')

            status = 'KO'
            report = 'No kernels directory found!'

            return status, report

        data = []
        diffim_directory = reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'].data[0]+'diffim/'
        images_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'].data[0]
        phot = np.zeros((145,793,16))
	time = []
        conn = phot_db.get_connection((setup.red_dir+ 'pyDANDIA_phot.db'))
        for idx,new_image in enumerate(new_images):
    	 
            log.info('Starting difference photometry of '+new_image)
            target_image,date = open_an_image(setup, images_directory, new_image, image_index=0, log=None)

	    row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
            target_exposure_time = float(reduction_metadata.headers_summary[1][row_index]['EXPKEY'])

            exp_time_ratio = 1/(target_exposure_time/ref_exposure_time )
           
            kernel_image,kernel_error,kernel_bkg = find_the_associated_kernel(setup, kernels_directory, new_image)
	    #import pdb; pdb.set_trace()
            difference_image = image_substraction(setup, reduction_metadata,reference_image, kernel_image, new_image)-kernel_bkg
	    #difference_image = image_substraction2(setup, diffim_directory, new_image)

	    time.append(date)

            save_control_stars_of_the_difference_image(setup, new_image, difference_image, star_coordinates)

            photometric_table, control_zone = photometry_on_the_difference_image(setup, reduction_metadata, log,ref_star_catalog,difference_image,  psf_model, sky_model, kernel_image,kernel_error,time,exp_time_ratio)
	    
	    phot[idx,:,:] = photometric_table

            #save_control_zone_of_residuals(setup, new_image, control_zone)	

            #ingest_photometric_table_in_db(setup, photometric_table)

 
    import pdb; pdb.set_trace()
    import matplotlib.pyplot as plt 
    ind = ((starlist['x_pixel']-150)**2<1) & ((starlist['y_pixel']-150)**2<1)
    plt.errorbar(time,phot[:,ind,8],fmt='.k')
    
    
    ind=177
    plt.errorbar(time,phot[:,ind,8],fmt='.r')
    ind = np.random.randint(0,600)
    ind=722
    print ind
    plt.errorbar(time,phot[:,ind,8],fmt='.g')
    plt.gca().invert_yaxis()
    plt.show()
    import pdb; pdb.set_trace()
    return status, report

def background_subtract(setup, image, max_adu):

    masked_image = mask_saturated_pixels(setup, image, max_adu,log = None)
    sky_params = { 'background_type': 'gradient', 
          'nx': image.shape[1], 'ny': image.shape[0],
          'a0': 0.0, 'a1': 0.0, 'a2': 0.0 }
    sky_model = generate_sky_model(sky_params) 
    sky_fit = fit_sky_background(masked_image,sky_model,'gradient',log=None)
    sky_params['a0'] = sky_fit[0][0]
    sky_params['a1'] = sky_fit[0][1]
    sky_params['a2'] = sky_fit[0][2]
    #sky_model = generate_sky_model(sky_params)
    sky_model_image = generate_sky_model_image(sky_params)
    
    return image - sky_model_image


def open_an_image(setup, image_directory, image_name,
                  image_index=0, log=None):
    '''
    Simply open an image using astropy.io.fits

    :param object reduction_metadata: the metadata object
    :param string image_directory: the image name
    :param string image_name: the image name
    :param string image_index: the image index of the astropy fits object

    :param boolean verbose: switch to True to have more informations

    :return: the opened image
    :rtype: astropy.image object
    '''
    image_directory_path = image_directory

    logs.ifverbose(log, setup,
                   'Attempting to open image ' + os.path.join(image_directory_path, image_name))
    #import pdb; pdb.set_trace()
    try:

        image_data = fits.open(os.path.join(image_directory_path, image_name),
                               mmap=True)
        image_data = image_data[image_index]
        try:
		date =  image_data.header['MJD-OBS']
	except :
		date = 0

        logs.ifverbose(log, setup, image_name + ' open : OK')

        return image_data.data,date

    except:
        logs.ifverbose(log, setup, image_name + ' open : not OK!')

        return None

def save_control_zone_of_residuals(setup, image_name, control_zone): 
    '''
    Save selected stars for difference image control

    :param object reduction_metadata: the metadata object
    :param str image_name: the name of the image
    :param array_likecontrol_zone: the residuals stamps

    '''

    control_images_directory = setup.red_dir+'/res_images/'
    os.makedirs(control_images_directory, exist_ok=True)

    control_size = 50

    image_name.replace('.fits','.res')

    hdu = fits.PrimaryHDU(control_zone)
    hdu.writeto(control_images_directory+image_name, overwrite=True)


def save_control_stars_of_the_difference_image(setup, image_name, difference_image, star_coordinates): 
    '''
    Save selected stars for difference image control

    :param object reduction_metadata: the metadata object
    :param str image_name: the name of the image
    :param array_like difference_image: the reference image data
    :param array_like stars_coordinates: the position of control stars
    '''

    control_images_directory = setup.red_dir+'diffim/'
    try:
    	os.makedirs(control_images_directory)
    except:
	pass

    control_size = 50

    
    for star in star_coordinates :

        ind_i = int(np.round(star[1]))
        ind_j = int(np.round(star[2]))

        stamp = difference_image[ind_i-control_size/2:ind_i+control_size/2,
		          ind_j-control_size/2:ind_j+control_size/2]

        try :

             control_zone = np.c_[control_zone, stamp]

        except:

             control_zone = stamp

    image_name = image_name.replace('.fits','.diff')

    hdu = fits.PrimaryHDU(difference_image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(control_images_directory+image_name, overwrite=True)


def image_substraction2(setup, diffim_directory, image_name, log=None):
    '''
    Subtract the image from model, i.e residuals = image-convolution(reference_image,kernel)

    :param object reduction_metadata: the metadata object
    :param array_like reference_image_data: the reference image data
    :param array_like kernel_data: the kernel image data
    :param array_like image_data: the image data

    :param boolean verbose: switch to True to have more informations

    :return: the difference image
    :rtype: array_like
    '''
    #import pdb; pdb.set_trace()
    diffim = 'diff_'+image_name
	 
    diffim,date = open_an_image(setup, diffim_directory, diffim,
                           image_index=0, log=None)

    return diffim
     

def image_substraction(setup, reduction_metadata, reference_image_data, kernel_data, image_name, log=None):
    '''
    Subtract the image from model, i.e residuals = image-convolution(reference_image,kernel)

    :param object reduction_metadata: the metadata object
    :param array_like reference_image_data: the reference image data
    :param array_like kernel_data: the kernel image data
    :param array_like image_data: the image data

    :param boolean verbose: switch to True to have more informations

    :return: the difference image
    :rtype: array_like
    '''

    image_data,date = open_an_image(setup, './data/', image_name, image_index=0, log=None)
    row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_name)[0][0]

    kernel_size = kernel_data.shape[0]

    background_ref = background_subtract(setup, reference_image_data, np.median(reference_image_data))
    #background_ref = reference_image_data
    ref_extended = np.zeros((np.shape( background_ref)[0] + 2 * kernel_size,
                             np.shape( background_ref)[1] + 2 * kernel_size))
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(background_ref, float)

	
    model = convolution.convolve_image_with_a_psf(ref_extended, kernel_data)

    model = model[kernel_size:-kernel_size,kernel_size:-kernel_size]

    background_image = background_subtract(setup, image_data, np.median(image_data))
    #background_image = image_data
    xshift, yshift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'],-reduction_metadata.images_stats[1][row_index]['SHIFT_Y'] 

    

    image_shifted = shift(background_image, (-yshift,-xshift), cval=0.) 

    

    difference_image = image_shifted - model

    return difference_image

def create_astropy_table_for_db(photometry_table):

	names = ['exposure_id',
		'star_id',
		'reference_flux',
		'reference_mag',
		'reference_flux_err',
		'reference_mag_err',
		'diff_flux',
		'diff_flux_err',
		'magnitude',
		'magnitude_err',
		'phot_scale_factor',
		'phot_scale_factor_err',
		'local_background',
		'local_background_err',
		'residual_x',
		'residual_y']       
 
        types = (' np.int64', ' np.int64','np.floatf64', 'np.floatf64','np.floatf64','np.floatf64','np.floatf64','np.floatf64','np.floatf64','np.floatf64','np.floatf64',
'np.floatf64','np.floatf64','np.floatf64','np.floatf64','np.floatf64')
	astropy_table = Table(photometry_table,names=names,dtype=types)
	return astropy_table
def find_the_associated_kernel(setup, kernels_directory, image_name):
    '''
    Find the appropriate kernel associated to an image
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''
    #import pdb; pdb.set_trace()
    kernel_name = 'kernel_'+image_name
    kernel_err = 'kernel_err_'+image_name
    
    kernel = fits.open( kernels_directory+kernel_name )
    kernel_error = fits.open( kernels_directory+kernel_err )
    #kernel,date = open_an_image(setup, kernels_directory, kernel_name,
    #                       image_index=0, log=None)
    #kernel_error,date = open_an_image(setup, kernels_directory, kernel_err,
    #                       image_index=0, log=None)
    bkgd = +kernel[0].header['KERBKG']

    kernel = kernel[0].data

    return kernel,kernel_error[0].data,bkgd

def photometry_on_the_difference_image(setup, reduction_metadata, log, star_catalog,difference_image, psf_model, sky_model, kernel,kernel_error,jd,exp_time_ratio):
    '''
    Find the appropriate kernel associated to an image
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''
    #import pdb; pdb.set_trace()

    differential_photometry = photometry.run_psf_photometry_on_difference_image(setup,reduction_metadata,log,star_catalog,
                       								difference_image,psf_model,kernel,kernel_error,jd,exp_time_ratio)
    
    #column_names = ('Exposure_id','Star_id','Ref_mag','Ref_mag_err','Ref_flux','Ref_flux_err','Delta_flux','Delta_flux_err','Mag','Mag_err',
     #               'Phot_scale_factor','Phot_scale_factor_err','Back','Back_err','delta_x','delta_y')
   
    #table = Table(differential_photometry, names = column_names)


    #return table
    return differential_photometry
def ingest_reference_in_db(setup, reference):

	conn = db_phot.get_connection(dsn=setup.red_dir)
	
	db_ai.load_astropy_table(conn, 'phot', photometric_table)

	
def ingest_exposure_in_db(setup, photometric_table):

	conn = db_phot.get_connection(dsn=setup.red_dir)
	
	db_ai.load_astropy_table(conn, 'phot', photometric_table)

def ingest_photometric_table_in_db(setup, photometric_table):

	conn = db_phot.get_connection(dsn=setup.red_dir)
	
	db_ai.load_astropy_table(conn, 'phot', photometric_table)
