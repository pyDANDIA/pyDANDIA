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
import astropy.time
import dateutil.parser

from pyDANDIA.sky_background import mask_saturated_pixels, generate_sky_model
from pyDANDIA.sky_background import fit_sky_background, generate_sky_model_image

from pyDANDIA.subtract_subimages import subtract_images, subtract_subimage

from pyDANDIA import config_utils

from pyDANDIA import metadata
from pyDANDIA import logs
from pyDANDIA import convolution
#from pyDANDIA import astropy_interface as db_phot
from pyDANDIA import phot_db as db_phot
from pyDANDIA import sky_background
from pyDANDIA import psf
from pyDANDIA import photometry
from pyDANDIA import stage3_db_ingest
from pyDANDIA import hd5_utils

def run_stage6(setup):
    """Main driver function to run stage 6: image substraction and photometry.
    This stage align the images to the reference frame!
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata], the stage4 status, the report, the metadata file
    :rtype: array_like

    """

    stage6_version = 'stage6 v0.2'

    log = logs.start_stage_log(setup.red_dir, 'stage6', version=stage6_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    dataset_params = harvest_stage6_parameters(setup,reduction_metadata,stage6_version)

    # Setup the DB connection and record dataset and software parameters
    conn = db_phot.get_connection(dsn=setup.phot_db_path)
    conn.execute('pragma synchronous=OFF')

    sane = check_stage3_ingest_complete(conn,dataset_params)

    (facility_keys, software_keys, image_keys) = stage3_db_ingest.define_table_keys()

    db_phot.check_before_commit(conn, dataset_params, 'facilities', facility_keys, 'facility_code')
    db_phot.check_before_commit(conn, dataset_params, 'software', software_keys, 'version')

    query = 'SELECT facility_id FROM facilities WHERE facility_code ="'+dataset_params['facility_code']+'"'
    dataset_params['facility'] = db_phot.query_to_astropy_table(conn, query, args=())['facility_id'][0]

    query = 'SELECT code_id FROM software WHERE version ="'+dataset_params['version']+'"'
    dataset_params['software'] = db_phot.query_to_astropy_table(conn, query, args=())['code_id'][0]

    query = 'SELECT filter_id FROM filters WHERE filter_name ="'+dataset_params['filter_name']+'"'
    dataset_params['filter'] = db_phot.query_to_astropy_table(conn, query, args=())['filter_id'][0]

    # Measure the offset between the reference image for this dataset relative
    # to the primary reference for this field
    #(transform, matched_stars) = match_dataset_with_field_primary_reference(setup,conn,dataset_params,
    #                                                                       reduction_metadata,log)
    (transform, matched_stars) = load_matched_stars_from_metadata(reduction_metadata,log)
    print(matched_stars.summary())
    print(transform)
    import import pdb; pdb.set_trace()
    
    # find the images needed to treat
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=6, rerun_all=True, log=log)

    # find the starlist
    starlist = reduction_metadata.star_catalog[1]

    max_x = np.max(starlist['x'].data)
    max_y = np.max(starlist['y'].data)
    mask = (starlist['psf_star'].data == 1) & (starlist['x'].data < max_x - 25) & (
    starlist['x'].data > 25) & (starlist['y'].data < max_y - 25) & (starlist['y'].data > 25)

    control_stars = starlist[mask][:10]
    star_coordinates = np.c_[control_stars['index'].data,
                             control_stars['x'].data,
                             control_stars['y'].data]

    for index, key in enumerate(starlist.columns.keys()):

        if index != 0:

            ref_star_catalog = np.c_[ref_star_catalog, starlist[key].data]

        else:

            ref_star_catalog = starlist[key].data
    log.info('Established ref_star_catalog array')

    psf_model = fits.open(reduction_metadata.data_architecture[1]['REF_PATH'].data[0] + '/psf_model.fits')

    psf_type = psf_model[0].header['PSFTYPE']
    #import pdb;
    #pdb.set_trace()
    psf_parameters = [ psf_model[0].header['INTENSIT'], psf_model[0].header['Y_CENTER'],
                      psf_model[0].header['X_CENTER'],
                      psf_model[0].header['GAMMA'],
                      psf_model[0].header['ALPHA']]

    log.info('Established PSF parameters')

    psf_model = psf.get_psf_object(psf_type)
    psf_model.update_psf_parameters(psf_parameters)
    log.info('Built PSF model')

    ind = ((starlist['x'] - 150) ** 2 < 1) & ((starlist['y'] - 150) ** 2 < 1)

    time = []
    exposures_id = []
    photometric_table = []

    photometry_data = build_photometry_array(setup,len(all_images),len(starlist),log)

    log.info('Starting photometry of difference images')

    if len(new_images) > 0:

        # find the reference image
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            reference_image_directory = reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
            reference_image, date = open_an_image(setup, reference_image_directory, reference_image_name, log, image_index=0)

            ref_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            index_reference = np.where(ref_image_name == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]
            ref_exposure_time = float(reduction_metadata.headers_summary[1]['EXPKEY'].data[index_reference])

            reference_header = reduction_metadata.headers_summary[1][index_reference]

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

            kernels_directory = os.path.join(reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'].data[0],
                                             'kernel')

            logs.ifverbose(log, setup,
                           'I found the kernels directory:' + kernels_directory)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find the kernels directory! Aboard stage6')

            status = 'KO'
            report = 'No kernels directory found!'

            return status, report

        date = []

        diffim_directory = os.path.join(reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'].data[0], 'diffim')


        n_images = len(new_images)
        list_of_stamps = reduction_metadata.stamps[1]['PIXEL_INDEX'].tolist()

        for idx, new_image in enumerate(new_images[:]):
            log.info('Extracting parameters of image ' + new_image + ' for photometry ('+str(idx)+' of '+str(n_images)+')')
            index_image = np.where(new_image == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]
            image_header = reduction_metadata.headers_summary[1][index_image]

            ddate = reduction_metadata.headers_summary[1]['DATEKEY'][index_image]
            jd =  dateutil.parser.parse(ddate)
            time = astropy.time.Time(jd)
            date.append(time.jd)

            image_params = stage3_db_ingest.harvest_image_params(reduction_metadata,
                                                                 os.path.join(setup.red_dir,'data',new_image),
                                                                 dataset_params['ref_filename'])
            image_params['version'] = stage6_version
            image_params['facility'] = dataset_params['facility']
            image_params['filter'] = dataset_params['filter']

            db_phot.check_before_commit(conn, image_params, 'facilities', facility_keys, 'facility_code')
            db_phot.check_before_commit(conn, image_params, 'images', image_keys, 'filename')
            log.info('Recorded image '+str(new_image)+' in DB')

            image_id = idx
            exposures_id.append(image_id)

            log.info('Starting difference photometry of ' + new_image)


            for stamp in list_of_stamps:

                image_params['stamp'] = str(stamp)

                stamp_row = np.where(reduction_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
                xmin = int(reduction_metadata.stamps[1][stamp_row]['X_MIN'])
                xmax = int(reduction_metadata.stamps[1][stamp_row]['X_MAX'])
                ymin = int(reduction_metadata.stamps[1][stamp_row]['Y_MIN'])
                ymax = int(reduction_metadata.stamps[1][stamp_row]['Y_MAX'])

                stamp_mask = (ref_star_catalog[:,1].astype(float) < xmax) & (ref_star_catalog[:,1].astype(float) > xmin) & \
                             (ref_star_catalog[:,2].astype(float) < ymax) & (ref_star_catalog[:,2].astype(float) > ymin)


                stamp_star_catalog = np.copy(ref_star_catalog[stamp_mask])
                stamp_star_catalog[:,1] =  (stamp_star_catalog[:,1].astype(float)-xmin).astype(str)
                stamp_star_catalog[:,2] =  (stamp_star_catalog[:,2].astype(float)-ymin).astype(str)


                kernel_image, kernel_error, kernel_bkg = find_the_associated_kernel_stamp(setup, kernels_directory, new_image,stamp, log)

                if kernel_image is None:
                    log.info('No kernel image available, so no photometry performed.')

                else:
                    sky_model = sky_background.model_sky_background(setup,
                                                                    reduction_metadata,
                                                                    log, ref_star_catalog,
                                                                    image_path=os.path.join(setup.red_dir,'data',new_image))
                    log.info('Built sky model')
                    stamp_directory = os.path.join(diffim_directory,new_image)
                    difference_image = open_an_image(setup, stamp_directory,'diff_stamp_'+str(stamp)+'.fits' , log, 0)[0]

                    if len(difference_image) > 1:

                        diff_table, control_zone, phot_table = photometry_on_the_difference_image_stamp(setup, reduction_metadata, log,
                                                                                  stamp_star_catalog, difference_image, psf_model,
                                                                                  sky_model, kernel_image, kernel_error,
                                                                                  ref_exposure_time,idx)
                        psf_model.update_psf_parameters(psf_parameters)

                        #commit_stamp_photometry_matching(conn, image_params, reduction_metadata, matched_stars, phot_table,
                        #                                 log, verbose=False)
                        #commit_image_photometry_matching(conn, image_params, reduction_metadata, matched_stars, phot_table, log)

                        photometry_data = store_stamp_photometry_to_array(setup, conn, image_params, reduction_metadata,
                                                            photometry_data,
                                                            phot_table, matched_stars,
                                                            new_image, log)

                    else:
                        log.info('No difference image available, so no photometry performed.')

        output_txt_files = False
        if output_txt_files:

            if os.path.isdir(os.path.join(setup.red_dir, 'lightcurves')) == False:
                os.mkdir(os.path.join(setup.red_dir, 'lightcurves'))

            jd = np.array(date)

            for star in range(len(photometric_table[0, :, 0]))[:]:
                mag = photometric_table[:, star, [8,9]]
                lightcurve = np.c_[jd,mag]

                file_to_write = open(os.path.join(setup.red_dir,'lightcurves','light_'+str(star),'ab'))

                np.savetxt(file_to_write,lightcurve)

                file_to_write.close()


        reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=6, status=1, log=log)
        reduction_metadata.software[1]['stage6_version'] = stage6_version
        reduction_metadata.save_updated_metadata(
            reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
            reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
            log=log)

    hd5_utils.write_phot_hd5(setup,photometry_data,log=log)

    conn.close()
    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'

    return status, report


def background_subtract(setup, image, max_adu):
    masked_image = mask_saturated_pixels(setup, image, max_adu, log=None)
    sky_params = {'background_type': 'gradient',
                  'nx': image.shape[1], 'ny': image.shape[0],
                  'a0': 0.0, 'a1': 0.0, 'a2': 0.0}
    sky_model = generate_sky_model(sky_params)
    sky_fit = fit_sky_background(masked_image, sky_model, 'gradient', log=None)
    sky_params['a0'] = sky_fit[0][0]
    sky_params['a1'] = sky_fit[0][1]
    sky_params['a2'] = sky_fit[0][2]
    # sky_model = generate_sky_model(sky_params)
    sky_model_image = generate_sky_model_image(sky_params)

    return image - sky_model_image


def open_an_image(setup, image_directory, image_name, log, image_index=0):
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
    try:

        image_data = fits.open(os.path.join(image_directory_path, image_name),
                               mmap=True)
        image_data = image_data[image_index]

        try:
            date = image_data.header['MJD-OBS']
	    #exptime = image_data.header['EXPTIME']
        except:
            date = 0
	    #exptime = 1
        log.info(image_name + ' open : OK')

        return image_data.data, date

    except:
        log.info('Warning: '+image_name + ' open : not OK!')

        return np.zeros([1]), 0


def save_control_zone_of_residuals(setup, image_name, control_zone):
    '''
    Save selected stars for difference image control

    :param object reduction_metadata: the metadata object
    :param str image_name: the name of the image
    :param array_likecontrol_zone: the residuals stamps

    '''

    control_images_directory = setup.red_dir + '/res_images/'
    os.makedirs(control_images_directory, exist_ok=True)

    control_size = 50

    image_name.replace('.fits', '.res')

    hdu = fits.PrimaryHDU(control_zone)
    hdu.writeto(control_images_directory + image_name, overwrite=True)


def save_control_stars_of_the_difference_image(setup, image_name, difference_image, star_coordinates):
    '''
    Save selected stars for difference image control

    :param object reduction_metadata: the metadata object
    :param str image_name: the name of the image
    :param array_like difference_image: the reference image data
    :param array_like stars_coordinates: the position of control stars
    '''

    control_images_directory = setup.red_dir + 'diffim/'
    try:
        os.makedirs(control_images_directory)
    except:
        pass

    control_size = 50

    for star in star_coordinates:

        ind_i = int(np.round(star[1]))
        ind_j = int(np.round(star[2]))

        stamp = difference_image[int(ind_i - control_size / 2):int(ind_i + control_size / 2),
                int(ind_j - control_size / 2):int(ind_j + control_size / 2)]

        try:

            control_zone = np.c_[control_zone, stamp]

        except:

            control_zone = stamp

    image_name = image_name.replace('.fits', '.diff')

    hdu = fits.PrimaryHDU(difference_image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(control_images_directory + image_name, overwrite=True)


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
    # import pdb; pdb.set_trace()
    diffim = 'diff_' + image_name

    diffim, date = open_an_image(setup, diffim_directory, diffim, log,
                                 image_index=0)

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

    image_data, date = open_an_image(setup, './data/', image_name, log, image_index=0)
    row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_name)[0][0]

    kernel_size = kernel_data.shape[0]

    background_ref = background_subtract(setup, reference_image_data, np.median(reference_image_data))
    # background_ref = reference_image_data
    ref_extended = np.zeros((np.shape(background_ref)[0] + 2 * kernel_size,
                             np.shape(background_ref)[1] + 2 * kernel_size))
    ref_extended[kernel_size:-kernel_size, kernel_size:-
    kernel_size] = np.array(background_ref, float)

    model = convolution.convolve_image_with_a_psf(ref_extended, kernel_data)

    model = model[kernel_size:-kernel_size, kernel_size:-kernel_size]

    background_image = background_subtract(setup, image_data, np.median(image_data))
    # background_image = image_data
    xshift, yshift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'], - \
    reduction_metadata.images_stats[1][row_index]['SHIFT_Y']

    image_shifted = shift(background_image, (-yshift - 1, -xshift - 1), cval=0.)

    # import pdb;pdb.set_trace()

    difference_image = model - image_shifted

    return difference_image


def find_the_associated_kernel(setup, kernels_directory, image_name):
    '''
    Find the appropriate kernel associated to an image
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''
    # import pdb; pdb.set_trace()
    kernel_name = 'kernel_' + image_name
    kernel_err = 'kernel_err_' + image_name

    kernel = fits.open(os.path.join(kernels_directory, kernel_name))
    kernel_error = fits.open(os.path.join(kernels_directory, kernel_err))
    # kernel,date = open_an_image(setup, kernels_directory, kernel_name, log,
    #                       image_index=0)
    # kernel_error,date = open_an_image(setup, kernels_directory, kernel_err, log,
    #                       image_index=0)
    bkgd = +kernel[0].header['KERBKG']

    kernel = kernel[0].data

    return kernel, kernel_error[0].data, bkgd

def find_the_associated_kernel_stamp(setup, kernels_directory, image_name, stamp, log):
    '''
    Find the appropriate kernel associated to an image stamp
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''
    # import pdb; pdb.set_trace()
    kernel_name = 'kernel_stamp_' + str(stamp) +'.fits'
    kernel_err =  'kernel_err_stamp_' + str(stamp) +'.fits'

    kernel_directory = kernels_directory+'/'+image_name+'/'
    try:
        kernel = fits.open(os.path.join(kernel_directory, kernel_name))
        kernel_error = fits.open(os.path.join(kernel_directory, kernel_err))
        # kernel,date = open_an_image(setup, kernels_directory, kernel_name, log,
        #                       image_index=0)
        # kernel_error,date = open_an_image(setup, kernels_directory, kernel_err, log,
        #                       image_index=0)
        bkgd = +kernel[0].header['KERBKG']

        kernel = kernel[0].data

        return kernel, kernel_error[0].data, bkgd

    except FileNotFoundError:
        log.info('WARNING: No kernel found for stamp '+kernel_name+', skipping.')
        log.info('Looked for '+os.path.join(kernel_directory, kernel_name)+' and '+os.path.join(kernel_directory, kernel_err))

        return None, None, None

def photometry_on_the_difference_image(setup, reduction_metadata, log, star_catalog, difference_image, psf_model,
                                       sky_model, kernel, kernel_error, ref_exposure_time,image_id):
    '''
    Find the appropriate kernel associated to an image
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''

    # PSF photometry function returns a list of lists
    (differential_photometry, control_zone) = photometry.run_psf_photometry_on_difference_image(setup, reduction_metadata, log,
                                                                                star_catalog, sky_model,
                                                                                difference_image, psf_model, kernel,
                                                                                kernel_error, ref_exposure_time,image_id)

    table_data = [ Column(name='star_id', data=differential_photometry[0]),
                   Column(name='diff_flux', data=differential_photometry[1]),
                   Column(name='diff_flux_err', data=differential_photometry[2]),
                   Column(name='magnitude', data=differential_photometry[3]),
                   Column(name='magnitude_err', data=differential_photometry[4]),
                   Column(name='cal_magnitude', data=differential_photometry[5]),
                   Column(name='cal_magnitude_err', data=differential_photometry[6]),
                   Column(name='flux', data=differential_photometry[7]),
                   Column(name='flux_err', data=differential_photometry[8]),
                   Column(name='cal_flux', data=differential_photometry[9]),
                   Column(name='cal_flux_err', data=differential_photometry[10]),
                   Column(name='phot_scale_factor', data=differential_photometry[11]),
                   Column(name='phot_scale_factor_err', data=differential_photometry[12]),
                   Column(name='local_background', data=differential_photometry[13]),
                   Column(name='local_background_err', data=differential_photometry[14]),
                   Column(name='residual_x', data=differential_photometry[15]),
                   Column(name='residual_y', data=differential_photometry[16]),
                   Column(name='radius', data=differential_photometry[17]) ]

    photometric_table = Table(data=table_data)

    # return table
    return differential_photometry, control_zone, photometric_table


def photometry_on_the_difference_image_stamp(setup, reduction_metadata, log, star_catalog, difference_image, psf_model,
                                            sky_model, kernel, kernel_error, ref_exposure_time, image_id):
    '''
    Find the appropriate kernel associated to an image
    :param object reduction_metadata: the metadata object
    :param string kernels_directory: the path to the kernels
    :param string image_name: the image name

    :return: the associated kernel to the image
    :rtype: array_like
    '''

    # PSF photometry function returns a list of lists
    (differential_photometry, control_zone) = photometry.run_psf_photometry_on_difference_image(setup,
                                                                                                reduction_metadata, log,
                                                                                                star_catalog, sky_model,
                                                                                                difference_image,
                                                                                                psf_model, kernel,
                                                                                                kernel_error,
                                                                                                ref_exposure_time,
                                                                                                image_id)

    table_data = [Column(name='star_id', data=differential_photometry[0]),
                  Column(name='diff_flux', data=differential_photometry[1]),
                  Column(name='diff_flux_err', data=differential_photometry[2]),
                  Column(name='magnitude', data=differential_photometry[3]),
                  Column(name='magnitude_err', data=differential_photometry[4]),
                  Column(name='cal_magnitude', data=differential_photometry[5]),
                  Column(name='cal_magnitude_err', data=differential_photometry[6]),
                  Column(name='flux', data=differential_photometry[7]),
                  Column(name='flux_err', data=differential_photometry[8]),
                  Column(name='cal_flux', data=differential_photometry[9]),
                  Column(name='cal_flux_err', data=differential_photometry[10]),
                  Column(name='phot_scale_factor', data=differential_photometry[11]),
                  Column(name='phot_scale_factor_err', data=differential_photometry[12]),
                  Column(name='local_background', data=differential_photometry[13]),
                  Column(name='local_background_err', data=differential_photometry[14]),
                  Column(name='residual_x', data=differential_photometry[15]),
                  Column(name='residual_y', data=differential_photometry[16]),
                  Column(name='radius', data=differential_photometry[17])]

    photometric_table = Table(data=table_data)

    # return table
    return differential_photometry, control_zone, photometric_table





def ingest_photometric_table_in_db(setup, exposures_indexes, star_indexes, photometric_table):
    names = ('exposure_id', 'star_id', 'reference_mag', 'reference_mag_err', 'reference_flux', 'reference_flux_err',
             'diff_flux', 'diff_flux_err', 'magnitude', 'magnitude_err', 'phot_scale_factor', 'phot_scale_factor_err',
             'local_background', 'local_background_err', 'residual_x', 'residual_y')

    # if photometric_table exists
    if len(photometric_table) != 0:
        conn = db_phot.get_connection(dsn=setup.red_dir + 'phot.db')

        for ind_exp, exposure in enumerate(exposures_indexes):

            for ind_star, star in enumerate(star_indexes):
                phot_table = photometric_table[ind_exp, ind_star, :]

                phot_table = [[i] for i in phot_table]

                new_table = Table(phot_table, names=names)

                db_phot.ingest_astropy_table(conn, 'phot', new_table)
        conn.commit()

def harvest_stage6_parameters(setup,reduction_metadata,version):
    """Function to harvest the parameters required for ingest of a single
    dataset into the photometric database."""

    dataset_params = {}

    ref_path = reduction_metadata.data_architecture[1]['REF_PATH'][0]
    ref_filename = reduction_metadata.data_architecture[1]['REF_IMAGE'][0]

    ref_image_path = os.path.join(ref_path, ref_filename)

    dataset_params = stage3_db_ingest.harvest_image_params(reduction_metadata, ref_image_path, ref_image_path)

    dataset_params['ref_filename'] = ref_filename

    # Software
    dataset_params['version'] = version
    dataset_params['stage'] = 'stage6'
    dataset_params['code_name'] = 'stage6.py'

    return dataset_params

def check_stage3_ingest_complete(conn,params):
    """Function to verify whether the ingest of stage 3 results has taken
    place, a pre-requiste before stage 6 results can be ingested."""

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="' + params['ref_filename'] + '"'
    refimage = db_phot.query_to_astropy_table(conn, query, args=())

    if len(refimage) == 0:
        raise ValueError(
            'No Stage 3 results for this reference image available in photometry DB.  Stage3_db_ingest needs to be run for this dataset first.')
    else:
        return True

def match_dataset_with_field_primary_reference(setup,conn,dataset_params,
                                               reduction_metadata,log):
    """Function to compare the stars detected in the reference image for the
    current dataset with those in the primary reference dataset for this field.
    The transformation between the two is calculated so that stars from this
    dataset can be accurately matched with the master starlist in the
    photometry DB for this field.
    """

    starlist = stage3_db_ingest.fetch_field_starlist(conn,dataset_params,log)

    primary_refimg_id = db_phot.find_primary_reference_image_for_field(conn)

    matched_stars = stage3_db_ingest.match_catalog_entries_with_starlist(conn,dataset_params,
                                                        starlist,
                                                        reduction_metadata,
                                                        primary_refimg_id,log,
                                                        verbose=True)

    transform = stage3_db_ingest.calc_transform_to_primary_ref(setup,matched_stars,log)

    matched_stars = stage3_db_ingest.match_all_entries_with_starlist(setup,conn,dataset_params,
                                                    starlist,reduction_metadata,
                                                    primary_refimg_id,transform,log,
                                                    verbose=True)

    return transform, matched_stars

def commit_image_photometry_matching(conn, params, reduction_metadata,
                                     matched_stars, phot_table, log):

    log.info('Starting database ingest')

    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="'+params['facility_code']+'"'
    facility = db_phot.query_to_astropy_table(conn, query, args=())

    query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="'+params['filter_name']+'"'
    f = db_phot.query_to_astropy_table(conn, query, args=())

    query = 'SELECT code_id, version FROM software WHERE version="'+params['version']+'"'
    code = db_phot.query_to_astropy_table(conn, query, args=())

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="'+params['ref_filename']+'"'
    refimage = db_phot.query_to_astropy_table(conn, query, args=())

    if len(refimage) == 0:
        raise ValueError('No Stage 3 results for this reference image available in photometry DB.  Stage3_db_ingest needs to be run for this dataset first.')

    query = 'SELECT img_id, filename FROM images WHERE filename ="'+params['filename']+'"'
    image = db_phot.query_to_astropy_table(conn, query, args=())

    log.info('Extracted dataset identifiers from database')

    key_list = ['star_id', 'reference_image', 'image',
                'facility', 'filter', 'software',
                'x', 'y', 'hjd', 'radius', 'magnitude', 'magnitude_err',
                'calibrated_mag', 'calibrated_mag_err',
                'flux', 'flux_err',
                'calibrated_flux', 'calibrated_flux_err',
                'phot_scale_factor', 'phot_scale_factor_err',
                'local_background', 'local_background_err',
                'phot_type']

    wildcards = ','.join(['?']*len(key_list))

    n_stars = len(phot_table)

    entries = []

    log.info('Building database entries array')

    for i in range(0,matched_stars.n_match,1):

        j_cat = matched_stars.cat1_index[i]     # Starlist index in DB
        j_new = matched_stars.cat2_index[i]     # Star detected in image

        x = str(phot_table['residual_x'][j_new])
        y = str(phot_table['residual_y'][j_new])
        radius = str(phot_table['radius'][j_new])
        mag = str(phot_table['magnitude'][j_new])
        mag_err = str(phot_table['magnitude_err'][j_new])
        cal_mag = str(phot_table['cal_magnitude'][j_new])
        cal_mag_err = str(phot_table['cal_magnitude_err'][j_new])
        flux = str(phot_table['flux'][j_new])
        flux_err = str(phot_table['flux_err'][j_new])
        cal_flux = str(phot_table['cal_flux'][j_new])
        cal_flux_err = str(phot_table['cal_flux_err'][j_new])
        ps = str(phot_table['phot_scale_factor'][j_new])
        ps_err = str(phot_table['phot_scale_factor_err'][j_new])
        bkgd = str(phot_table['local_background'][j_new])
        bkgd_err = str(phot_table['local_background_err'][j_new])

        entry = (str(int(j_cat)), str(refimage['refimg_id'][0]), str(image['img_id'][0]),
                   str(facility['facility_id'][0]), str(f['filter_id'][0]), str(code['code_id'][0]),
                    x, y, str(params['hjd']), radius,
                    mag, mag_err, cal_mag, cal_mag_err,
                    flux, flux_err, cal_flux, cal_flux_err,
                    ps, ps_err,   # No phot scale factor for PSF fitting photometry
                    bkgd, bkgd_err,  # No background measurements propageted
                    'DIA' )

        entries.append(entry)

    if len(entries) > 0:

        log.info('Ingesting data to phot_db')

        command = 'INSERT OR REPLACE INTO phot('+','.join(key_list)+\
                ') VALUES ('+wildcards+')'

        cursor = conn.cursor()

        cursor.executemany(command,entries)

        conn.commit()

    else:

        log.info('No photometry to be ingested')

    log.info('Completed ingest of photometry for '+str(len(matched_stars.cat1_index))+' stars')


def commit_stamp_photometry_matching(conn, params, reduction_metadata,
                                     matched_stars, phot_table, log,
                                     verbose=False):
    log.info('Starting database ingest')

    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="' + params['facility_code'] + '"'
    facility = db_phot.query_to_astropy_table(conn, query, args=())

    query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="' + params['filter_name'] + '"'
    f = db_phot.query_to_astropy_table(conn, query, args=())

    query = 'SELECT code_id, version FROM software WHERE version="' + params['version'] + '"'
    code = db_phot.query_to_astropy_table(conn, query, args=())

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="' + params['ref_filename'] + '"'
    refimage = db_phot.query_to_astropy_table(conn, query, args=())

    if len(refimage) == 0:
        raise ValueError(
            'No Stage 3 results for this reference image available in photometry DB.  Stage3_db_ingest needs to be run for this dataset first.')

    query = 'SELECT img_id, filename FROM images WHERE filename ="' + params['filename'] + '"'
    image = db_phot.query_to_astropy_table(conn, query, args=())

    query = 'SELECT stamp_id FROM stamps WHERE stamp_index ="' + params['stamp'] + '"'
    stamp = db_phot.query_to_astropy_table(conn, query, args=())


    log.info('Extracted dataset identifiers from database')

    key_list = ['star_id', 'reference_image', 'image','stamp',
                'facility', 'filter', 'software',
                'x', 'y', 'hjd', 'radius', 'magnitude', 'magnitude_err',
                'calibrated_mag', 'calibrated_mag_err',
                'flux', 'flux_err',
                'calibrated_flux', 'calibrated_flux_err',
                'phot_scale_factor', 'phot_scale_factor_err',
                'local_background', 'local_background_err',
                'phot_type']

    wildcards = ','.join(['?'] * len(key_list))

    n_stars = len(phot_table)

    entries = []

    log.info('Building database entries array for '+str(len(phot_table))+' stars in stamp')

    for i in range(0, len(phot_table), 1):
        star_dataset_id = int(float(phot_table[i]['star_id']))
        star_match_idx = matched_stars.find_star_match_index('cat2_index',star_dataset_id)

        if star_match_idx > 0:
            j_cat = matched_stars.cat1_index[star_match_idx]  # Starlist index in DB
            if verbose:
                log.info('Dataset star '+str(star_dataset_id)+\
                ' photometry being associated with primary reference star '+str(j_cat))

            x = str(phot_table['residual_x'][i])
            y = str(phot_table['residual_y'][i])
            radius = str(phot_table['radius'][i])
            mag = str(phot_table['magnitude'][i])
            mag_err = str(phot_table['magnitude_err'][i])
            cal_mag = str(phot_table['cal_magnitude'][i])
            cal_mag_err = str(phot_table['cal_magnitude_err'][i])
            flux = str(phot_table['flux'][i])
            flux_err = str(phot_table['flux_err'][i])
            cal_flux = str(phot_table['cal_flux'][i])
            cal_flux_err = str(phot_table['cal_flux_err'][i])
            ps = str(phot_table['phot_scale_factor'][i])
            ps_err = str(phot_table['phot_scale_factor_err'][i])
            bkgd = str(phot_table['local_background'][i])
            bkgd_err = str(phot_table['local_background_err'][i])

            entry = (str(int(j_cat)), str(refimage['refimg_id'][0]), str(image['img_id'][0]),str(stamp['stamp_id'][0]),
                     str(facility['facility_id'][0]), str(f['filter_id'][0]), str(code['code_id'][0]),
                     x, y, str(params['hjd']), radius,
                     mag, mag_err, cal_mag, cal_mag_err,
                     flux, flux_err, cal_flux, cal_flux_err,
                     ps, ps_err,  # No phot scale factor for PSF fitting photometry
                     bkgd, bkgd_err,  # No background measurements propageted
                     'DIA')

            entries.append(entry)

            if verbose:
                log.info(str(entry))

    log.info('Starting database ingest for '+str(len(entries))+' array')

    if len(entries) > 0:

        log.info('Ingesting data to phot_db')

        command = 'INSERT OR REPLACE INTO phot(' + ','.join(key_list) + \
                  ') VALUES (' + wildcards + ')'

        cursor = conn.cursor()

        cursor.executemany(command, entries)

        conn.commit()

    else:

        log.info('No photometry to be ingested')

    log.info('Completed ingest of photometry for ' + str(len(entries)) + ' stars')

def build_photometry_array(setup,nimages,nstars,log):
    """Function to construct an array to receive the photometry data.
    This loads pre-existing photometry from earlier pipeline reductions,
    if available.
    Note that the photometry_data array is listed by the dataset's star index,
    but includes the photometry database index for the primary reference for
    each star.
    """

    # Number of columns of measurements per star in the photometry table
    ncolumns = 23

    existing_phot = hd5_utils.read_phot_hd5(setup,log=log)

    if len(existing_phot) > 0 and existing_phot.shape[2] != ncolumns:
        raise IOError('Existing matched photometry array has '+\
                        str(matched_existing_phot.shape[2])+
                        ' which is incompatible with the expected '+\
                        str(ncolumns)+' columns')

    photometry_data = np.zeros((nstars,nimages,ncolumns))

    # If available, transfer the existing photometry into the data arrays
    if len(existing_phot) > 0:
        for i in range(0,existing_phot.shape[1],1):
            photometry_data[:,int(i),:] = existing_phot[:,int(i),:]

    log.info('Completed build of the photometry array')

    return photometry_data

def get_entry_db_indices(conn, params, new_image, log):

    log.info('Extracting the photometry DB pk indices for '+new_image)

    db_pk = {}

    query = 'SELECT facility_id, facility_code FROM facilities WHERE facility_code="' + params['facility_code'] + '"'
    result = db_phot.query_to_astropy_table(conn, query, args=())
    if len(result) > 0:
        db_pk['facility'] = result['facility_id'][0]
    else:
        raise IOError('Facility '+params['facility_code']+' unknown to phot_db')

    query = 'SELECT filter_id, filter_name FROM filters WHERE filter_name="' + params['filter_name'] + '"'
    result = db_phot.query_to_astropy_table(conn, query, args=())
    if len(result) > 0:
        db_pk['filter'] = result['filter_id'][0]
    else:
        raise IOError('Filter '+params['filter_name']+' unknown to phot_db')

    query = 'SELECT code_id, version FROM software WHERE version="' + params['version'] + '"'
    result = db_phot.query_to_astropy_table(conn, query, args=())
    if len(result) > 0:
        db_pk['code'] = result['code_id'][0]
    else:
        raise IOError('Software '+params['version']+' unknown to phot_db')

    query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="' + params['ref_filename'] + '"'
    result = db_phot.query_to_astropy_table(conn, query, args=())
    if len(result) > 0:
        db_pk['refimage'] = result['refimg_id'][0]
    else:
        raise ValueError(
            'No Stage 3 results for this reference image available in photometry DB.  Stage3_db_ingest needs to be run for this dataset first.')

    query = 'SELECT img_id, filename FROM images WHERE filename ="' + params['filename'] + '"'
    result = db_phot.query_to_astropy_table(conn, query, args=())
    if len(result) > 0:
        db_pk['image'] = result['img_id'][0]
    else:
        raise IOError('Image '+params['filename']+' unknown to phot_db')

    query = 'SELECT stamp_id FROM stamps WHERE stamp_index ="' + params['stamp'] + '"'
    result = db_phot.query_to_astropy_table(conn, query, args=())
    if len(result) > 0:
        db_pk['stamp'] = result['stamp_id'][0]
    else:
        raise IOError('Stamp '+params['stamp']+' unknown to phot_db')

    log.info('Extracted dataset identifiers from database')

    return db_pk


def store_stamp_photometry_to_array_starloop(conn, params, reduction_metadata,
                                    matched_photometry_data, unmatched_photometry_data,
                                    phot_table, matched_stars,
                                    new_image, log, verbose=False):
    """Function to store photometry data from a stamp to the main
    photometry array"""

    log.info('Starting to store photometry for image '+new_image)

    db_pk = get_entry_db_indices(conn, params, new_image, log)

    # The index of the data from a given image corresponds to the index of that
    # image in the metadata
    image_dataset_id = np.where(new_image == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]

    for j in range(0, len(phot_table), 1):
        star_dataset_id = int(float(phot_table[j]['star_id']))
        star_match_idx = matched_stars.find_star_match_index('cat2_index',star_dataset_id)

        x = phot_table['residual_x'][j]
        y = phot_table['residual_y'][j]
        radius = phot_table['radius'][j]
        mag = phot_table['magnitude'][j]
        mag_err = phot_table['magnitude_err'][j]
        cal_mag = phot_table['cal_magnitude'][j]
        cal_mag_err = phot_table['cal_magnitude_err'][j]
        flux = phot_table['flux'][j]
        flux_err = phot_table['flux_err'][j]
        cal_flux = phot_table['cal_flux'][j]
        cal_flux_err = phot_table['cal_flux_err'][j]
        ps = phot_table['phot_scale_factor'][j]
        ps_err = phot_table['phot_scale_factor_err'][j]
        bkgd = phot_table['local_background'][j]
        bkgd_err = phot_table['local_background_err'][j]

        if star_match_idx > 0:
            j_cat = matched_stars.cat1_index[star_match_idx]  # Starlist index in DB
            if verbose:
                log.info('Dataset star '+str(star_dataset_id)+\
                ' photometry being associated with primary reference star '+str(j_cat))

            entry = np.array([str(int(j_cat)), str(db_pk['refimage']), str(db_pk['image']),str(db_pk['stamp']),
                     str(db_pk['facility']), str(db_pk['filter']), str(db_pk['code']),
                     x, y, str(params['hjd']), radius,
                     mag, mag_err, cal_mag, cal_mag_err,
                     flux, flux_err, cal_flux, cal_flux_err,
                     ps, ps_err,  # No phot scale factor for PSF fitting photometry
                     bkgd, bkgd_err])  # No background measurements propageted

            matched_photometry_data[star_dataset_id-1,image_dataset_id-1,:] = entry

            if verbose:
                log.info(str(entry))

        else:
            if verbose:
                log.info('Dataset star '+str(star_dataset_id)+\
                ' is unmatched with the primary reference catalogue')

            entry = np.array([str(db_pk['refimage']), str(db_pk['image']),str(db_pk['stamp']),
                     str(db_pk['facility']), str(db_pk['filter']), str(db_pk['code']),
                     x, y, str(params['hjd']), radius,
                     mag, mag_err, cal_mag, cal_mag_err,
                     flux, flux_err, cal_flux, cal_flux_err,
                     ps, ps_err,  # No phot scale factor for PSF fitting photometry
                     bkgd, bkgd_err])  # No background measurements propageted

            unmatched_photometry_data[star_dataset_id-1,image_dataset_id-1,:] = entry

            if verbose:
                log.info(str(entry))

    log.info('Completed build of the photometry array')

    return matched_photometry_data, unmatched_photometry_data

def store_stamp_photometry_to_array(setup, conn, params, reduction_metadata,
                                    photometry_data,
                                    phot_table, matched_stars,
                                    new_image, log, verbose=False, debug=False):
    """Function to store photometry data from a stamp to the main
    photometry array"""

    log.info('Starting to store photometry for image '+new_image)

    if debug:
        matched_stars.output_match_list(os.path.join(setup.red_dir,'matched_stars.txt'))

    db_pk = get_entry_db_indices(conn, params, new_image, log)

    # The index of the data from a given image corresponds to the index of that
    # image in the metadata
    image_dataset_id = np.where(new_image == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]
    image_dataset_index = image_dataset_id - 1

    star_dataset_ids = np.array(phot_table['star_id'].data)
    star_dataset_ids = star_dataset_ids.astype('float')
    star_dataset_ids = star_dataset_ids.astype('int')
    star_dataset_index = star_dataset_ids - 1

    if debug:
        f = open(os.path.join(setup.red_dir, 'star_dataset_ids.txt'),'w')
        for s in star_dataset_ids:
            f.write(str(s)+'\n')
        f.close()

    log.info('Starting match index search for '+str(len(star_dataset_ids))+' stars')
    (star_dataset_ids, star_field_ids) = matched_stars.find_starlist_match_ids('cat2_index', star_dataset_ids, log,
                                                                                verbose=True)

    log.info('Starting to array data transfer')
    photometry_data[star_dataset_index,image_dataset_index,0] = star_field_ids
    photometry_data[star_dataset_index,image_dataset_index,1] = db_pk['refimage']
    photometry_data[star_dataset_index,image_dataset_index,2] = db_pk['image']
    photometry_data[star_dataset_index,image_dataset_index,3] = db_pk['stamp']
    photometry_data[star_dataset_index,image_dataset_index,4] = db_pk['facility']
    photometry_data[star_dataset_index,image_dataset_index,5] = db_pk['filter']
    photometry_data[star_dataset_index,image_dataset_index,6] = db_pk['code']
    photometry_data[star_dataset_index,image_dataset_index,7] = phot_table['residual_x'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,8] = phot_table['residual_y'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,9] = params['hjd']
    photometry_data[star_dataset_index,image_dataset_index,10] = phot_table['radius'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,11] = phot_table['magnitude'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,12] = phot_table['magnitude_err'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,13] = phot_table['cal_magnitude'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,14] = phot_table['cal_magnitude_err'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,15] = phot_table['flux'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,16] = phot_table['flux_err'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,17] = phot_table['cal_flux'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,18] = phot_table['cal_flux_err'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,19] = phot_table['phot_scale_factor'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,20] = phot_table['phot_scale_factor_err'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,21] = phot_table['local_background'][:].astype('float')
    photometry_data[star_dataset_index,image_dataset_index,22] = phot_table['local_background_err'][:].astype('float')

    log.info('Completed build of the photometry array')

    return photometry_data

def load_matched_stars_from_metadata(reduction_metadata,log):
    """Function to read the list of dataset stars matched against the field catalog
    and the transformation between the two from the metadata"""

    return transform, matched_stars
