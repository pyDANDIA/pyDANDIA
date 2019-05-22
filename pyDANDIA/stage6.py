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

    # create the starlist table in db, if needed
    #ingest_the_stars_in_db(setup, starlist)

    # find star indexes in the db
    #star_indexes = find_stars_indexes_in_db(setup) 
    
    psf_model = fits.open(reduction_metadata.data_architecture[1]['REF_PATH'].data[0] + '/psf_model.fits')

    psf_type = psf_model[0].header['PSFTYPE']
    #import pdb;
    #pdb.set_trace()
    psf_parameters = [ psf_model[0].header['INTENSIT'], psf_model[0].header['Y_CENTER'],
                      psf_model[0].header['X_CENTER'],
                      psf_model[0].header['GAMMA'],
                      psf_model[0].header['ALPHA']]

    sky_model = sky_background.model_sky_background(setup,
                                                    reduction_metadata, log, ref_star_catalog)

    psf_model = psf.get_psf_object(psf_type)
    psf_model.update_psf_parameters(psf_parameters)

    ind = ((starlist['x'] - 150) ** 2 < 1) & ((starlist['y'] - 150) ** 2 < 1)

    time = []
    exposures_id = []
    photometric_table = []

    if len(new_images) > 0:

        # find the reference image
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            reference_image_directory = reduction_metadata.data_architecture[1]['REF_PATH'].data[0]
            reference_image, date = open_an_image(setup, reference_image_directory, reference_image_name, image_index=0,
                                                  log=None)

            ref_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
            index_reference = np.where(ref_image_name == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]
            ref_exposure_time = float(reduction_metadata.headers_summary[1]['EXPKEY'].data[index_reference])

            reference_header = reduction_metadata.headers_summary[1][index_reference]

            # create the reference table in db
            #ingest_reference_in_db(setup, reference_header, reference_image_directory, reference_image_name)
            #conn = db_phot.get_connection(dsn=setup.red_dir + 'phot.db')
            #ref_image_id = db_phot.query_to_astropy_table(conn, "SELECT refimg_id FROM reference_images")[0][0]
            #conn.commit()

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

        photometric_table = np.zeros((len(new_images), len(ref_star_catalog), 16))
        compt_db = 0

        for idx, new_image in enumerate(new_images[:]):
            print(new_image)
            try:
                index_image = np.where(new_image == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]
                image_header = reduction_metadata.headers_summary[1][index_image]

                ddate = reduction_metadata.headers_summary[1]['DATEKEY'][index_image]
                jd =  dateutil.parser.parse(ddate)
                time = astropy.time.Time(jd)
                date.append(time.jd)

                #ingest_exposure_in_db(setup, image_header, ref_image_id)
                #conn = db_phot.get_connection(dsn=setup.red_dir + 'phot.db')

               # image_id = db_phot.query_to_astropy_table(conn,
              #                                          "SELECT exposure_id FROM exposures WHERE exposure_name='%s'" % new_image)[
                #    0][0]

                #conn.commit()
                image_id = idx
                exposures_id.append(image_id)

                log.info('Starting difference photometry of ' + new_image)
                #target_image,date = open_an_image(setup, images_directory, new_image, image_index=0, log=None)
                kernel_image, kernel_error, kernel_bkg = find_the_associated_kernel(setup, kernels_directory, new_image)

                # difference_image = subtract_images(target_image, reference_image, kernel_image, kernel_size, kernel_bkg)
                difference_image = open_an_image(setup, diffim_directory, 'diff_' + new_image, 0, log=None)[0]


                # save_control_stars_of_the_difference_image(setup, new_image, difference_image, star_coordinates)
                #import pdb;
                #pdb.set_trace()
                phot_table, control_zone = photometry_on_the_difference_image(setup, reduction_metadata, log,
                                                                              ref_star_catalog, difference_image, psf_model,
                                                                              sky_model, kernel_image, kernel_error,
                                                                              ref_exposure_time,idx)
                psf_model.update_psf_parameters(psf_parameters)

                photometric_table[compt_db, :, :] = phot_table
                phot_table = np.zeros(phot_table.shape)
                compt_db += 1

            except:

                # save_control_zone_of_residuals(setup, new_image, control_zone)

                # ingest_photometric_table_in_db(setup, photometric_table)
                compt_db += 1

                # if compt_db >9:

                #   ingest_photometric_table_in_db(setup, exposures_id, star_indexes, photometric_table)
                #   photometric_table = np.zeros((10,len(ref_star_catalog),16))
                #   exposures_id = []
                #   compt_db = 0


        jd = np.array(date)

        for star in range(len(photometric_table[0, :, 0]))[:]:
            mag = photometric_table[:, star, [8,9]]
            lightcurve = np.c_[jd,mag]

            file_to_write = open('./lightcurves/light_'+str(star),'ab')


            np.savetxt(file_to_write,lightcurve)

            file_to_write.close()

        #ingest_photometric_table_in_db(setup, exposures_id, star_indexes, photometric_table)

        reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=6, status=1, log=log)
        reduction_metadata.save_updated_metadata(
            reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
            reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
            log=log)

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
    # import pdb; pdb.set_trace()
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
        logs.ifverbose(log, setup, image_name + ' open : OK')

        return image_data.data, date

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

    diffim, date = open_an_image(setup, diffim_directory, diffim,
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

    image_data, date = open_an_image(setup, './data/', image_name, image_index=0, log=None)
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
    # kernel,date = open_an_image(setup, kernels_directory, kernel_name,
    #                       image_index=0, log=None)
    # kernel_error,date = open_an_image(setup, kernels_directory, kernel_err,
    #                       image_index=0, log=None)
    bkgd = +kernel[0].header['KERBKG']

    kernel = kernel[0].data

    return kernel, kernel_error[0].data, bkgd


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

    differential_photometry = photometry.run_psf_photometry_on_difference_image(setup, reduction_metadata, log,
                                                                                star_catalog,
                                                                                difference_image, psf_model, kernel,
                                                                                kernel_error, ref_exposure_time,image_id)

    column_names = (
    'exposure_id', 'star_id', 'reference_mag', 'reference_mag_err', 'reference_flux', 'reference_flux_err', 'diff_flux',
    'diff_flux_err', 'magnitude', 'magnitude_err',
    'phot_scale_factor', 'phot_scale_factor_err', 'local_background', 'local_background_err', 'residual_x',
    'residual_y')

    column_types = ('i8', 'i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
    photometric_table = Table(differential_photometry[0], names=column_names, dtype=column_types)

    # return table
    return differential_photometry


def ingest_the_stars_in_db(setup, star_catalog):
    conn = db_phot.get_connection(dsn=setup.red_dir + 'phot.db')

    # checkif the catalog exist
    indexes = db_phot.query_to_astropy_table(conn, "SELECT star_id FROM stars")['star_id']

    if len(indexes) == 0:

        print('I create a new star catalog for the db')

        new_table = star_catalog[['RA_J2000', 'DEC_J2000']]
        new_table['RA_J2000'].name = 'ra'
        new_table['DEC_J2000'].name = 'dec'
        db_phot.ingest_astropy_table(conn, 'stars', new_table)
        conn.commit()

    else:

        print('A star catalog for exists in the db, I skip the creation.')


def find_stars_indexes_in_db(setup):
    conn = db_phot.get_connection(dsn=setup.red_dir + 'phot.db')
    indexes = db_phot.query_to_astropy_table(conn, "SELECT star_id FROM stars")['star_id']

    return indexes


def ingest_reference_in_db(setup, reference_header, reference_image_directory, reference_image_name):
    names = ('telescope_id', 'instrument_id', 'filter_id', 'refimg_fwhm', 'refimg_fwhm_err', 'refimg_ellipticity',
             'refimge_ellipticity_err', 'refimg_name', 'wcsfrcat', 'wcsimcat', 'wcsmatch', 'wcsnref', 'wcstol',
             'wcsra', 'wcsdec', 'wequinox', 'wepoch', 'radecsys', 'cdelt1', 'cdelt2', 'crota1', 'crota2', 'secpix1',
             'secpix2',
             'wcssep', 'equinox', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2', 'epoch')

    conn = db_phot.get_connection(dsn=setup.red_dir + 'phot.db')

    name = reference_image_name
    cam_filter = reference_header['FILTKEY']
    telescope_id = name.split('-')[0]
    camera_id = name.split('-')[1]

    new_table = Table([[name], [cam_filter], [telescope_id], [camera_id]],
                      names=('refimg_name', 'filter_id', 'telescope_id',
                             'instrument_id'))

    db_phot.ingest_astropy_table(conn, 'reference_images', new_table)
    conn.commit()
    # what need to be filled...

    # c_020_telescope_id = 'TEXT'
    # c_030_instrument_id = 'TEXT'
    # c_040_filter_id = 'TEXT'
    # c_050_refimg_fwhm = 'REAL'
    # c_060_refimg_fwhm_err = 'REAL'
    # c_070_refimg_ellipticity = 'REAL'
    # c_080_refimg_ellipticity_err = 'REAL'
    # c_090_slope = 'REAL' #The slope of the photometric calibration: VPHAS mags vs instr mags
    # c_095_slope_err = 'REAL'
    # c_100_intercept = 'REAL' #The intercept of the photometric calibration: VPHAS mags vs instr mags
    # c_105_intercept_err = 'REAL'
    # c_120_refimg_name = 'TEXT'
    # c_130_wcsfrcat = 'TEXT' #WCS fit information stored in the next lines (c_130 to c_152)
    # c_131_wcsimcat = 'TEXT'
    # c_132_wcsmatch = 'INTEGER'
    # c_133_wcsnref = 'INTEGER'
    # c_134_wcstol = 'REAL'
    # c_135_wcsra = 'TEXT'
    # c_136_wcsdec = 'TEXT'
    # c_137_wequinox = 'INTEGER'
    # c_138_wepoch = 'INTEGER'
    # c_139_radecsys = 'FK5'
    # c_140_cdelt1 = 'DOUBLE PRECISION'
    # c_141_cdelt2 = 'DOUBLE PRECISION'
    # c_142_crota1 = 'DOUBLE PRECISION'
    # c_143_crota2 = 'DOUBLE PRECISION'
    # c_144_secpix1 = 'REAL'
    # c_145_secpix2 = 'REAL'
    # c_146_wcssep = 'REAL'
    # c_147_equinox = 'INTEGER'
    # c_148_cd1_1 = 'DOUBLE PRECISION'
    # c_149_cd1_2 = 'DOUBLE PRECISION'
    # c_150_cd2_1 = 'DOUBLE PRECISION'
    # c_151_cd2_2 = 'DOUBLE PRECISION'
    # c_152_epoch = 'INTEGER'
    #


def ingest_exposure_in_db(setup, image_header, ref_image_id):
    conn = db_phot.get_connection(dsn=setup.red_dir + 'phot.db')

    # import pdb; pdb.set_trace()
    image_name = image_header['IMAGES']
    exposure_time = float(image_header['EXPKEY'])
    new_table = Table([[image_name], [exposure_time]], names=('exposure_name', 'exposure_time'))

    db_phot.ingest_astropy_table(conn, 'exposures', new_table)
    conn.commit()
    # what need to be filled...

    # c_000_exposure_id = 'INTEGER PRIMARY KEY'
    # c_005_reference_image = 'INTEGER REFERENCES reference_images(refimg_id)'
    # c_010_jd = 'DOUBLE PRECISION'
    # c_050_exposure_fwhm = 'REAL'
    # c_060_exposure_fwhm_err = 'REAL'
    # c_050_exposure_ellipticity = 'REAL'
    # c_060_exposure_ellipticity_err = 'REAL'
    # c_110_airmass = 'REAL'
    # c_120_exposure_time = 'INTEGER'
    # c_130_moon_phase = 'REAL'
    # c_140_moon_separation = 'REAL'
    # c_150_delta_x = 'REAL'
    # c_160_delta_y = 'REAL'
    # c_170_exposure_name = 'TEXT'


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
