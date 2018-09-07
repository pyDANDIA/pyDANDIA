import numpy as np
import mock
import pytest
import os
from os import getcwd, path
from sys import path as systempath
import collections
from astropy.io import fits

cwd = getcwd()
systempath.append(path.join(cwd, '../'))

import stage0
import metadata


def test_read_the_config_file():
    pipeline_config = stage0.read_the_config_file('../../Config/', config_file_name='config.json',
                                                  log=None)

    assert len(pipeline_config) != 0


def test_find_the_inst_config_file_name():
    setup = mock.MagicMock()
    reduction_metadata = mock.MagicMock()
    reduction_metadata.data_architecture = [0, {
        'IMAGES_PATH': ['../tests/data/proc/ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip/data']}]

    image_name = 'lsc1m005-fl15-20170418-0131-e91_cropped.fits'

    inst_config_file_name = stage0.find_the_inst_config_file_name(setup, reduction_metadata, image_name, '../../Config',
                                                                  image_index=0, log=None)
    assert inst_config_file_name == 'inst_config_fl15.json'


def test_read_the_inst_config_file():
    inst_config = stage0.read_the_inst_config_file('../../Config/', inst_config_file_name='inst_config_fl15.json',
                                                   log=None)

    assert len(inst_config) != 0


def test_update_reduction_metadata_with_inst_config_file():
    reduction_metadata = metadata.MetaData()

    pipeline_config = stage0.read_the_config_file('../../Config/', log=None)

    stage0.update_reduction_metadata_with_config_file(reduction_metadata, pipeline_config, log=None)
    inst_config_dictionnary = {'fromage': {"comment": "tres bon",
                                           "value": "camembert",
                                           "format": "S200",
                                           "unit": ""}
        , 'dessert': {"comment": "moins bon",
                      "value": "pomme",
                      "format": "S200",
                      "unit": ""}}

    stage0.update_reduction_metadata_with_inst_config_file(reduction_metadata, inst_config_dictionnary, log=None)

    assert 'FROMAGE' in reduction_metadata.reduction_parameters[1].keys()
    assert reduction_metadata.reduction_parameters[1]['FROMAGE'] == 'camembert'
    assert reduction_metadata.reduction_parameters[1]['FROMAGE'].dtype == 'S200'
    assert reduction_metadata.reduction_parameters[1]['FROMAGE'].unit == ''

    assert 'DESSERT' in reduction_metadata.reduction_parameters[1].keys()
    assert reduction_metadata.reduction_parameters[1]['DESSERT'] == 'pomme'
    assert reduction_metadata.reduction_parameters[1]['DESSERT'].dtype == 'S200'
    assert reduction_metadata.reduction_parameters[1]['DESSERT'].unit == ''


def test_create_or_load_the_reduction_metadata():
    setup = mock.MagicMock()

    reduction_metadata = stage0.create_or_load_the_reduction_metadata(setup, './',
                                                                      metadata_name='HanSolo.fits',
                                                                      log=None)

    assert reduction_metadata.data_architecture[1]['METADATA_NAME'] == 'HanSolo.fits'
    assert reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'] == './'
    assert reduction_metadata.reduction_status[1].keys() == ['IMAGES', 'STAGE_0', 'STAGE_1', 'STAGE_2', 'STAGE_3',
                                                             'STAGE_4', 'STAGE_5', 'STAGE_6', 'STAGE_7']

    filehere = [i for i in os.listdir('./') if ('.fits' in i)]

    assert 'HanSolo.fits' in filehere

    os.remove('./HanSolo.fits')


def test_read_the_config_file():
    pipeline_config = stage0.read_the_config_file('../../Config/', log=None)

    expected_output = {'det_thresh': {'value': 3.0, 'format': 'float',
                                      'comment': 'det_thresh - DOUBLE - Detection threshold used to detect stars in units of image sky sigma. This parameter should be positive (Default value = 3.0).',
                                      'unit': ''}, 'mdarkpro': {'value': 0, 'format': 'int',
                                                                'comment': 'mdarkpro - INT - Switch for master dark frame correction (0 = NO : 1 = YES : Default value = 0).',
                                                                'unit': ''},
                       'expfrac': {'value': 0.5, 'format': 'float',
                                   'comment': 'expfrac - DOUBLE - Fraction of the exposure time to be added to the universal time at the start of the observation when calculating the GJD and HJD of the observation (Default value = 0.5).',
                                   'unit': ''}, 'mflatsigma': {'value': -1.0, 'format': 'float',
                                                               'comment': 'mflatsigma - DOUBLE - Sigma-clip threshold to be used if oscanmethod is set to sigclip (Default value = -1.0).',
                                                               'unit': ''},
                       'min_scale': {'value': -1.0, 'format': 'float',
                                     'comment': 'min_scale - DOUBLE - Minimum possible transformation scale factor (magnification) between any two images (Default value = -1.0).',
                                     'unit': ''}, 'trans_type': {'value': 'polynomial', 'format': 'S200',
                                                                 'comment': 'trans_type - STRING - Type of coordinate transformation to fit when fitting a coordinate transformation between two images (shift = General pixel shift : rot_shift = Rotation and general pixel shift : rot_mag_shift = Rotation, magnification, and general pixel shift : linear = Linear : polynomial = Polynomial : Default value = linear).',
                                                                 'unit': ''},
                       'mdarkmethod': {'value': 'sigclip', 'format': 'S200',
                                       'comment': 'mdarkmethod - STRING - Method for combining the dark frames (mean = Mean : median = Median : sigclip = Sigma-clipped mean : Default value = sigclip).',
                                       'unit': ''}, 'coeff2': {'value': 1e-06, 'format': 'float',
                                                               'comment': 'coeff2 - DOUBLE -|- Coefficients a1 and a2 in the linearisation equation: Xnew = X + a1*X^2 + a2*X^3',
                                                               'unit': ''}, 'oscanpro': {'value': 0, 'format': 'int',
                                                                                         'comment': 'oscanpro - INT - Switch for bias level correction (0 = No bias level correction : 1 = Single constant bias level correction : 2 = Per-image constant overscan correction : 3 = Vector overscan correction per image row : 4 = Vector overscan correction per image column : Default value = 0).',
                                                                                         'unit': ''},
                       'icdeg': {'value': 0, 'format': 'int',
                                 'comment': 'icdeg - INT - Degree of the 2D polynomial in spatial coordinates used to model non-uniform illumination in the flat frames. This parameter should be non-negative (Default value = 0).',
                                 'unit': ''}, 'psf_size': {'value': 8.0, 'format': 'float',
                                                           'comment': 'psf_size - DOUBLE - Size of the model PSF stamp in units of FWHM. This parameter should be positive (Default value = 8.0).',
                                                           'unit': ''}, 'diffpro': {'value': 0, 'format': 'int',
                                                                                    'comment': 'diffpro - INT - Switch for the method of difference image creation (see subtract routine : Default Value = 0).',
                                                                                    'unit': ''},
                       'smooth_pro': {'value': 0, 'format': 'int',
                                      'comment': 'smooth_pro - INT - Switch for image smoothing [0: No smoothing, 1,2,3: Gaussian smoothing (see reference image documentation)] (Default value = 0).',
                                      'unit': ''}, 'trans_auto': {'value': 0, 'format': 'int',
                                                                  'comment': 'trans_auto - INT - Switch for automatic determination of the coordinate transformation type when fitting a coordinate transformation between two images (0 = Pre-defined transformation type : 1 = Automatic transformation type : Default value = 0).',
                                                                  'unit': ''},
                       'use_reflist': {'value': 0, 'format': 'int',
                                       'comment': 'use_reflist - INT - Switch for using the images listed in the reference image combination list (0 = NO : 1 = YES : Default value = 0).',
                                       'unit': ''}, 'min_ell': {'value': 0.8, 'format': 'float',
                                                                'comment': 'min_ell - DOUBLE - Smallest acceptable value of the PSF ellipticity for an image to be used in the combined reference image. This parameter should be non-negative and less than or equal to 1.0 (Default value = 0.8).',
                                                                'unit': ''},
                       'star_space': {'value': 30.0, 'format': 'float',
                                      'comment': 'star_space - DOUBLE - Average spacing (pix) in each of the x and y coordinates between the stars to be considered for matching purposes. This parameter should be positive (Default value = 30.0).',
                                      'unit': ''}, 'smooth_fwhm': {'value': 0, 'format': 'float',
                                                                   'comment': 'smooth_fwhm - DOUBLE - Amount of image smoothing to perform (Default value = 0).',
                                                                   'unit': ''},
                       'mbiasmethod': {'value': 'sigclip', 'format': 'S200',
                                       'comment': 'mbiasmethod - STRING - Method for combining the bias frames (mean = Mean : median = Median : sigclip = Sigma-clipped mean : Default value = sigclip).',
                                       'unit': ''}, 'max_scale': {'value': -1.0, 'format': 'float',
                                                                  'comment': 'max_scale - DOUBLE - Maximum possible transformation scale factor (magnification) between any two images (Default value = -1.0).',
                                                                  'unit': ''}, 'growsatx': {'value': 0, 'format': 'int',
                                                                                            'comment': 'growsatx - INT - Half box size in the x direction (pix) to be used for growing saturated bad pixels in the bad pixel mask for each science image. This parameter should be non-negative (Default value = 0).',
                                                                                            'unit': 'pixel'},
                       'psf_corr_thresh': {'value': 0.9, 'format': 'float',
                                           'comment': 'psf_corr_thresh - DOUBLE - Minimum correlation coefficient of a star with the image PSF model in order to be considered a PSF star. This parameter should be non-negative (Default value = 0.9).',
                                           'unit': ''}, 'oscandeg': {'value': 0, 'format': 'int',
                                                                     'comment': 'oscandeg - INT - Degree (or maximum degree) of the polynomial model to be fitted to the overscan region as a function of the image row/column (oscanpro set to 3 or 4). This parameter should be non-negative (Default value = 0).',
                                                                     'unit': ''},
                       'mdarksigma': {'value': -1.0, 'format': 'float',
                                      'comment': 'mdarksigma - DOUBLE - Sigma-clip threshold to be used if mdarkmethod is set to sigclip (Default value = -1.0).',
                                      'unit': ''},
                       'proc_data': {'value': '/home/Tux/ytsapras/Programs/Workspace/pyDANDIA/pyDANDIA/tests/data/proc',
                                     'format': 'S200',
                                     'comment': 'proc_data - STRING - The full path to the reduction directory.',
                                     'unit': ''}, 'var_deg': {'value': 1, 'format': 'int',
                                                              'comment': 'var_deg - INT - Polynomial degree of the spatial variation of the model used to represent the image PSF (0 = Constant : 1 = Linear : 2 = Quadratic : 3 = Cubic : Default value = 0).',
                                                              'unit': ''}, 'back_var': {'value': 1, 'format': 'int',
                                                                                        'comment': 'back_var - INT - Switch for a spatially variable differential background (1 = YES : 0 = NO : Default Value = 1).',
                                                                                        'unit': ''},
                       'flim': {'value': 2.0, 'format': 'float',
                                'comment': 'flim - DOUBLE - Minimum contrast between the Laplacian image and the fine structure image. This parameter should be positive (Default value = 2.0).',
                                'unit': ''}, 'max_sky': {'value': 5000, 'format': 'float',
                                                         'comment': 'max_sky - DOUBLE - Largest acceptable value of the sky background for an image to be used in the combined reference image (ADU : Default value = 5000).',
                                                         'unit': 'adu'}, 'mflatpro': {'value': 0, 'format': 'int',
                                                                                      'comment': 'mflatpro - INT - Switch for master flat frame correction (0 = NO : 1 = YES : Default value = 0).',
                                                                                      'unit': ''},
                       'grow': {'value': 0.0, 'format': 'float',
                                'comment': 'grow - DOUBLE - Controls the amount of overlap between the image regions used for the kernel solutions (Default value = 0.0).',
                                'unit': 'pixel'}, 'ps_var': {'value': 0, 'format': 'int',
                                                             'comment': 'ps_var - INT - Switch for a spatially variable photometric scale factor (1 = YES : 0 = NO : Default Value = 0).',
                                                             'unit': ''}, 'replace_cr': {'value': 0, 'format': 'int',
                                                                                         'comment': 'replace_cr - INT - Switch for replacing pixel values that have been contaminated by cosmic ray events (0 = NO : 1 = YES : Default value = 0).',
                                                                                         'unit': ''},
                       'mbiaspro': {'value': 0, 'format': 'int',
                                    'comment': 'mbiaspro - INT - Switch for master bias frame correction (0 = NO : 1 = YES : Default value = 0).',
                                    'unit': ''}, 'psf_comp_flux': {'value': 0.1, 'format': 'float',
                                                                   'comment': 'psf_comp_flux - DOUBLE - Maximum flux ratio that any companion star may have for a star to be considered a PSF star. This parameter should be non-negative (Default value = 0.1).',
                                                                   'unit': ''},
                       'subframes_y': {'value': 1, 'format': 'int',
                                       'comment': 'subframes_y - INT - Number of subdivisions in the y direction used in defining the grid of kernel solutions (Default value = 1).',
                                       'unit': ''}, 'mbiassigma': {'value': -1.0, 'format': 'float',
                                                                   'comment': 'mbiassigma - DOUBLE - Sigma-clip threshold to be used if mbiasmethod is set to sigclip (Default value = -1.0).',
                                                                   'unit': ''},
                       'niter_ker': {'value': 3, 'format': 'int',
                                     'comment': 'niter_ker - INT - Maximum number of iterations to perform when determining the kernel solution (Default value = 3),',
                                     'unit': ''}, 'max_nimages': {'value': 1, 'format': 'int',
                                                                  'comment': 'max_nimages - INT - Maximum number of images to be used in the combined reference image (Default value = 1).',
                                                                  'unit': ''},
                       'lres_ker_rad': {'value': 2.0, 'format': 'float',
                                        'comment': 'lres_ker_rad - DOUBLE - Threshold radius of the kernel pixel array, in units of image FWHM, beyond which kernel pixels are of lower resolution (Default value = 2.0).',
                                        'unit': ''}, 'niter_cos': {'value': 4, 'format': 'int',
                                                                   'comment': 'niter_cos - INT - Maximum number of iterations to perform. This parameter should be positive (Default value = 4).',
                                                                   'unit': ''},
                       'cppcode': {'value': '/path/to/C++/code/', 'format': 'S200',
                                   'comment': 'cppcode - STRING - The full directory path indicating where the DanIDL C++ code is installed.',
                                   'unit': ''}, 'psf_comp_dist': {'value': 0.7, 'format': 'float',
                                                                  'comment': 'psf_comp_dist - DOUBLE - Any star within a distance 0.5*psf_comp_dist*psf_size, in units of FWHM, of another star is considered to be a companion of that star for PSF star selection purposes. This parameter should be non-negative (Default value = 0.7).',
                                                                  'unit': ''},
                       'psf_thresh': {'value': 10.0, 'format': 'float',
                                      'comment': 'psf_thresh - DOUBLE - Detection threshold used to detect candidate PSF stars in units of image sky sigma. This parameter should be positive (Default value = 10.0).',
                                      'unit': ''}, 'growsaty': {'value': 0, 'format': 'int',
                                                                'comment': 'growsaty - INT - Half box size in the y direction (pix) to be used for growing saturated bad pixels in the bad pixel mask for each science image. This parameter should be non-negative (Default value = 0).',
                                                                'unit': 'pixel'},
                       'init_mthresh': {'value': 1.0, 'format': 'float',
                                        'comment': 'init_mthresh - DOUBLE - Initial distance threshold (pix) to reject false star matches. This parameter should be positive (Default value = 1.0).',
                                        'unit': 'pixel'}, 'ker_rad': {'value': 2.0, 'format': 'float',
                                                                      'comment': 'ker_rad - DOUBLE - Radius of the kernel pixel array in units of image FWHM (Default value = 2.0).',
                                                                      'unit': ''},
                       'sigfrac': {'value': 0.5, 'format': 'float',
                                   'comment': 'sigfrac - DOUBLE - Fraction of sigclip to be used as a threshold for cosmic ray growth. This parameter should be positive (Default value = 0.5).',
                                   'unit': ''}, 'oscanauto': {'value': 0, 'format': 'int',
                                                              'comment': 'oscanauto - INT - Switch for automatic polynomial model degree determination when calculating a vector overscan correction (oscanpro set to 3 or 4) (0 = Pre-defined degree : 1 = Automatic degree : Default value = 0).',
                                                              'unit': ''}, 'subframes_x': {'value': 1, 'format': 'int',
                                                                                           'comment': 'subframes_x - INT - Number of subdivisions in the x direction used in defining the grid of kernel solutions (Default value = 1).',
                                                                                           'unit': ''},
                       'mflatmethod': {'value': 'median', 'format': 'S200',
                                       'comment': 'mflatmethod - STRING - Method for combining the flat frames (mean = Mean : median = Median : sigclip = Sigma-clipped mean : Default value = median).',
                                       'unit': ''}, 'coeff3': {'value': 1e-12, 'format': 'float',
                                                               'comment': 'coeff3 - DOUBLE -|  where X represents the image counts after bias level and bias pattern correction.',
                                                               'unit': ''}, 'sigclip': {'value': 4.5, 'format': 'float',
                                                                                        'comment': 'sigclip - DOUBLE - Threshold in units of sigma for cosmic ray detection on the Laplacian image. This parameter should be positive (Default value = 4.5).',
                                                                                        'unit': ''}}
    assert len(pipeline_config) != 0
    assert pipeline_config == expected_output


def test_read_the_inst_config_file_name():
    inst_config = stage0.read_the_inst_config_file('../../Config/', inst_config_file_name='inst_config_fl15.json',
                                                   log=None)

    expected_output = {'datekey': {
        'comment': 'datekey - STRING - Header keyword recording the date (universal time) at the start of the observation. The value of this keyword should be a STRING of the format YYYY-MM-DDxxx...xxx where YYYY-MM-DD is a valid date and xxx...xxx represents a string of arbitrary characters of any length.',
        'format': 'S200', 'unit': 'day', 'value': 'DATE-OBS'}, 'gain': {
        'comment': 'gain - DOUBLE - CCD gain (e-/ADU). This parameter should be positive (Default value = 1.0)',
        'format': 'float', 'unit': 'e/adu', 'value': 1.0}, 'timekey': {
        'comment': 'timekey - STRING - Header keyword recording the universal time at the start of the observation. The value of this keyword should be a NUMBER in hours, or a STRING of the format HH:MM:SS.SSS, where HH:MM:SS.SSS is a valid time.',
        'format': 'S200', 'unit': '', 'value': 'UTSTART'},
        'pix_scale': {'comment': 'pix_scale - DOUBLE - The pixel scale in arcsec per pixel',
                      'format': 'float', 'unit': 'arcsec/pixel', 'value': 0.389}, 'deckey': {
            'comment': 'deckey - STRING - Header keyword recording the declination (J2000) of the observation. The value of this keyword should be a STRING of the format pDD:MM:SS.SSS, where pDD:MM:SS.SSS is an angle between +-90 degrees in sexagesimal format.',
            'format': 'S200', 'unit': 'deg', 'value': 'DEC'},
        'minval': {'comment': 'minval - DOUBLE - Minimum useful pixel value in a raw image (ADU).',
                   'format': 'float', 'unit': '', 'value': 1.0}, 'rakey': {
            'comment': 'rakey - STRING - Header keyword recording the right ascension (J2000) of the observation. The value of this keyword should be a STRING of the format HH:MM:SS.SSS, where HH:MM:SS.SSS is a valid time.',
            'format': 'S200', 'unit': 'h', 'value': 'RA'}, 'ron': {
            'comment': 'ron - DOUBLE - CCD readout noise (ADU). This parameter should be non-negative (Default value = 0.0)',
            'format': 'float', 'unit': 'adu', 'value': 9.2589}, 'imagey1': {
            'comment': 'imagey1 - INT - Image region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': 'pixel', 'value': 1}, 'obskeyf': {
            'comment': 'obskeyf - STRING - Value of the observation type header keyword for a flat frame.',
            'format': 'S200', 'unit': '', 'value': 'SKYFLAT'}, 'oscany1': {
            'comment': 'oscany1 - INT - Overscan region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': '', 'value': 1}, 'oscanx1': {
            'comment': 'oscanx1 - INT - Overscan region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': '', 'value': 1}, 'obskeys': {
            'comment': 'obskeys - STRING - Value of the observation type header keyword for a science frame.',
            'format': 'S200', 'unit': '', 'value': 'EXPOSE'},
        'maxval': {'comment': 'maxval - DOUBLE - Maximum useful pixel value in a raw image (ADU).',
                   'format': 'float', 'unit': 'adu', 'value': 140000.0}, 'obskeyb': {
            'comment': 'obskeyb - STRING - Value of the observation type header keyword for a bias frame.',
            'format': 'S200', 'unit': '', 'value': 'BIAS'}, 'obskeyd': {
            'comment': 'obskeyd - STRING - Value of the observation type header keyword for a dark frame.',
            'format': 'S200', 'unit': '', 'value': 'DARK'},
        'instrid': {'comment': 'instrid - STRING  - The instrument name.', 'format': 'S200', 'unit': '',
                    'value': 'fl15'}, 'imagex2': {
            'comment': 'imagex2 - INT - Image region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': 'pixel', 'value': 4096}, 'oscany2': {
            'comment': 'oscany2 - INT - Overscan region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': '', 'value': 1}, 'filtkey': {
            'comment': 'filtkey - STRING - Header keyword recording the filter employed. The value of this keyword should be a STRING.',
            'format': 'S200', 'unit': '', 'value': 'FILTER2'}, 'fov': {
            'comment': 'fov - DOUBLE - Field of view of the CCD camera (deg). This parameter should be positive and less than or equal to 90.0 (Default value = 0.1).',
            'format': 'float', 'unit': 'deg', 'value': 0.196}, 'oscanx2': {
            'comment': 'oscanx2 - INT - Overscan region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': '', 'value': 1},
        'objkey': {'comment': 'objkey - STRING  - Header keyword recording the object type.',
                   'format': 'S200', 'unit': '', 'value': 'OBJECT'}, 'imagey2': {
            'comment': 'imagey2 - INT - Image region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': 'pixel', 'value': 4096},
        'obskey': {'comment': 'obskey - STRING  - Header keyword recording the observation type.',
                   'format': 'S200', 'unit': '', 'value': 'OBSTYPE'}, 'imagex1': {
            'comment': 'imagex1 - INT - Image region bottom left hand corner (x1,y1) and top right hand corner (x2,y2) in pixel number coordinates.',
            'format': 'int', 'unit': 'pixel', 'value': 1}, 'expkey': {
            'comment': 'expkey - STRING - Header keyword recording the image exposure time in seconds. The value of this keyword should be a NUMBER with a non-negative value, or a STRING that represents a valid non-negative number.',
            'format': 'S200', 'unit': '', 'value': 'EXPTIME'}}

    assert len(inst_config) != 0
    assert inst_config == expected_output


def test_set_bad_pixel_mask_directory():
    setup = mock.MagicMock()

    reduction_metadata = stage0.create_or_load_the_reduction_metadata(setup, './',
                                                                      metadata_name='HanSolo.fits',
                                                                      log=None)
    stage0.set_bad_pixel_mask_directory(setup, reduction_metadata,
                                        bpm_directory_path='./MamaSam',
                                        verbose=False, log=None)

    assert reduction_metadata.data_architecture[1]['BPM_PATH'] == './MamaSam'


def test_open_an_image():
    setup = mock.MagicMock()

    image = stage0.open_an_image(setup, './', 'dummy.fits')

    assert image is None

    image = stage0.open_an_image(setup, '../tests/data/proc/ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip/data',
                                 'lsc1m005-fl15-20170418-0131-e91_cropped.fits')
    assert image is not None


def test_open_an_bad_pixel_mask():
    setup = mock.MagicMock()

    reduction_metadata = stage0.create_or_load_the_reduction_metadata(setup, './',
                                                                      metadata_name='HanSolo.fits',
                                                                      log=None)
    reduction_metadata.data_architecture[1]['BPM_PATH'] = './'
    bpm = stage0.open_an_bad_pixel_mask(reduction_metadata, 'lsc1m005-fl15-20170418-0131-e91_cropped.fits',
                                        bpm_index=2, verbose=False)

    os.remove('./HanSolo.fits')


def test_save_the_pixel_mask_in_image():
    setup = mock.MagicMock()

    reduction_metadata = stage0.create_or_load_the_reduction_metadata(setup, './',
                                                                      metadata_name='HanSolo.fits',
                                                                      log=None)
    reduction_metadata.data_architecture[1]['IMAGES_PATH'] = ['./']
    image_bad_pixel_mask = np.zeros((3, 3))
    image_bad_pixel_mask += 89
    image = fits.PrimaryHDU(image_bad_pixel_mask)
    hdulist = fits.HDUList([image])
    hdulist.writeto('Leia.fits', overwrite=True)

    bpm = stage0.construct_the_pixel_mask(image, image_bad_pixel_mask, [8],
                                          saturation_level=65535, low_level=0, log=None)

    stage0.save_the_pixel_mask_in_image(reduction_metadata, 'Leia.fits', bpm)

    test_image = fits.open('Leia.fits')

    assert np.allclose(test_image['pyDANDIA_PIXEL_MASK'].data, bpm)

    os.remove('Leia.fits')
    os.remove('HanSolo.fits')


def test_update_reduction_metadata_with_config_file():
    reduction_metadata = metadata.MetaData()

    pipeline_config = stage0.read_the_config_file('../../Config/', log=None)

    stage0.update_reduction_metadata_with_config_file(reduction_metadata, pipeline_config, log=None)

    assert len(reduction_metadata.reduction_parameters[1]) != 0


def test_parse_the_image_header():
    setup = mock.MagicMock()

    reduction_metadata = metadata.MetaData()

    pipeline_config = stage0.read_the_config_file('../../Config/', log=None)

    stage0.update_reduction_metadata_with_config_file(reduction_metadata, pipeline_config, log=None)
    inst_config_dictionnary = {'OBSTYPE': {"comment": "tres bon",
                                           "value": "OBJECT",
                                           "format": "S200",
                                           "unit": ""}}

    stage0.update_reduction_metadata_with_inst_config_file(reduction_metadata, inst_config_dictionnary, log=None)

    image = stage0.open_an_image(setup, '../tests/data/proc/ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip/data',
                                 'lsc1m005-fl15-20170418-0131-e91_cropped.fits')

    values = stage0.parse_the_image_header(reduction_metadata, image)

    assert values[0][1] == 'ROME-FIELD-02'


def test_update_reduction_metadata_headers_summary_with_new_images():
    setup = mock.MagicMock()

    reduction_metadata = metadata.MetaData()

    pipeline_config = stage0.read_the_config_file('../../Config/', log=None)

    stage0.update_reduction_metadata_with_config_file(reduction_metadata, pipeline_config, log=None)
    inst_config_dictionnary = {'OBSTYPE': {"comment": "tres bon",
                                           "value": "OBJECT",
                                           "format": "S200",
                                           "unit": ""}}

    stage0.update_reduction_metadata_with_inst_config_file(reduction_metadata, inst_config_dictionnary, log=None)
    reduction_metadata.data_architecture[1] = {'IMAGES_PATH': ['./']}

    image_bad_pixel_mask = np.zeros((3, 3))
    image_bad_pixel_mask += 89
    header = fits.Header(('OBJECT', 'HUNGRY'))
    header['OBJECT'] = 'NDG'

    image = fits.PrimaryHDU(image_bad_pixel_mask, header=header)
    hdulist = fits.HDUList([image])

    hdulist.writeto('Leia.fits', overwrite=True)

    stage0.update_reduction_metadata_headers_summary_with_new_images(setup,
                                                                     reduction_metadata,
                                                                     ['Leia.fits'], log=None)

    assert reduction_metadata.headers_summary[1]['IMAGES'][0] == 'Leia.fits'
    assert reduction_metadata.headers_summary[1]['OBSTYPE'][0] == 'NDG'
    os.remove('Leia.fits')


def test_construct_the_stamps():
    image = np.zeros((300, 300))
    image = fits.PrimaryHDU(image)

    status, report, stamps = stage0.construct_the_stamps(image, stamp_size=None, arcseconds_stamp_size=(60, 60),
                                                         pixel_scale=0.51,
                                                         number_of_overlaping_pixels=25, log=None)

    assert np.allclose(stamps, np.array([[0., 0., 142., 0., 142.],
                                         [1., 0., 142., 92., 259.],
                                         [2., 0., 142., 300., 376.],
                                         [3., 92., 259., 0., 142.],
                                         [4., 92., 259., 92., 259.],
                                         [5., 92., 259., 300., 376.],
                                         [6., 300., 376., 0., 142.],
                                         [7., 300., 376., 92., 259.],
                                         [8., 300., 376., 300., 376.]]))

    assert status == 'OK'
    assert report == 'Completed successfully'


def test_update_reduction_metadata_stamps():
    setup = mock.MagicMock()

    reduction_metadata = metadata.MetaData()

    image = np.zeros((300, 300))
    image = fits.PrimaryHDU(image)

    stage0.update_reduction_metadata_stamps(setup, reduction_metadata, image,
                                            stamp_size=None, arcseconds_stamp_size=(60, 60),
                                            pixel_scale=0.51, number_of_overlaping_pixels=25,
                                            log=None)

    expected_values = np.array([[0., 0., 142., 0., 142.],
                                [1., 0., 142., 92., 259.],
                                [2., 0., 142., 300., 376.],
                                [3., 92., 259., 0., 142.],
                                [4., 92., 259., 92., 259.],
                                [5., 92., 259., 300., 376.],
                                [6., 300., 376., 0., 142.],
                                [7., 300., 376., 92., 259.],
                                [8., 300., 376., 300., 376.]])

    assert reduction_metadata.stamps[0]['NAME'] == 'stamps'
    assert np.allclose(reduction_metadata.stamps[1]['PIXEL_INDEX'].data.astype(float), expected_values[:,0])
    assert np.allclose(reduction_metadata.stamps[1]['Y_MIN'].data.astype(float), expected_values[:,1])
    assert np.allclose(reduction_metadata.stamps[1]['Y_MAX'].data.astype(float), expected_values[:, 2])
    assert np.allclose(reduction_metadata.stamps[1]['X_MIN'].data.astype(float), expected_values[:, 3])
    assert np.allclose(reduction_metadata.stamps[1]['X_MAX'].data.astype(float), expected_values[:, 4])

