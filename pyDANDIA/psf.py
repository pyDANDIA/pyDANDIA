######################################################################
#
# psf.py - Module defining the PSF models.
# For model details see individual function descriptions.
#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

import abc
import collections
import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from astropy import visualization
from mpl_toolkits.mplot3d import Axes3D
from astropy.nddata import Cutout2D
from astropy.io import fits
from pyDANDIA import logs
import os
import sys
from pyDANDIA import convolution
import copy


class PSFModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = None
        self.model = None
        self.psf_parameters = None
        self.define_psf_parameters()

    @abc.abstractproperty
    def define_psf_parameters(self):
        pass

    @abc.abstractproperty
    def psf_model(self, star_data, parameters):
        pass

    @abc.abstractproperty
    def update_psf_parameters(self):
        pass

    @abc.abstractproperty
    def psf_guess(self):
        pass

    @abc.abstractproperty
    def get_FWHM(self):
        pass

    @abc.abstractproperty
    def get_parameters(self):
        pass


class Moffat2D(PSFModel):

    def psf_type(self):

        return 'Moffat2D'

    def define_psf_parameters(self):

        self.model = ['intensity', 'y_center', 'x_center', 'gamma', 'alpha']

        self.psf_parameters = collections.namedtuple('parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, None)

    def update_psf_parameters(self, parameters):

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, parameters[index])

    def psf_model(self, Y_star, X_star, parameters):

        self.update_psf_parameters(parameters)

        model = self.psf_parameters.intensity * (1 + ((X_star - self.psf_parameters.x_center) ** 2 + \
                                                      (
                                                              Y_star - self.psf_parameters.y_center) ** 2) / self.psf_parameters.gamma ** 2) ** (
                    -self.psf_parameters.alpha)

        return model

    def psf_model_star(self, Y_star, X_star, star_params=[]):

        params = self.get_parameters()

        for i in range(0, len(star_params), 1):
            params[i] = star_params[i]

        model = self.psf_model(Y_star, X_star, params)

        return model

    def psf_guess(self):

        # gamma, alpha
        return [2.0, 2.0]

    def get_FWHM(self, gamma, alpha, pix_scale):

        fwhm = gamma * 2 * (2 ** (1 / alpha) - 1) ** 0.5 * pix_scale
        return fwhm

    def get_parameters(self):

        params = []

        for par in self.model:
            params.append(getattr(self.psf_parameters, par))

        return params

    def calc_flux(self, Y_star, X_star):

        model = self.psf_model_star(Y_star, X_star)

        flux = model.sum()

        if flux > 0:

            flux_err = np.sqrt(flux)

        else:

            flux_err = -99.999

        return flux, flux_err

    def calc_flux_with_kernel(self, Y_star, X_star, kernel):

        model = self.psf_model_star(Y_star, X_star)

        model_with_kernel = convolution.convolve_image_with_a_psf(model, kernel, fourrier_transform_psf=None,
                                                                  fourrier_transform_image=None,
                                                                  correlate=None, auto_correlation=None)
        flux = model_with_kernel.sum()

        flux_err = np.sqrt(flux)
        
        return flux, flux_err
    
    def normalize_psf(self,psf_diameter):
        
        Y_data, X_data = np.indices((int(psf_diameter),int(psf_diameter)))
    
        (f_total,ferr) = self.calc_flux(Y_data,X_data)
        
        self.psf_parameters.intensity = self.psf_parameters.intensity / f_total
    
    def calc_optimized_flux(self,ref_flux,var_sky,y_star,x_star,
                            Y_star,X_star,gain,
                            residuals):
        """Method to compute the flux and flux error in a star's PSF fit, 
        following the method of Naylor (1997).
        
        :param float ref_flux: Reference flux for the image
        :param float var_sky: Varience in the sky background for the image
        :param float x_star, ystar: Centroid of the star in the X_star, Y_star 
                                    grid coordinates
        :param np.indices X_star,Y_star: Indices of the pixel size of the PSF
        :param float gain: Detector gain in e-/ADU
        :param array residuals: Data SECTION - sky background SECTION
        
        Note the input data and sky model arrays should be pre-cut to match the
        size of the PSF
        
        Returns:
        :param float flux: Flux measure for the star
        :param float flux_err: Uncertainty on the flux measurement
        """
        
        hdu = fits.PrimaryHDU(residuals)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto('optimize_flux_input_data.fits', overwrite=True)
    
        Pij = self.psf_model_star(Y_star, X_star)
        
        hdu = fits.PrimaryHDU(Pij)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto('optimize_flux_Pij.fits', overwrite=True)
    
        Vij = var_sky + (ref_flux * Pij)/np.sqrt(gain)
        
        hdu = fits.PrimaryHDU(Vij)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto('optimize_flux_Vij.fits', overwrite=True)
        
        Wij = (Pij / Vij) / ((Pij*Pij)/Vij).sum()
        
        hdu = fits.PrimaryHDU(Wij)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto('optimize_flux_Wij.fits', overwrite=True)
        
        Fij = (Wij * residuals)
        flux = Fij.sum()
        
        hdu = fits.PrimaryHDU(Fij)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto('optimize_flux_fluximage.fits', overwrite=True)
        
        var_psf = (Wij*Wij * Vij).sum()
        flux_err = np.sqrt( var_psf + flux )
        
        return flux, flux_err, Fij
        
class Gaussian2D(PSFModel):

    def psf_type(self):

        return 'Gaussian2D'

    def define_psf_parameters(self):

        self.model = ['intensity', 'y_center',
                      'x_center', 'width_y', 'width_x']

        self.psf_parameters = collections.namedtuple('parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, None)

    def update_psf_parameters(self, parameters):

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, parameters[index])

    def psf_model(self, Y_star, X_star, parameters):

        self.update_psf_parameters(parameters)

        model = self.psf_parameters.intensity * np.exp(
            -(((X_star - self.psf_parameters.x_center) / self.psf_parameters.width_x) ** 2 + \
              ((Y_star - self.psf_parameters.y_center) / self.psf_parameters.width_y) ** 2) / 2)

        return model

    def psf_model_star(self, Y_star, X_star, star_params=[]):

        params = self.get_parameters()

        for i in range(0, len(star_params), 1):
            params[i] = star_params[i]

        model = self.psf_model(Y_star, X_star, params)

        return model

    def psf_guess(self):

        # width_x, width_y
        return [1.0, 1.0]

    def get_FWHM(self, width_x, width_y, pixel_scale):

        # fwhm = (width_x + width_y) / 2 * 2 * (2 * np.log(2))**0.5 * pixel_scale
        fwhm = (width_x + width_y) * 1.1774100225154747 * pixel_scale

        return fwhm

    def get_parameters(self):

        params = []

        for par in self.model:
            params.append(getattr(self.psf_parameters, par))

        return params

    def calc_flux(self, Y_star, X_star):

        model = self.psf_model_star(Y_star, X_star)

        flux = model.sum()

        flux_err = np.sqrt(flux)

        return flux, flux_err
    
    def normalize_psf(self,psf_diameter):
        
        Y_data, X_data = np.indices((int(psf_diameter),int(psf_diameter)))
    
        (f_total,ferr) = self.calc_flux(Y_data,X_data)
        
        self.psf_parameters.intensity = self.psf_parameters.intensity / f_total
        
class BivariateNormal(PSFModel):
    def psf_type():

        return 'BivariateNormal'

    def define_psf_parameters(self):

        self.model = ['intensity', 'y_center',
                      'x_center', 'width_y', 'width_x', 'corr_xy']

        self.psf_parameters = collections.namedtuple('parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, None)

    def update_psf_parameters(self, parameters):

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, parameters[index])

    def psf_model(self, Y_star, X_star, parameters):

        self.update_psf_parameters(parameters)

        zeta = ((X_star - self.psf_parameters.x_center) / self.psf_parameters.width_x) ** 2 - (
                2 * self.psf_parameters.corr_xy * (X_star - self.psf_parameters.x_center) * \
                (Y_star - self.psf_parameters.y_center)) / (
                       self.psf_parameters.width_x * self.psf_parameters.width_y) + \
               ((Y_star - self.psf_parameters.y_center) / self.psf_parameters.width_y) ** 2

        model = self.psf_parameters.intensity * \
                np.exp(-zeta / (2 * (1 - self.psf_parameters.corr_xy * self.psf_parameters.corr_xy)))

        return model

    def psf_model_star(self, Y_star, X_star, star_params=[]):

        params = self.get_parameters()

        for i in range(0, len(star_params), 1):
            params[i] = star_params[i]

        model = self.psf_model(Y_star, X_star, params)

        return model

    def psf_guess(self):

        # width_x, width_y, corr_xy
        return [1.0, 1.0, 0.7]

    def get_FWHM(self, width_x, width_y, pixel_scale):

        # fwhm = (width_x + width_y) / 2 * 2 * (2 * np.log(2))**0.5 * pixel_scale
        fwhm = (width_x + width_y) * 1.1774100225154747 * pixel_scale

        return fwhm

    def get_parameters(self):

        params = []

        for par in self.model:
            params.append(getattr(self.psf_parameters, par))

        return params

    def calc_flux(self, Y_star, X_star):

        model = self.psf_model_star(Y_star, X_star)

        flux = model.sum()

        flux_err = np.sqrt(flux)

        return flux, flux_err
    
    def normalize_psf(self,psf_diameter):
        
        Y_data, X_data = np.indices((int(psf_diameter),int(psf_diameter)))
    
        (f_total,ferr) = self.calc_flux(Y_data,X_data)
        
        self.psf_parameters.intensity = self.psf_parameters.intensity / f_total
        
class Lorentzian2D(PSFModel):

    def psf_type():

        return 'Lorentzian2D'

    def define_psf_parameters(self):

        self.model = ['intensity', 'y_center', 'x_center', 'gamma']
        self.psf_parameters = collections.namedtuple('parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, None)

    def update_psf_parameters(self, parameters):

        for index, key in enumerate(self.model):
            setattr(self.psf_parameters, key, parameters[index])

    def psf_model(self, Y_star, X_star, parameters):

        self.update_psf_parameters(parameters)

        model = self.psf_parameters.intensity * (
                self.psf_parameters.gamma / ((X_star - self.psf_parameters.x_center) ** 2 + \
                                             (
                                                     Y_star - self.psf_parameters.y_center) ** 2 + self.psf_parameters.gamma ** 2) ** (
                    1.5))
        return model

    def psf_model_star(self, Y_star, X_star, star_params=[]):

        params = self.get_parameters()

        for i in range(0, len(star_params), 1):
            params[i] = star_params[i]

        model = self.psf_model(Y_star, X_star, params)

        return model

    def psf_guess(self):

        # width_x
        return [1.0]

    def get_FWHM(self, gamma, pixel_scale):

        fwhm = 2 * gamma * pixel_scale

        return fwhm

    def get_parameters(self):

        params = []

        for par in self.model:
            params.append(getattr(self.psf_parameters, par))

        return params

    def calc_flux(self, Y_star, X_star):

        model = self.psf_model_star(Y_star, X_star)

        flux = model.sum()

        flux_err = np.sqrt(flux)

        return flux, flux_err
    
    def normalize_psf(self,psf_diameter):
        
        Y_data, X_data = np.indices((int(psf_diameter),int(psf_diameter)))
    
        (f_total,ferr) = self.calc_flux(Y_data,X_data)
        
        self.psf_parameters.intensity = self.psf_parameters.intensity / f_total
        
class BackgroundModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = None
        self.model = None
        self.background_parameters = None
        self.define_background_parameters()
        self.varience = None
        
    @abc.abstractproperty
    def define_background_parameters(self, background_data, parameters):
        pass

    @abc.abstractproperty
    def update_background_parameters(self, background_data, parameters):
        pass

    @abc.abstractproperty
    def background_model(self, background_data, parameters):
        pass

    @abc.abstractproperty
    def update_background_parameters(self):
        pass

    @abc.abstractproperty
    def background_guess(self):
        pass

    @abc.abstractproperty
    def get_parameters(self):
        pass


class ConstantBackground(BackgroundModel):

    def background_type(self):

        return 'Constant'

    def define_background_parameters(self):

        self.model = ['constant']
        self.background_parameters = collections.namedtuple(
            'parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.background_parameters, key, None)

    def update_background_parameters(self, parameters):

        self.background_parameters = collections.namedtuple(
            'parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.background_parameters, key, parameters[index])

    def background_model(self, data_shape, parameters=None):

        if parameters != None:
            self.update_background_parameters(parameters)

        model = np.ones(data_shape) * \
            self.background_parameters.constant
            
        return model

    def background_guess(self):

        # background constant
        return [0]

    def get_parameters(self):

        params = []

        for par in self.model:
            params.append(getattr(self.background_parameters, par))

        return params

class GradientBackground(BackgroundModel):

    def background_type(self):

        return 'Gradient'

    def define_background_parameters(self):

        self.model = ['a0', 'a1', 'a2']
        self.background_parameters = collections.namedtuple(
            'parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.background_parameters, key, None)

    def update_background_parameters(self, parameters):

        self.background_parameters = collections.namedtuple(
            'parameters', self.model)
        
        for index, key in enumerate(self.model):
            setattr(self.background_parameters, key, parameters[key])

    def background_model(self, data_shape, parameters=None):
        
        if parameters != None:
            self.update_background_parameters(parameters)
        
        
        Y_data, X_data = np.indices(data_shape)
    
        model = np.ones(data_shape) * self.background_parameters.a0
        
        model = model + ( self.background_parameters.a1 * X_data ) + \
                    + ( self.background_parameters.a2 * Y_data )
        
        return model

    def background_guess(self):
        """Method to return an initial estimate of the parameters of a 2D
        gradient sky background model.  The parameters returned represent 
        a flat, constant background of zero."""

        return [0.0, 0.0, 0.0]

    def get_parameters(self):

        params = []

        for par in self.model:
            params.append(getattr(self, par))

        return params


class QuadraticBackground(BackgroundModel):

    def background_type(self):

        return 'Quadratic'

    def define_background_parameters(self):

        self.model = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5']
        self.background_parameters = collections.namedtuple(
            'parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.background_parameters, key, None)

    def update_background_parameters(self, parameters):

        self.background_parameters = collections.namedtuple(
            'parameters', self.model)

        for index, key in enumerate(self.model):
            setattr(self.background_parameters, key, parameters[index])

    def background_model(self, Y_background, X_background, parameters):

        self.update_background_parameters(parameters)

        model = np.ones(Y_background.shape) * self.background_parameters.a0
        model = model + (self.background_parameters.a1 * X_background) + \
                + (self.background_parameters.a2 * Y_background) + (
                        self.background_parameters.a3 * X_background * Y_background) + (
                            self.background_parameters.a4 * X_background * X_background) + (
                            self.background_parameters.a5 * Y_background * Y_background)

        return model

    def background_guess(self):
        """Method to return an initial estimate of the parameters of a 2D
        gradient sky background model.  The parameters returned represent
        a flat, constant background of zero."""

        return [0.0, 0.0, 0.0, 0.0, 0.0,0.0]

    def get_parameters(self):

        params = []

        for par in self.model:
            params.append(getattr(self, par))

        return params


class Image(object):

    def __init__(self, full_data, psf_model):

        self.full_data = full_data
        self.residuals = np.zeros(full_data.shape)

        self.model = np.zeros(self.full_data.shape)
        x_data = np.arange(0, self.full_data.shape[1])
        y_data = np.arange(0, self.full_data.shape[0])

        self.X_data, self.Y_data = np.meshgrid(x_data, y_data)

        if psf_model == 'Moffat2D':
            self.psf_model = Moffat2D()

        if psf_model == 'Gaussian2D':
            self.psf_model = Gaussian2D()

        if psf_model == 'BivariateNormal':
            self.psf_model = BivariateNormal()

        if psf_model == 'Lorentzian2D':
            self.psf_model = Gaussian2D()

    def inject_psf_in_stars(self, model, parameters):

        X_data = self.X_data
        Y_data = self.Y_data

        psf_width = parameters[-1]

        for index in xrange(len(parameters[0])):
            index_star = (int(parameters[1][index]), int(parameters[2][index]))

            params = [parameters[0][index], parameters[1][index],
                      parameters[2][index], parameters[3], parameters[4]]
            X_star = X_data[index_star[0] - psf_width:index_star[0] +
                                                      psf_width, index_star[1] - psf_width:index_star[1] + psf_width]
            Y_star = Y_data[index_star[0] - psf_width:index_star[0] +
                                                      psf_width, index_star[1] - psf_width:index_star[1] + psf_width]

            stamp = self.psf_model.psf_model(Y_star, X_star, *params)

            model[index_star[0] - psf_width:index_star[0] + psf_width,
            index_star[1] - psf_width:index_star[1] + psf_width] += stamp

        return model

    def image_model(self, psf_parameters, background_parameters=0):

        model = np.zeros(self.full_data.shape)

        model = self.inject_psf_in_stars(model, psf_parameters)

        model += background_parameters

        self.model = model

    def image_residuals(self, psf_parameters, background_parameters=0):

        self.image_model(psf_parameters, background_parameters)

        self.residuals = self.full_data - self.model

    def stars_guess(self, star_positions, star_width=10):

        x_centers = []
        y_centers = []
        intensities = []

        for index in xrange(len(star_positions)):
            data = self.full_data[int(star_positions[0]) - star_width:int(star_positions[0]) + star_width,
                   int(star_positions[1]) - star_width:int(star_positions[1]) + star_width]

            intensities.append(data.max)
            x_centers.append(star_positions[1])
            y_centers.append(star_positions[0])

        return [np.array(intensities), np.array(y_centers), np.array(x_centers)]


def fit_background(data, Y_data, X_data, mask, background_model='Constant'):
    if background_model == 'Constant':
        back_model = ConstantBackground()

    if background_model == 'Gradient':
        back_model = GradientBackground()

    if background_model == 'Quadratic':
        back_model = QuadraticBackground()

    guess_back = back_model.background_guess()

    guess = guess_back

    fit = optimize.leastsq(error_background_fit_function, guess, args=(
        data, back_model, Y_data, X_data, mask), full_output=1)

    return fit


def error_background_fit_function(params, data, background, Y_data, X_data, mask):
    back_params = params

    back_model = background.background_model(Y_data, X_data, back_params)

    residuals = (data - back_model)[mask]

    return residuals

def fit_star(data, Y_data, X_data, psf_model='Moffat2D', background_model='Constant'):
    psf_model = get_psf_object(psf_model)

    if background_model == 'Constant':
        back_model = ConstantBackground()

    guess_psf = [data.max(), Y_data[int(len(Y_data[:, 0]) / 2), 0],
                 X_data[0, int(len(Y_data[0, :]) / 2)]] + psf_model.psf_guess()

    guess_back = back_model.background_guess()

    guess = guess_psf + guess_back

    fit = optimize.leastsq(error_star_fit_function, guess, args=(
        data, psf_model, back_model, Y_data, X_data), full_output=1)

    return fit


def error_star_fit_function(params, data, psf, background, Y_data, X_data):
    psf_params = params[:len(psf.psf_parameters._fields)]
    back_params = params[len(psf.psf_parameters._fields):]

    psf_model = psf.psf_model(Y_data, X_data, psf_params)
    back_model = background.background_model(Y_data.shape, parameters=back_params)

    residuals = np.ravel(data - psf_model - back_model)

    return residuals


def fit_existing_psf_stamp(setup, x_cen, y_cen, psf_radius, 
                            input_psf_model, psf_image_data, psf_sky_bkgd,
                            centroiding=True, diagnostics=False):
    """Function to fit an existing PSF and sky model to a star at a given 
    location in an image, optimizing only the peak intensity of the PSF rather 
    than all parameters.
    
    :param SetUp setup: Fundamental reduction parameters
    :param array data: image data to be fitted
    :param float x_cen: the x-pixel location of the PSF center in the stamp
    :param float y_cen: the y-pixel location of the PSF center in the stamp
    :param float psf_radius: the radius of data to fit the PSF to
    :param PSFModel input_psf_model: existing psf model
    :param array psf_image_data: image stamp surrounding the PSF
    :param BackgroundModel psf_sky_bkgd: image sky background surrounding the PSF
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    :param boolean diagnostics: optional switch for diagnostic output
    
    Returns
    
    :param PSFModel fitted_model: PSF model for the star with optimized intensity
    """
                                
    psf_model = get_psf_object(input_psf_model.psf_type())
    psf_model.update_psf_parameters(input_psf_model.get_parameters())
    
    # Recenter the PSF temporarily to the middle of the stamp to be fitted
    # NOTE PSF parameters in order intensity, Y, X
    #psf_params = psf_model.get_parameters()
    #psf_params[1] = stamp_centre[1]
    #psf_params[2] = stamp_centre[0]
    #psf_model.update_psf_parameters(psf_params)

    Y_data, X_data = np.indices(psf_image_data.shape)
    
    if diagnostics:
        
        hdu = fits.PrimaryHDU(stamps[0].data)
        
        hdulist = fits.HDUList([hdu])

        file_path = os.path.join(setup.red_dir,'ref',\
        'psf_star_stamp_'+str(round(x_cen,0))+'_'+str(round(y_cen,0))+'.fits')

        hdulist.writeto(file_path,overwrite=True)

    if centroiding:
        
        init_par = [ psf_model.get_parameters()[0], y_cen, x_cen ]
        
    else:
        
        init_par = [ psf_model.get_parameters()[0] ]

    fit = optimize.leastsq(error_star_fit_existing_model, init_par, 
        args=(psf_image_data, psf_model, psf_sky_bkgd, Y_data, X_data), 
        full_output=1)

    fitted_model = get_psf_object( psf_model.psf_type() )
    
    psf_params = psf_model.get_parameters()
    psf_params[0] = fit[0][0]
    
    if centroiding:
        psf_params[1] = fit[0][1]
        psf_params[2] = fit[0][2]
    
    fitted_model.update_psf_parameters(psf_params)
    
    good_fit = check_fit_quality(setup,psf_image_data,psf_sky_bkgd,fitted_model)
    
    return fitted_model, good_fit

def fit_star_existing_model(setup, data, x_cen, y_cen, psf_radius, 
                            input_psf_model, psf_sky_bkgd,
                            centroiding=False,
                            diagnostics=False):
    """Function to fit an existing PSF and sky model to a star at a given 
    location in an image, optimizing only the peak intensity of the PSF rather 
    than all parameters.
    
    :param SetUp setup: Fundamental reduction parameters
    :param array data: image data to be fitted
    :param float x_cen: the x-pixel location of the PSF to be fitted in the 
                        coordinates of the image data array
    :param float x_cen: the x-pixel location of the PSF to be fitted in the 
                        coordinates of the image data array
    :param float psf_radius: the radius of data to fit the PSF to
    :param PSFModel input_psf_model: existing psf model
    :param array psf_sky_bkgd: sky background image data array for the region
                                of the PSF
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    :param boolean diagnostics: optional switch for diagnostic output
    
    Returns
    
    :param PSFModel fitted_model: PSF model for the star with optimized intensity
    """

    psf_model = get_psf_object(input_psf_model.psf_type())
    psf_model.update_psf_parameters(input_psf_model.get_parameters())
    
    #stamp_dims = (2.0*psf_radius, 2.0*psf_radius)
    
    #stamps = cut_image_stamps(setup, data, np.array([[x_cen,y_cen]]), 
    #                          stamp_dims, log=None, 
    #                            over_edge=True)
    
    #stamp_centre = ( (x_cen - stamps[0].xmin_original), 
    #                (y_cen - stamps[0].ymin_original) )
    
    if diagnostics:
        
        hdu = fits.PrimaryHDU(data)
        
        hdulist = fits.HDUList([hdu])

        file_path = os.path.join(setup.red_dir,'ref',\
        'fit_star_stamp_'+str(round(x_cen,0))+'_'+str(round(y_cen,0))+'.fits')
        hdulist.writeto(file_path, overwrite=True)

    Y_data, X_data = np.indices(data.shape)

    psf_params = psf_model.get_parameters()
    psf_params[1] = y_cen
    psf_params[2] = x_cen
    psf_model.update_psf_parameters(psf_params)
    
    if centroiding:
        init_par = [ psf_model.get_parameters()[0], 
                    y_cen, x_cen ]
    else:
        init_par = [ psf_model.get_parameters()[0] ]
        
    fit = optimize.leastsq(error_star_fit_existing_model, init_par, 
        args=(data, psf_model, psf_sky_bkgd, Y_data, X_data), 
        full_output=1)
    
    fitted_model = get_psf_object(psf_model.psf_type())

    psf_params = psf_model.get_parameters()
    psf_params[0] = fit[0][0]

    if centroiding:
        psf_params[1] = fit[0][1]
        psf_params[2] = fit[0][2]

    fitted_model.update_psf_parameters(psf_params)

    if diagnostics:
        Y_data, X_data = np.indices(data.shape)
        
        pars = fitted_model.get_parameters()

        model_data = fitted_model.psf_model_star(Y_data, X_data, star_params=pars)

        hdu = fits.PrimaryHDU(model_data)

        hdulist = fits.HDUList([hdu])

        file_path = os.path.join(setup.red_dir,'ref',\
        'fit_star_model_stamp_'+str(round(x_cen,0))+'_'+str(round(y_cen,0))+'.fits')
        
        hdulist.writeto(file_path,overwrite=True)
    
    psf_params = fitted_model.get_parameters()
    psf_params[0] = fit[0][0]
    psf_params[1] = psf_params[1]
    psf_params[2] = psf_params[2]
    
    fitted_model.update_psf_parameters(psf_params)
    
    good_fit = check_fit_quality(setup, data, psf_sky_bkgd, fitted_model)
    
    return fitted_model, good_fit


def fit_star_existing_model_with_kernel(setup, data, x_cen, y_cen, psf_radius,
                                        input_psf_model, sky_model, kernel,
                                        centroiding=True,
                                        diagnostics=False):
    """Function to fit an existing PSF and sky model to a star at a given 
    location in an image, optimizing only the peak intensity of the PSF rather 
    than all parameters.
    
    :param SetUp setup: Fundamental reduction parameters
    :param array data: image data to be fitted
    :param float x_cen: the x-pixel location of the PSF to be fitted in the 
                        coordinates of the image
    :param float x_cen: the x-pixel location of the PSF to be fitted in the 
                        coordinates of the image
    :param float psf_radius: the radius of data to fit the PSF to
    :param PSFModel input_psf_model: existing psf model
    :param BackgroundModel sky_model: existing model for the image sky background
    :param array_like kernel: the kernel data in 2d np.array
    :param boolean centroiding: Switch to (dis)-allow re-fitting of each star's
                                x, y centroid.  Default=allowed (True)
    :param boolean diagnostics: optional switch for diagnostic output
    
    Returns
    
    :param PSFModel fitted_model: PSF model for the star with optimized intensity
    """

    psf_model = get_psf_object(input_psf_model.psf_type())
    psf_model.update_psf_parameters(input_psf_model.get_parameters())

    stamp_dims = (2.0 * psf_radius, 2.0 * psf_radius)

    stamps = data

    stamp_centre = (stamps.shape[0] / 2,
                    stamps.shape[1] / 2)

    # if diagnostics:

    # hdu = fits.PrimaryHDU(stamps[0].data)

    # hdulist = fits.HDUList([hdu])

    # file_path = os.path.join(setup.red_dir,'ref',\
    # 'psf_star_stamp_'+str(round(x_cen,0))+'_'+str(round(y_cen,0))+'.fits')

    # hdulist.writeto(file_path,overwrite=True)

    Y_data, X_data = np.indices(stamps.shape)

    sky_bkgd = sky_model.background_model(X_data, Y_data,
                                          sky_model.get_parameters())

    if centroiding:

        init_par = [psf_model.get_parameters()[0],
                    stamp_centre[1], stamp_centre[0]]
    else:

        init_par = [psf_model.get_parameters()[0]]

    fit = optimize.leastsq(error_star_fit_existing_model_with_kernel, init_par,
                           args=(stamps, psf_model, sky_bkgd, Y_data, X_data, kernel),
                           full_output=1)

    fitted_model = get_psf_object(psf_model.psf_type())

    psf_params = psf_model.get_parameters()
    psf_params[0] = fit[0][0]

    if centroiding:
        psf_params[1] = fit[0][1]
        psf_params[2] = fit[0][2]

    fitted_model.update_psf_parameters(psf_params)
    
    good_fit = check_fit_quality(setup, data, sky_bkgd, fitted_model)

    return fitted_model, good_fit


def error_star_fit_existing_model(params, data, psf_model, sky_bkgd,
                                  Y_data, X_data):
    sky_subtracted_data = data - sky_bkgd
    
    psf_image = psf_model.psf_model_star(Y_data, X_data, star_params=params)
    
    residuals = np.ravel(sky_subtracted_data - psf_image)

    return residuals


def error_star_fit_existing_model_with_kernel(params, data, psf_model, sky_bkgd,
                                              Y_data, X_data, kernel):
    sky_subtracted_data = data - sky_bkgd

    psf_image = psf_model.psf_model_star(Y_data, X_data, star_params=params)

    psf_image_with_kernel = convolution.convolve_image_with_a_psf(psf_image, kernel, fourrier_transform_psf=None,
                                                                  fourrier_transform_image=None,
                                                                  correlate=None, auto_correlation=None)

    residuals = np.ravel(sky_subtracted_data - psf_image_with_kernel)

    return residuals


def check_fit_quality(setup, psf_stamp_data, sky_model, fitted_model):
    """Function to sanity check the quality of the photometric fit"""

    good_fit = True

    model_pars = fitted_model.get_parameters()

    max_peak_flux = psf_stamp_data.max() + 0.1 * psf_stamp_data.max()

    if model_pars[0] > max_peak_flux or model_pars[0] <= 0.0:
        
        good_fit = False
        
    return good_fit


def fit_multiple_stars(data_stamps, Y_data, X_data, psf_model='Moffat2D', background_model='Constant'):
    model_fits = []
    for stamp in data_stamps:
        model_fits.append(fit_star(stamp, Y_data, X_data,
                                   psf_model, background_model))

    return model_fits


def plot3d(xdata, ydata, zdata):
    '''
    Plots 3D data.
    '''
    fig = plt.figure()
    ax1 = Axes3D(fig)
    ax1.plot_wireframe(xdata, ydata, zdata, alpha=0.5)
    ax1.plot_surface(xdata, ydata, z, alpha=0.2)
    cset = ax1.contourf(xdata, ydata, zdata, zdir='z',
                        offset=min(z.flatten()), alpha=0.2)
    cset = ax1.contourf(xdata, ydata, zdata, zdir='x',
                        offset=min(x[0]), alpha=0.3)
    cset = ax1.contourf(xdata, ydata, zdata, zdir='y',
                        offset=max(y[-1]), alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.show()


def build_psf(setup, reduction_metadata, log, image, ref_star_catalog, 
              sky_model, psf_diameter, diagnostics=True):
    """Function to build a PSF model based on the PSF stars
    selected from a reference image."""

    status = 'OK'

    log.info('Building a PSF model based on the reference image')

    # Cut large stamps around selected PSF stars
    psf_diameter = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]

    psf_model_type = 'Moffat2D'

    logs.ifverbose(log, setup, ' -> Applying PSF size=' + str(psf_diameter))

    idx = np.where(ref_star_catalog[:, 15] == 1.0)
    psf_idx = ref_star_catalog[idx[0], 0]
    psf_star_centres = ref_star_catalog[idx[0], 1:3]

    if len(psf_star_centres) == 0:
        status = 'ERROR: No PSF stars selected'

        log.info(status)

        return None, status
    
    log.info('Cutting stamps for '+str(len(psf_star_centres))+' PSF stars')
    
    stamp_dims = (int(psf_diameter)*4, int(psf_diameter)*4)

    logs.ifverbose(log, setup, ' -> Stamp dimensions=' + repr(stamp_dims))

    stamps = cut_image_stamps(setup, image, psf_star_centres,
                              stamp_dims, log=log)

    # Combine stamps into a single, high-signal-to-noise PSF
    master_stamp = coadd_stamps(setup, stamps, log, diagnostics=diagnostics)

    if diagnostics:
        output_stamp_image(master_stamp.data,
                           os.path.join(setup.red_dir, 'ref',
                                        'initial_psf_master_stamp.png'))

    # Build an initial PSF: fit a PSF model to the high S/N stamp
    init_psf_model = fit_psf_model(setup,log,psf_model_type, psf_diameter,
                                   sky_model.background_type(),
                                    master_stamp, diagnostics=diagnostics)
    init_psf_model.normalize_psf(psf_diameter)
    
    if diagnostics:
        psf_image = generate_psf_image(init_psf_model.psf_type(),
                                       init_psf_model.get_parameters(),
                                       stamp_dims, psf_diameter)

        output_fits(psf_image, 
                    os.path.join(setup.red_dir,'ref','initial_psf_model.fits'))
    
    clean_stamps = subtract_companions_from_psf_stamps(setup, 
                                        reduction_metadata, log, 
                                        ref_star_catalog, psf_idx, stamps,
                                        psf_star_centres,
                                        init_psf_model,sky_model,psf_diameter,
                                        diagnostics=diagnostics)
    
    # Re-combine the companion-subtracted stamps to re-generate the 
    # high S/N stamp
    master_stamp = coadd_stamps(setup, clean_stamps, log,
                                diagnostics=diagnostics)

    if diagnostics:
        output_fits(master_stamp.data, 
                    os.path.join(setup.red_dir,'ref','final_psf_master_stamp.fits'))
                
    # Re-build the final PSF by fitting a PSF model to the updated high 
    # S/N stamp
    psf_model = fit_psf_model(setup,log,psf_model_type,psf_diameter,
                                   sky_model.background_type(),
                                    master_stamp, diagnostics=diagnostics)
    psf_model.normalize_psf(psf_diameter)
    
    psf_image = generate_psf_image(psf_model.psf_type(),
                                   psf_model.get_parameters(),
                                    stamp_dims, psf_diameter)

    header = fits.Header()
    psf_params = psf_model.get_parameters()
    for i,key in enumerate(psf_model.model):    
        header[key[0:8].upper()] = psf_params[i]
    header['PSFTYPE'] = psf_model.psf_type()
    
    
    output_fits_model(psf_image,header, 
                os.path.join(setup.red_dir,'ref','psf_model.fits'))
                                     
    log.info('Completed build of PSF model with status '+status)
    
    return psf_model, status

def output_fits_model(image_data,header, file_path):
    """Function to output a FITS image of the given data"""
    
    hdu = fits.PrimaryHDU( image_data,header )
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_path,overwrite=True)
       

def output_fits(image_data, file_path):
    """Function to output a FITS image of the given data"""
    
    hdu = fits.PrimaryHDU( image_data )
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_path,overwrite=True)
       
    
def cut_image_stamps(setup, image, stamp_centres, stamp_dims, log=None, 
                     over_edge=False):
    """Function to extract a set of stamps (2D image sections) at the locations
    given and with the dimensions specified in pixels.
    
    No stamp will be returned for stars that are too close to the edge of the
    frame, that is, where the stamp would overlap with the edge of the frame. 
    
    :param SetUp object setup: Fundamental reduction parameters
    :param array image: the image data array from which to take stamps
    :param list psf_idx: the indices of the PSF stars in the ref_star_catalog
    :param array stamp_centres: 2-col array with the x, y centres of the stamps
    :param tuple stamp_dims: the width and height of the stamps
    :param boolean over_edge: switch to allow the stamps to be smaller than
                            specified if a stamp_centre lies close to the edge
                            of the image.  Code will then return a Cutout2D of 
                            that portion of the image within the stamp
                            boundaries.
    Returns
    
    :param list Cutout2D objects
            If a PSF star is too close to the edge to have a complete stamp
            cut for it, this list will contain a None entry for that PSF star
    """

    if log != None:
        log.info('Cutting PSF stamp images')

    stamps = []

    for j in range(0, len(stamp_centres), 1):
        xcen = stamp_centres[j, 0]
        ycen = stamp_centres[j, 1]

        corners = calc_stamp_corners(xcen, ycen, stamp_dims[1], stamp_dims[0],
                                     image.shape[1], image.shape[0],
                                     over_edge=over_edge)

        logs.ifverbose(log, setup, str(j) + 'th PSF Star at (' + str(xcen) + ', ' + str(ycen) +
                       ') has stamp x,y ranges: ' + repr(corners))

        if None not in corners:

            dxc = corners[1] - corners[0]
            dyc = corners[3] - corners[2]

            if over_edge == False and \
                    dxc == stamp_dims[1] and dyc == stamp_dims[0]:

                cutout = Cutout2D(image, (xcen, ycen), (dxc, dyc), copy=True)

                stamps.append(cutout)

            elif over_edge == True:

                cutout = Cutout2D(image, (xcen, ycen), (dxc, dyc), copy=True)

                stamps.append(cutout)

        else:

            stamps.append(None)

    if log != None:
        log.info('Made stamps for ' + str(len(stamps)) + ' out of ' + \
                 str(len(stamp_centres)) + ' locations')

    return stamps


def calc_stamp_corners(xcen, ycen, dx, dy, maxx, maxy, over_edge=False,
                       diagnostics=False):
    """Function to calculate the pixel coordinates of the x- and y- ranges 
    spanned by a box in an image or an image stamps.
    
    :param float xcen: x-pixel location of the box centre in the wider image
                        reference frame
    :param float ycen: y-pixel location of the box centre in the wider image
                        reference frame
    :param int dx: full width of the box in the x-direction
    :param int dy: full width of the box in the y-direction
    :param int maxx: Maximum possible x-dimension, bounded by the edge of the 
                    original image
    :param int maxy: Maximum possible y-dimension, bounded by the edge of the 
                    original image
    :param boolean over_edge: If True, allow the box to overlap the edge of the
                    original image by returning the ranges bounded by the edges
                    of the original image.  Default=False
    :param boolean diagnostics: Switch for debugging output
    
    Return:
    
    :param tuple corners: x- and y-pixel ranges spanned by the box;
                        (xmin, xmax, ymin, ymax)
                        If over_edge=False, a tuple of None entries will be
                        returned if a box intersects any edge of the frame.
    """
    
    x = int(xcen)
    y = int(ycen)
    
    if np.mod(dx,2) > 0:
        dx = int(dx) + 1
        halfx = (dx - 1)/2
    else:
        halfx = int(dx)/2
    
    if np.mod(dy,2) > 0:
        dy = int(dy) + 1
        halfy = (dy - 1)/2
    else:
        halfy = int(dy)/2

    xmin = int(x) - halfx
    xmax = int(x) + halfx
    ymin = int(y) - halfy
    ymax = int(y) + halfy
    
    if diagnostics:
        print('Center: ',xcen,ycen)
        print('Limits: ',maxx,maxy,dx,dy)
        print('CORNERS: ',xmin, xmax, ymin, ymax)
        
    if xmin >= 0 and xmax < maxx and ymin >= 0 and ymax < maxy:

        return np.array([xmin, xmax, ymin, ymax]).astype(int)

    else:

        if over_edge == False:

            return (None, None, None, None)

        else:

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(maxx, xmax)
            ymax = min(maxy, ymax)

            if diagnostics:
                print('CORNERS: ', xmin, xmax, ymin, ymax)

            return np.array([xmin, xmax, ymin, ymax]).astype(int)


def coadd_stamps(setup, stamps, log, diagnostics=True):
    """Function to combine a set of identically-sized image cutout2D objects,
    by co-adding to generate a single high signal-to-noise stamp.
    
    :param SetUp setup: Fundamental reduction parameters
    :param list stamps: List of Cutout2D image stamps.  May include None entries
    :param logger log: Open reduction log file object
    :param boolean diagnostics: Switch for debugging output. Default=False
    
    Returns:
    
    :param Cutout2D master_stamp: Combined stamp image
    """

    i = 0
    while i < len(stamps):

        if stamps[i] != None:

            coadd = np.zeros(stamps[i].shape)

            i = len(stamps) + 1

        else:

            i += 1

    xc = coadd.shape[1] / 2
    yc = coadd.shape[0] / 2

    for i, s in enumerate(stamps):

        if s != None:
            coadd += s.data

    master_stamp = Cutout2D(coadd, (xc, yc), coadd.shape, copy=True)

    log.info('Co-added ' + str(len(stamps)) + ' to produce a master_stamp')

    if diagnostics:
        hdu = fits.PrimaryHDU(master_stamp.data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(setup.red_dir,
                                     'ref', 'master_stamp.fits'),
                        overwrite=True)

    return master_stamp

def fit_psf_model(setup,log,psf_model_type,psf_diameter,sky_model_type,stamp_image, 
                  diagnostics=False):
    """Function to fit a PSF model to a stamp image"""
    
    half_stamp = stamp_image.shape[0]/2
    if (psf_diameter % 2) != 0:
        half_psf = int(psf_diameter / 2.0) + 1
    else:
        half_psf = int(psf_diameter / 2.0)
        
    xmin = int(half_stamp - half_psf)
    xmax = int(half_stamp + half_psf)
    ymin = int(half_stamp - half_psf)
    ymax = int(half_stamp + half_psf)
    
    #Y_data, X_data = np.indices(stamp_image.shape)
    
    #Y_data, X_data = np.indices([int(round(psf_diameter,0)),int(round(psf_diameter,0))])
    Y_data, X_data = np.indices([2*half_psf,2*half_psf])
    
    if diagnostics:
        hdu = fits.PrimaryHDU(stamp_image.data[ymin:ymax, xmin:xmax])
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(setup.red_dir,
                                     'ref', 'stamp_image_fitting.fits'),
                        overwrite=True)

    substamp = stamp_image.data[ymin:ymax, xmin:xmax]

    psf_fit = fit_star(substamp, Y_data, X_data,
                       psf_model_type, sky_model_type)

    log.info('Fitted PSF model parameters using a ' + psf_model_type + \
             ' PSF and ' + sky_model_type.lower() + ' sky background model')

    for p in psf_fit[0]:
        log.info(str(p))
    
    fitted_model = get_psf_object( psf_model_type )
    
    fitted_model.update_psf_parameters(psf_fit[0])
    
    if diagnostics:
        psf_stamp = generate_psf_image(psf_model_type,psf_fit[0],
                                       substamp.shape, psf_diameter)
        
        hdu = fits.PrimaryHDU(psf_stamp)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(setup.red_dir,
                                     'ref','psf.fits'),
                                     overwrite=True)
    
    fitted_model = get_psf_object(psf_model_type)

    fitted_model.update_psf_parameters(psf_fit[0])

    logs.ifverbose(log, setup, ' -> Parameters of fitted PSF model: ' + \
                   repr(fitted_model.get_parameters()))

    return fitted_model


def generate_psf_image(psf_model_type, psf_model_pars, stamp_dims, psf_size):
    new_psf_model_pars = []
    new_psf_model_pars = [x for x in psf_model_pars]

    if stamp_dims[0] != psf_size:
        (ix, iy) = get_psf_centre_indices(psf_model_type)

        x_centre = stamp_dims[1] / 2.0
        y_centre = stamp_dims[0] / 2.0
        
        new_psf_model_pars[ix] = x_centre
        new_psf_model_pars[iy] = y_centre

    psf = Image(np.zeros(stamp_dims), psf_model_type)

    Y_data, X_data = np.indices(stamp_dims)

    psf_image = psf.psf_model.psf_model(Y_data, X_data, new_psf_model_pars)

    return psf_image


def get_psf_object(psf_type):
    if psf_type == 'Moffat2D':

        model = Moffat2D()

    elif psf_type == 'Gaussian2D':

        model = Gaussian2D()

    elif psf_type == 'BivariateNormal':

        model = BivariateNormal()

    elif psf_type == 'Lorentzian2D':

        model = Lorentzian2D()

    else:

        model = Gaussian2D()

    return model


def get_psf_centre_indices(psf_model_type):
    model = get_psf_object(psf_model_type)

    iy = model.model.index('y_center')
    ix = model.model.index('x_center')

    return ix, iy


def subtract_companions_from_psf_stamps(setup, reduction_metadata, log,
                                        ref_star_catalog, psf_idx, stamps,
                                        psf_star_locations,
                                        psf_model,sky_model,psf_diameter,
                                        diagnostics=False):
    """Function to perform a PSF fit to all companions in the PSF star stamps,
    so that these companion stars can be subtracted from the stamps.
    
    :param setup object setup: the fundamental pipeline setup configuration
    :param metadata object reduction_metadata: detailed pipeline metadata object
    :param logging object log: an open logging file
    :param array ref_star_catalog: positions and magnitudes of stars in the 
                                reference image
    :param list psf_idx: Indices of PSF stars in the ref_star_catalog
    :param list stamps: list of Cutout2D image stamps around the PSF stars
    :param array psf_star_locations: x,y pixel locations of PSF stars in the
                                    full reference frame
    :param PSFModel object psf_model: the PSF model to be fitted
    :param BackgroundModel object sky_model: Model of the image background
    :param boolean diagnostics: Switch for optional output
    
    Returns
    
    :param list clean_stamps: list of Cutout2D image stamps around the PSF stars
                                with the companion stars subtracted
                                May contain None entries where PSF stars are
                                too close to the edge to generate a complete
                                stamp image.
    """

    log.info('Cleaning PSF stamps')

    dx = psf_diameter
    dy = psf_diameter
    half_psf = int(float(psf_diameter)/2.0)

    clean_stamps = []
    for i, j in enumerate(psf_idx):

        istar = j - 1

        s = stamps[i]

        if s != None:
            
            comps_list = find_psf_companion_stars(setup, istar,
                                                  psf_star_locations[i, 0],
                                                  psf_star_locations[i, 1],
                                                  dx, ref_star_catalog, log,
                                                  s.shape)

            if diagnostics:
                output_stamp_image(s.data,
                        os.path.join(setup.red_dir,'ref','psf_stamp'+str(int(j))+'.png'),
                        comps_list=comps_list)
                
                output_fits(s.data,
                        os.path.join(setup.red_dir,'ref','psf_stamp'+str(int(j))+'.fits'))
                        
            x_psf_box = psf_star_locations[i,0] - int(float(s.shape[1])/2.0)
            y_psf_box = psf_star_locations[i,1] - int(float(s.shape[0])/2.0)
            
            comp_image = np.zeros(s.data.shape)
            diff_image = copy.copy(s.data)
            
            for star_data in comps_list:

                (substamp, corners) = extract_sub_stamp(setup, log, star_data[0], s,
                                                        star_data[1], star_data[2],
                                                        dx, dy,
                                                        diagnostics=False)
                
                if substamp != None:
                    output_fits(substamp.data,
                    os.path.join(setup.red_dir,'ref','companions_star_stamp'+str(int(star_data[0]))+'.fits'))
                    
                if substamp != None:
                    
                    sub_psf_model = get_psf_object('Moffat2D')
                    pars = psf_model.get_parameters()
                    pars[1] = star_data[2]  # Y
                    pars[2] = star_data[1]  # X
                    sub_psf_model.update_psf_parameters(pars)
                    log.info('Made sub PSF model centered at '+str(pars[2])+', '+str(pars[1]))
                    
                    # THIS ONLY WORKS FOR CONSTANT BACKGROUND
                    # Otherwise it will need to be extracted for the specific 
                    # region of the full image
                    Y_grid, X_grid = np.indices(s.data.shape)
                    sky_model_bkgd = sky_bkgd = sky_model.background_model(Y_grid.shape,
                                                                           sky_model.get_parameters())
                    
                    s.data[corners[2]:corners[3],corners[0]:corners[1]] = sky_model_bkgd[corners[2]:corners[3],corners[0]:corners[1]]
                    
                    (comp_psf,good_fit) = fit_star_existing_model(setup, s.data, 
                                                       pars[2], pars[1],
                                                       psf_diameter, 
                                                       sub_psf_model, 
                                                       sky_model_bkgd,
                                                       diagnostics=True)
                    
                    logs.ifverbose(log,setup,' -> Fitted PSF parameters for companion '+
                            str(star_data[0]+1)+': '+repr(comp_psf.get_parameters())+' Good fit? '+repr(good_fit))
                    
                    if good_fit:
                        comp_psf_image = subtract_psf_from_image(s.data, comp_psf,
                                                         star_data, corners)
                        
                        s.data -= comp_psf_image
                        comp_image += comp_psf_image
                        
                        logs.ifverbose(log,setup,' -> Subtracted companion star from PSF stamp')
                    
            clean_stamps.append(s)

            if diagnostics:
                output_stamp_image(s.data,
                    os.path.join(setup.red_dir,'ref','clean_stamp'+str(int(j))+'.png'),
                    comps_list=comps_list)
                output_fits(s.data,
                    os.path.join(setup.red_dir,'ref','clean_stamp'+str(int(j))+'.fits'))
                
                output_fits(comp_image,
                    os.path.join(setup.red_dir,'ref','companions_models_stamp'+str(int(j))+'.fits'))
                
                diff_image -= s.data
                output_fits(diff_image,
                    os.path.join(setup.red_dir,'ref','diff_stamp'+str(int(j))+'.fits'))
                
        else:

            log.info(' -> No stamp available for PSF star ' + str(int(j)))

            clean_stamps.append(None)
    
    return clean_stamps

def subtract_psf_from_image(image, psf_model, star_data, corners, 
                            diagnostics=True):
    """Function to subtract a PSF Model from an array of image pixel data
    at a specified location.
    
    :param array image: Image pixel data
    :param PSFModel psf_model: PSF model object to be subtracted
    :param list, floats star_data: (x,y)-centre of stellar PSF to be subtracted
    :param list, floats: Corners of sub-stamp [xmin,xmax,ymin,ymax]
    
    Returns:
    
    :param array residuals: Image pixel data with star subtracted
    """

    dxc = corners[1] - corners[0]
    dyc = corners[3] - corners[2]
    
    if dxc > 0 and dyc > 0:

        Y_data, X_data = np.indices(image.shape)
        
        psf_image = psf_model.psf_model(Y_data, X_data, 
                                        psf_model.get_parameters())

        residuals = np.copy(image)
    
        residuals -= psf_image
        
    else:
        residuals = np.copy(image)
        psf_image = np.copy(image)
        
    return psf_image



def subtract_psf_from_image_with_kernel(image, psf_model, xstar, ystar, dx, dy, kernel,
                                        diagnostics=True):
    """Function to subtract a PSF Model from an array of image pixel data
    at a specified location.
    
    :param array image: Image pixel data
    :param PSFModel psf_model: PSF model object to be subtracted
    :param float xstar: x-centre of stellar PSF to be subtracted
    :param float ystar: y-centre of stellar PSF to be subtracted
    :param float dx: width of PSF stamp
    :param float dy: height of PSF stamp
    :param array_like kernel: the kernel data in 2d np.array
    
    Returns:
    
    :param array residuals: Image pixel data with star subtracted
    """

    corners = calc_stamp_corners(xstar, ystar, dx, dy,
                                 image.shape[1], image.shape[0],
                                 over_edge=True)

    dxc = corners[1] - corners[0]
    dyc = corners[3] - corners[2]

    Y_data, X_data = np.indices([int(dyc), int(dxc)])

    psf_image = psf_model.psf_model(Y_data, X_data, psf_model.get_parameters())

    psf_image_with_kernel = convolution.convolve_image_with_a_psf(psf_image, kernel, fourrier_transform_psf=None,
                                                                  fourrier_transform_image=None,
                                                                  correlate=None, auto_correlation=None)

    residuals = np.copy(image)

    # output_stamp_image(residuals[corners[2]:corners[3],corners[0]:corners[1]],
    #                  '/Users/rstreet/software/pyDANDIA/pyDANDIA/tests/data/subtractions/presubtraction_'+str(round(xstar,0))+'_'+str(round(ystar,0))+'.png')
    # output_stamp_image(psf_image,
    #                   '/Users/rstreet/software/pyDANDIA/pyDANDIA/tests/data/subtractions/residuals_psf_model_'+str(round(xstar,0))+'_'+str(round(ystar,0))+'.png')

    residuals[corners[2]:corners[3], corners[0]:corners[1]] -= psf_image_with_kernel

    # output_stamp_image(residuals[corners[2]:corners[3],corners[0]:corners[1]],
    #                   '/Users/rstreet/software/pyDANDIA/pyDANDIA/tests/data/subtractions/residuals_'+str(round(xstar,0))+'_'+str(round(ystar,0))+'.png')

    return residuals, corners


def output_stamp_image(image, file_path, comps_list=None):
    """Function to output a PNG image of a stamp"""

    fig = plt.figure(1)

    norm = visualization.ImageNormalize(image, \
                                        interval=visualization.ZScaleInterval())

    plt.imshow(image, origin='lower', cmap=plt.cm.viridis,
               norm=norm)

    if comps_list != None:
        x = []
        y = []
        for j in range(0, len(comps_list), 1):
            x.append(comps_list[j][1])
            y.append(comps_list[j][2])

        plt.plot(x, y, 'r+')

    plt.xlabel('X pixel')

    plt.ylabel('Y pixel')

    plt.axis('equal')

    plt.colorbar()

    plt.savefig(file_path)

    plt.close(1)


def find_psf_companion_stars(setup, psf_idx, psf_x, psf_y, psf_size,
                             ref_star_catalog, log, stamp_dims):
    """Function to identify stars in close proximity to a selected PSF star, 
    that lie within the image stamp used to build the PSF. 
    
    This necessarily means searching a box slightly bigger that the stamp
    dimensions given, in order to identify stars whose centroid may lie slightly
    outside the PSF box but whose PSF overlaps it. 
    
    :param SetUp object setup: Fundamental reduction parameters
    :param int psf_idx: Index of PSF star in the data array (NOT the PSF star number)
    :param float psf_x: x-pixel position of the PSF star in the full frame image
    :param float psf_y: y-pixel position of the PSF star in the full frame image
    :param float psf_size: diameter of the stellar PSF
    :param array ref_star_catalog: positions and magnitudes of stars in the 
                                reference image
    :param logging object log: an open logging file
    :param tup stamp_dims: Dimension of the stamp image
    
    Returns
    
    :param list comps_list: List of lists of the indices and x,y positions
                                of companion stars in the PSF stamp and in
                                the ref_star_catalog
    """

    psf_radius = int(float(psf_size / 2.0))
    dx_psf_box = int(float(stamp_dims[1]) / 2.0) + psf_radius
    dy_psf_box = int(float(stamp_dims[0]) / 2.0) + psf_radius
    x_psf_box = psf_x - dx_psf_box + psf_radius
    y_psf_box = psf_y - dy_psf_box + psf_radius

    x_sep = ref_star_catalog[:, 1] - psf_x
    y_sep = ref_star_catalog[:, 2] - psf_y

    logs.ifverbose(log, setup, 'Searching for companions to PSF star ' +
                   str(psf_idx + 1) + ' located at (' + str(psf_x) + ', ' + str(psf_y) + ')')
    logs.ifverbose(log, setup, 'PSF box: ' + \
                   'xmin=' + str(x_psf_box) + ', ymin=' + str(y_psf_box) + \
                   ' stamp dims=' + repr(stamp_dims) + ', PSF diameter=' + str(psf_size))

    comps_list = []

    idx = np.where(abs(x_sep) < dx_psf_box)

    logs.ifverbose(log, setup, ' -> Identified ' + str(len(idx[0])) + \
                   ' possible companions with the x-range of the PSF box')

    jdx = np.where(abs(y_sep) < dy_psf_box)

    logs.ifverbose(log, setup, ' -> Identified ' + str(len(jdx[0])) + \
                   ' possible companions with the y-range of the PSF box')

    comp_idx = list(set(idx[0]) & set(jdx[0]))

    if psf_idx in comp_idx:
        j = comp_idx.index(psf_idx)

        i = comp_idx.pop(j)

    logs.ifverbose(log, setup, ' -> Identified ' + str(len(comp_idx)) + \
                   ' objects within the PSF box (not including the PSF star itself)')

    xc = ref_star_catalog[comp_idx, 1] - x_psf_box
    yc = ref_star_catalog[comp_idx, 2] - y_psf_box

    comps_list = list(zip(comp_idx, xc, yc,
                          ref_star_catalog[comp_idx, 1].tolist(),
                          ref_star_catalog[comp_idx, 2].tolist()))

    if setup.verbosity == 2:

        for j in range(0, len(comp_idx), 1):
            (i, x, y, xo, yo) = comps_list[j]

            logs.ifverbose(log, setup, ' -> Near neighbour: ' + str(i + 1) + ' ' + str(x) + \
                           ' ' + str(y) + ' ' + str(xo) + ' ' + str(yo) + ' ' + \
                           str(x_sep[i]) + ' ' + str(y_sep[i]))

    logs.ifverbose(log, setup, ' -> Found ' + str(len(comps_list)) + \
                   ' companions near (' + str(psf_x) + ', ' + str(psf_y) + \
                   ') for PSF star ' + str(psf_idx))

    return comps_list


def extract_sub_stamp(setup, log, sidx, stamp, xcen, ycen, dx, dy, diagnostics=False):
    """Function to extract the sub-section of an image stamp around a given
    pixel location, taking into account that if the location is close to one
    of the edges of the image, the sub-section returned may be curtailed.
    
    :param SetUp object setup: Fundamental reduction parameters
    :param logger object log: Open reduction log file
    :param int sidx: ARRAY index of star in the ref_star_catalog 
    :param Cutout2D object stamp: Image stamp centred around a PSF star
    :param float xcen: x-pixel position of PSF companion in stamp coordinates
    :param float ycen: y-pixel position of PSF companion in stamp coordinates
    :param int dx: Width of substamp in x-direction in pixels
    :param int dy: Width of substamp in y-direction in pixels
    :param boolean diagnostics: Switch for debugging output
    
    Returns:
    
    :param Cutout2D object substamp: Image sub-stamp
    """

    logs.ifverbose(log, setup, 'Extracting a ' + str(dx) + 'x' + str(dy) +
                   ' sub stamp image for position (' + str(xcen) + ', ' + str(ycen) + ')')

#    (ymax_stamp, xmax_stamp) = stamp.shape

#    halfdx = int(float(dx) / 2.0)
#    halfdy = int(float(dy) / 2.0)

#    x1 = int(xcen) - halfdx
#    x2 = int(xcen) + halfdx
#    y1 = int(ycen) - halfdy
#    y2 = int(ycen) + halfdy
    
#    x1 = max(x1,0)
#    y1 = max(y1,0)
#    x2 = min(x2,xmax_stamp)
#    y2 = min(y2,ymax_stamp)
#    corners = [x1,x2,y1,y2]
    
    corners = calc_stamp_corners(xcen, ycen, dy, dx,
                                 stamp.shape[1], stamp.shape[0],
                                 over_edge=True)
    x1 = corners[0]
    x2 = corners[1]
    y1 = corners[2]
    y2 = corners[3]
    
    logs.ifverbose(log,setup,'X, Y pixel limits of substamp: '+repr(corners))
    
    if (x2-x1) <= dx-2 or (y2-y1) <= dy-2:
        
        logs.ifverbose(log,setup,' -> companion star too close to the stamp edge to model')
        
        substamp = None

    else:

        xmidss = int((x2 - x1) / 2)
        ymidss = int((y2 - y1) / 2)

        substamp = Cutout2D(stamp.data[int(y1):int(y2), int(x1):int(x2)], (xmidss, ymidss), (int(y2 - y1), int(x2 - x1)), copy=True)

        if diagnostics:
            fig = plt.figure(1)

            norm = visualization.ImageNormalize(substamp.data, \
                                                interval=visualization.ZScaleInterval())

            plt.imshow(substamp.data, origin='lower', cmap=plt.cm.viridis,
                       norm=norm)

            plt.xlabel('X pixel')

            plt.ylabel('Y pixel')

            plt.axis('equal')

            plt.savefig(os.path.join(setup.red_dir, 'ref',
                                     'psf_sub_stamp' + str(sidx + 1) + '.png'))

            plt.close(1)

    return substamp, corners
