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
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.nddata import Cutout2D
from astropy.io import fits
import logs
import os

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

        model = self.psf_parameters.intensity * (1 + ((X_star - self.psf_parameters.x_center)**2 +\
                                                      (Y_star - self.psf_parameters.y_center)**2) / self.psf_parameters.gamma**2)**(-self.psf_parameters.alpha)

        return model

    def psf_guess(self):

        #gamma, alpha
        return [2.0, 2.0]

    def get_FWHM(self, gamma, alpha, pix_scale):

        fwhm = gamma * 2 * (2**(1 / alpha) - 1)**0.5 * pix_scale
        return fwhm


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

        model = self.psf_parameters.intensity * np.exp(-(((X_star - self.psf_parameters.x_center) / self.psf_parameters.width_x)**2 +\
                                                         ((Y_star - self.psf_parameters.y_center) / self.psf_parameters.width_y)**2) / 2)

        return model

    def psf_guess(self):

        # width_x, width_y
        return [1.0, 1.0]

    def get_FWHM(self, width_x, width_y, pixel_scale):

        #fwhm = (width_x + width_y) / 2 * 2 * (2 * np.log(2))**0.5 * pixel_scale
        fwhm = (width_x + width_y) * 1.1774100225154747 * pixel_scale

        return fwhm


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

        zeta = ((X_star - self.psf_parameters.x_center) / self.psf_parameters.width_x)**2 - (2 * self.psf_parameters.corr_xy * (X_star - self.psf_parameters.x_center) *\
                (Y_star - self.psf_parameters.y_center)) / (self.psf_parameters.width_x * self.psf_parameters.width_y) +\
                ((Y_star - self.psf_parameters.y_center) / self.psf_parameters.width_y)**2

        model = self.psf_parameters.intensity * \
            np.exp(-zeta / (2 * (1 - self.psf_parameters.corr_xy * self.psf_parameters.corr_xy)))

        return model

    def psf_guess(self):

        # width_x, width_y, corr_xy
        return [1.0, 1.0, 0.7]

    def get_FWHM(self, width_x, width_y, pixel_scale):

        #fwhm = (width_x + width_y) / 2 * 2 * (2 * np.log(2))**0.5 * pixel_scale
        fwhm = (width_x + width_y) * 1.1774100225154747 * pixel_scale

        return fwhm


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

        model = self.psf_parameters.intensity * (self.psf_parameters.gamma / ((X_star - self.psf_parameters.x_center)**2 +\
                                                                              (Y_star - self.psf_parameters.y_center)**2 + self.psf_parameters.gamma**2)**(1.5))
        return model

    def psf_guess(self):

        # width_x
        return [1.0]

    def get_FWHM(self, gamma, pixel_scale):

        fwhm = 2 * gamma * pixel_scale

        return fwhm


class BackgroundModel(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):

        self.name = None
        self.model = None
        self.background_parameters = None
        self.define_background_parameters()

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

    def background_model(self, Y_background, X_background, parameters):

        self.update_background_parameters(parameters)

        model = np.ones(Y_background.shape) * \
            self.background_parameters.constant
        return model

    def background_guess(self):

        # background constant
        return [0]

class GradientBackground(BackgroundModel):

    def background_type(self):

        return 'Gradient'

    def define_background_parameters(self):

        self.model = ['a0','a1','a2']
        self.background_parameters = collections.namedtuple(
            'parameters', self.model)

        for index, key in enumerate(self.model):

            setattr(self.background_parameters, key, None)

    def update_background_parameters(self, parameters):

        self.background_parameters = collections.namedtuple(
            'parameters', self.model)

        for key, value in parameters.items():

            setattr(self.background_parameters, key, value)

    def background_model(self, Y_background, X_background, parameters):

        self.update_background_parameters(parameters)

        model = np.ones(Y_background.shape) * self.background_parameters.a0
        model = model + ( self.background_parameters.a1 * X_background ) + \
                    + ( self.background_parameters.a2 * Y_background )
        
        return model

    def background_guess(self):
        """Method to return an initial estimate of the parameters of a 2D
        gradient sky background model.  The parameters returned represent 
        a flat, constant background of zero."""
        
        return [0.0, 0.0, 0.0]


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


def fit_star(data, Y_data, X_data, psf_model='Moffat2D', background_model='Constant'):

    if psf_model == 'Moffat2D':

        psf_model = Moffat2D()

    if psf_model == 'Gaussian2D':

        psf_model = Gaussian2D()

    if psf_model == 'BivariateNormal':

        psf_model = BivariateNormal()

    if psf_model == 'Lorentzian2D':

        psf_model = Lorentzian2D()

    if background_model == 'Constant':

        back_model = ConstantBackground()

    guess_psf = [data.max(), Y_data[len(Y_data[:, 0]) / 2, 0],
                 X_data[0, len(Y_data[0, :]) / 2]] + psf_model.psf_guess()

    guess_back = back_model.background_guess()

    guess = guess_psf + guess_back

    fit = optimize.leastsq(error_star_fit_function, guess, args=(
        data, psf_model, back_model, Y_data, X_data), full_output=1)

    return fit

def fit_star_existing_model(data, x_cen, y_cen, psf_radius, psf_model, sky_model):
    """Function to fit an existing PSF and sky model to a star at a given 
    location in an image, optimizing only the peak intensity of the PSF rather 
    than all parameters.
    
    :param array data: image data to be fitted
    :param float x_cen: the x-pixel location of the PSF to be fitted in the 
                        coordinates of the image
    :param float x_cen: the x-pixel location of the PSF to be fitted in the 
                        coordinates of the image
    :param float psf_radius: the radius of data to fit the PSF to
    :param PSFModel psf_model: existing psf model
    :param BackgroundModel sky_model: existing model for the image sky background
    
    Returns
    
    """
    
    Y_data, X_data = np.indices(data.shape)
    
    sep = np.sqrt((X_data - x_cen)**2 + (Y_data - y_cen)**2)
    
    idx = np.where(sep <= psf_radius)
    
    Y_data = Y_data[idx]
    X_data = X_data[idx]
    
    init_par = psf_model.XXX
    
    fit = optimize.leastsq(error_star_fit_existing_model, guess, args=(
        data, psf_model, back_model, Y_data, X_data), full_output=1)

def error_star_fit_function(params, data, psf, background, Y_data, X_data):

    psf_params = params[:len(psf.psf_parameters._fields)]
    back_params = params[len(psf.psf_parameters._fields):]

    psf_model = psf.psf_model(Y_data, X_data, psf_params)
    back_model = background.background_model(Y_data, X_data, back_params)

    residuals = np.ravel(data - psf_model - back_model)

    return residuals

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
              psf_stars_idx, diagnostics=True):
    """Function to build a PSF model based on the PSF stars
    selected from a reference image."""
    
    log.info('Building a PSF model based on the reference image')
    
    # Cut large stamps around selected PSF stars
    psf_size = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    psf_model_type = 'Moffat2D'
    sky_model_type = 'Constant'
    logs.ifverbose(log,setup,' -> Applying PSF size='+str(psf_size))

    idx = np.where(psf_stars_idx == 1.0)
    psf_star_centres = ref_star_catalog[idx[0],1:3]
    
    log.info('Cutting stamps for '+str(len(psf_star_centres))+' PSF stars')
    
    stamp_dims = (int(psf_size)*4, int(psf_size)*4)
    logs.ifverbose(log,setup,' -> Stamp dimensions='+repr(stamp_dims))

    stamps = cut_image_stamps(image, psf_star_centres, stamp_dims, log)
    
    # Combine stamps into a single, high-signal-to-noise PSF
    master_stamp = coadd_stamps(setup, stamps, log, diagnostics=diagnostics)
    
    # Build initial PSF: fit a PSF model to the high S/N stamp
    fit_psf_model(setup,log,psf_model_type,sky_model_type,master_stamp, 
                  diagnostics=diagnostics)
    
    # Identify all stars neighbouring the PSF stars within the large stamps.
    subtract_companions_from_psf_stamps()
    
    # Remove the companion stars from the PSF stamps
    
    # Re-combine the companion-subtracted stamps to re-generate the 
    # high S/N stamp
    
    # Re-build the final PSF by fitting a PSF model to the updated high 
    # S/N stamp
    
    # Output diagnostic plots of high S/N stamps and PSF model.
    
    # return PSF model parameters

def cut_image_stamps(image, stamp_centres, stamp_dims, log=None):
    """Function to extract a set of stamps (2D image sections) at the locations
    given and with the dimensions specified in pixels.
    
    No stamp will be returned for stars that are too close to the edge of the
    frame, that is, where the stamp would overlap with the edge of the frame. 
    
    :param array image: the image data array from which to take stamps
    :param array stamp_centres: 2-col array with the x, y centres of the stamps
    :param tuple stamp_dims: the width and height of the stamps
    
    Returns
    
    :param list Cutout2D objects
    """

    stamps = []
    
    dx = int(stamp_dims[1]/2.0)
    dy = int(stamp_dims[0]/2.0)
    
    for j in range(0,len(stamp_centres),1):
        xcen = stamp_centres[j,0]
        ycen = stamp_centres[j,1]
        
        corners = calc_stamp_corners(xcen, ycen, dx, dy, 
                                     image.shape[1], image.shape[0])
                                     
        if None not in corners:
            cutout = Cutout2D(image, (xcen,ycen), stamp_dims)
            
            stamps.append(cutout)
    
    if log != None:
        
        log.info('Made stamps for '+str(len(stamps))+' out of '+\
                    str(len(stamp_centres))+' locations')
        
    return stamps

def calc_stamp_corners(xcen, ycen, dx, dy, maxx, maxy):
    
    xmin = int(xcen) - dx
    xmax = int(xcen) + dx
    ymin = int(ycen) - dy
    ymax = int(ycen) + dy
        
    if xmin >= 0 and xmax <= maxx and ymin >= 0 and ymax <= maxy:
        
        return (xmin, xmax, ymin, ymax)
        
    else:
        
        return (None, None, None, None)
        
def coadd_stamps(setup, stamps, log, diagnostics=True):
    """Function to combine a set of identically-sized image cutout2D objects,
    by co-adding to generate a single high signal-to-noise stamp."""
    
    master_stamp = np.zeros(stamps[0].shape)
    
    for s in stamps:
        
        if s != None:
            
            master_stamp += s.data
    
    log.info('Co-added '+str(len(stamps))+' to produce a master_stamp')
    
    if diagnostics:
        
        hdu = fits.PrimaryHDU(master_stamp)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(setup.red_dir,
                                     'ref','master_stamp.fits'),
                                     overwrite=True)
        
    return master_stamp

def fit_psf_model(setup,log,psf_model_type,sky_model_type,stamp_image, 
                  diagnostics=False):
    """Function to fit a PSF model to a stamp image"""
    
    psf_model_type = 'Moffat2D'
    sky_model_type = 'Constant'
    Y_data, X_data = np.indices(stamp_image.shape)
    
    psf_fit = fit_star(stamp_image, Y_data, X_data, 
                                      psf_model_type, sky_model_type)
    
    log.info('Initial PSF model parameters using a '+psf_model_type+\
            ' PSF and '+sky_model_type.lower()+' sky background model')
    for p in psf_fit[0]:
        log.info(str(p))
    
    if diagnostics:
        
        psf_stamp = generate_psf_image(psf_model_type,psf_fit[0],stamp_image.shape)
        
        hdu = fits.PrimaryHDU(psf_stamp)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(setup.red_dir,
                                     'ref','psf.fits'),
                                     overwrite=True)
    
    return psf_fit

def generate_psf_image(psf_model_type,psf_model_pars,stamp_dims):
    
    psf = Image(np.zeros(stamp_dims),psf_model_type)
    
    Y_data, X_data = np.indices(stamp_dims)
    
    psf_image = psf.psf_model.psf_model(Y_data, X_data, psf_model_pars)
    
    return psf_image
    
def subtract_companions_from_psf_stamps(setup, reduction_metadata, log, 
                                        ref_star_catalog, stamps,
                                        psf_model):
    """Function to perform a PSF fit to all companions in the PSF star stamps,
    so that these companion stars can be subtracted from the stamps.
    
    :param setup object setup: the fundamental pipeline setup configuration
    :param metadata object reduction_metadata: detailed pipeline metadata object
    :param logging object log: an open logging file
    :param array ref_star_catalog: positions and magnitudes of stars in the 
                                reference image
    :param list stamps: list of Cutout2D image stamps around the PSF stars
    :param PSFModel object psf_model: the PSF model to be fitted
    :param str sky_model_type: Name of the type of sky model to be used
    
    Returns
    
    :param list clean_stamps: list of Cutout2D image stamps around the PSF stars
                                with the companion stars subtracted
    """
    
    dx = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    dy = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]
    
    for s in stamps:
        
        comps_list = find_psf_companion_stars(psf_x, psf_y,ref_star_catalog,log)
    
        Y_data, X_data = np.indices(stamp.shape)
    
        for star_data in comps_list:
        
            substamp = extract_sub_stamp(stamp,star_data[1],star_data[2],dx,dy)
            
        # Fit the existing psf_model to the sub-stamp, returning a model
        # PSF sub-stamp image
        
        # Subtract the model sub-stamp image from the image stamp.
            pass
                          
        
def find_psf_companion_stars(psf_idx, psf_x, psf_y, ref_star_catalog,
                             log, stamp_dims):
    """Function to identify stars in close proximity to a selected PSF star, 
    that lie within the image stamp used to build the PSF. 
    
    :param float psf_x: x-pixel position of the PSF star in the full frame image
    :param float psf_y: y-pixel position of the PSF star in the full frame image
    :param array ref_star_catalog: positions and magnitudes of stars in the 
                                reference image
    :param logging object log: an open logging file
    
    Returns
    
    :param list comps_list: List of lists of the indices and x,y positions
                                of companion stars in the PSF stamp and in
                                the ref_star_catalog
    """
    
    x_psf_box = psf_x - int(float(stamp_dims[1])/2.0)
    y_psf_box = psf_y - int(float(stamp_dims[0])/2.0)
    
    dx = ref_star_catalog[:,1] - psf_x
    dy = ref_star_catalog[:,2] - psf_y
    max_radius = np.sqrt( (float(stamp_dims[1])/2.0)**2 + \
                            (float(stamp_dims[0])/2.0)**2 )
                            
    separations = np.sqrt( dx*dx + dy*dy )
    
    idx = separations.argsort()
    
    comps_list = []
    
    jdx = np.where(separations[idx] < max_radius)
    
    for j in jdx:
        i = idx[j]
        
        if abs(dx[j]) <= stamp_dims[1] and abs(dy[j]) <= stamp_dims[0]:
            
            xc = ref_star_catalog[j,1] - x_psf_box
            xy = ref_star_catalog[j,2] - y_psf_box 
            
            comps_list.append([j, xc, xy, 
                               ref_star_catalog[j,1], ref_star_catalog[j,2]])
    
    log.info(' -> Found '+str(len(comps_list))+' near ('+str(psf_x)+\
            ', '+str(psf_y)+') for PSF star '+str(psf_idx))
    
    return comps_list
    
def extract_sub_stamp(stamp,xcen,ycen,dx,dy):
    """Function to extract the sub-section of an image stamp around a given
    pixel location, taking into account that if the location is close to one
    of the edges of the image, the sub-section returned may be curtailed.
    
    :param Cutout2D object stamp: Image stamp centred around a PSF star
    :param float xcen: x-pixel position of PSF companion in stamp coordinates
    :param float ycen: y-pixel position of PSF companion in stamp coordinates
    :param int dx: Width of substamp in x-direction in pixels
    :param int dy: Width of substamp in y-direction in pixels
    
    Returns:
    
    :param Cutout2D object substamp: Image sub-stamp
    """
    
    (ymax_stamp, xmax_stamp) = stamp.shape

    halfdx = int(float(dx)/2.0)
    halfdy = int(float(dy)/2.0)
    
    x1 = int(xcen) - halfdx
    x2 = int(xcen) + halfdx
    y1 = int(ycen) - halfdy
    y2 = int(ycen) + halfdy
    
    x1 = max(x1,0)
    y1 = max(y1,0)
    x2 = min(x2,xmax_stamp)
    y2 = min(y2,ymax_stamp)
    
    xmidss = (x2-x1)/2
    ymidss = (y2-y1)/2
        
    substamp = Cutout2D(stamp.data[y1:y2,x1:x2], (xmidss,ymidss), (x2-x1,y2-y1))
    
    return substamp
    