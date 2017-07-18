######################################################################
#                                                                   
# psf.py - Module which defines PSF models.
# More details in individual fonctions.

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+ 
######################################################################

import numpy as np
import abc
import collections
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

		self.model = ['intensity' , 'y_center' , 'x_center' , 'gamma' , 'alpha']
	
		self.psf_parameters = collections.namedtuple('parameters', self.model)
		
		for index, key in enumerate(self.model) :

			setattr(self.psf_parameters, key, None)


	def update_psf_parameters(self, parameters):

		for index, key in enumerate(self.model) :

			setattr(self.psf_parameters, key, parameters[index])

		

	def psf_model(self, Y_star, X_star, parameters):

		self.update_psf_parameters( parameters)

		model = self.psf_parameters.intensity * (1+((X_star-self.psf_parameters.x_center)**2+
			(Y_star-self.psf_parameters.y_center)**2)/self.psf_parameters.gamma**2)**(-self.psf_parameters.alpha)


		return model

	def psf_guess(self):
		
		#gamma, alpha
		return [2.0,2.0]

	def get_FWHM(self,gamma, alpha, pix_scale):
		
		fwhm = gamma*2*(2**(1/alpha)-1)**0.5*pix_scale 
		return fwhm

class Gaussian2D(PSFModel):
	
	def psf_type(self):

		return 'Gaussian2D'
	
	def define_psf_parameters(self):

		self.model = ['intensity' , 'y_center' , 'x_center' , 'width_y' , 'width_x']
		
		self.psf_parameters = collections.namedtuple('parameters', self.model)

		for index, key in enumerate(self.model) :

			setattr(self.psf_parameters, key, None)

	def update_psf_parameters(self, parameters):

		for index, key in enumerate(self.model) :

			setattr(self.psf_parameters, key, parameters[index])


	def psf_model(self, Y_star, X_star, parameters):

		self.update_psf_parameters( parameters)

		model = self.psf_parameters.intensity * np.exp(-(((X_star-self.psf_parameters.x_center)/self.psf_parameters.width_x)**2
			+((Y_star-self.psf_parameters.y_center)/self.psf_parameters.width_y)**2)/2) 


		return model

	def psf_guess(self):
		
		#width_x,with_x
		return [1.0,1.0]

	def get_FWHM(self,width_x, width_y, pixel_scale):
		
		fwhm = (width_x+width_y)/2*2*(2*np.log(2))**0.5*pixel_scale

		return fwhm

class Lorentzian2D(PSFModel):

        def psf_type():
	
	        return 'Lorentzian2D'
	

	def define_psf_parameters(self):

		self.model = ['intensity' , 'y_center' , 'x_center' , 'gamma']
		self.psf_parameters = collections.namedtuple('parameters', self.model)

		for index, key in enumerate(self.model) :

			setattr(self.psf_parameters, key, None)

	def update_psf_parameters(self, parameters):

		for index, key in enumerate(self.model) :

			setattr(self.psf_parameters, key, parameters[index])

	def psf_model(self, Y_star, X_star, parameters):
	        
		 self.update_psf_parameters( parameters)
		
		 model = self.psf_parameters.intensity * (self.psf_parameters.gamma/((X_star-self.psf_parameters.x_center)**2 + 
			 (Y_star-self.psf_parameters.y_center)**2 + self.psf_parameters.gamma**2 )**(1.5)) 
		 return model

	def psf_guess(self):
		
		#width_x
		return [1.0]
	
	def get_FWHM(self, gamma, pixel_scale):
		
		fwhm = 2*gamma*pixel_scale

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

		self.model =  ['constant']
		self.background_parameters = collections.namedtuple('parameters', self.model)

		for index, key in enumerate(self.model) :

			setattr(self.background_parameters, key, None)

	def update_background_parameters(self, parameters):

		self.background_parameters = collections.namedtuple('parameters', self.model)
		
		for index, key in enumerate(self.model) :

			setattr(self.background_parameters, key, parameters[index])

	def background_model(self, Y_background, X_background, parameters):
	        
		 self.update_background_parameters( parameters)
		
		 model =  np.ones(Y_background.shape)*self.background_parameters.constant
		 return model

	def background_guess(self):
		
		#background constant
		return [0]
	

class Image(object):

	def __init__(self, full_data, psf_model):

		self.full_data = full_data
		self.residuals = np.zeros(full_data.shape)

		self.model = np.zeros(self.full_data.shape)
		x_data = np.arange(0,self.full_data.shape[1])
		y_data = np.arange(0,self.full_data.shape[0])

		self.X_data, self.Y_data = np.meshgrid(x_data,y_data)
		
		if psf_model == 'Moffat2D':

			self.psf_model = Moffat2D()

		if psf_model == 'Gaussian2D':

			self.psf_model = Gaussian2D()

		if psf_model == 'Lorentzian2D':

			self.psf_model = Gaussian2D()


	def inject_psf_in_stars(self, model, parameters):



		X_data = self.X_data
		Y_data = self.Y_data

		psf_width = parameters[-1]

		for index in xrange(len(parameters[0])):

			index_star = (int(parameters[1][index]), int(parameters[2][index]))
			
				
			params = [parameters[0][index], parameters[1][index], parameters[2][index],parameters[3],parameters[4]]	
			X_star = X_data[index_star[0]-psf_width:index_star[0]+psf_width, index_star[1]-psf_width:index_star[1]+psf_width]
			Y_star = Y_data[index_star[0]-psf_width:index_star[0]+psf_width, index_star[1]-psf_width:index_star[1]+psf_width]
			
			stamp = self.psf_model.psf_model(Y_star, X_star, *params)

			model[index_star[0]-psf_width:index_star[0]+psf_width, index_star[1]-psf_width:index_star[1]+psf_width] += stamp

		return model

	def image_model(self,psf_parameters,background_parameters=0):
		
		model = np.zeros(self.full_data.shape)
		
		model = self.inject_psf_in_stars(model, psf_parameters)
		
		model += background_parameters


		self.model = model
		
	def image_residuals(self, psf_parameters, background_parameters=0):
		
		self.image_model(psf_parameters, background_parameters)

		self.residuals = self.full_data-self.model
		
		
	def stars_guess(self, star_positions, star_width = 10):

		x_centers = []
		y_centers = []
		intensities = []


		for index in xrange(len(star_positions)):

			data = self.full_data[int(star_positions[0])-star_width:int(star_positions[0])+star_width, 
					      int(star_positions[1])-star_width:int(star_positions[1])+star_width]

			intensities.append(data.max)
			x_centers.append(star_positions[1])
			y_centers.append(star_positions[0])

			
		return [np.array(intensities), np.array(y_centers), np.array(x_centers)]

	
def fit_star(data, Y_data,X_data, psf_model = 'Moffat2D', background_model = 'Constant' ):

	if psf_model == 'Moffat2D':

		psf_model = Moffat2D()

	if psf_model == 'Gaussian2D':

		psf_model = Gaussian2D()

	if psf_model == 'Lorentzian2D':

		psf_model = Lorentzian2D()

	if background_model == 'Constant':

		back_model = ConstantBackground()

	guess_psf = [data.max(), Y_data[len(Y_data[:,0])/2,0], X_data[0,len(Y_data[0,:])/2]] + psf_model.psf_guess()
	guess_back = back_model.background_guess()

	guess = guess_psf+guess_back

	fit = optimize.leastsq(error_star_fit_function, guess, args=(data, psf_model, back_model, Y_data, X_data), full_output=1)
	


	return fit

def error_star_fit_function( params, data, psf, background, Y_data, X_data):


	psf_params = params[:len( psf.psf_parameters._fields)]
	back_params = 	params[len( psf.psf_parameters._fields):]

	psf_model = psf.psf_model(Y_data,X_data, psf_params)
	back_model = background.background_model(Y_data,X_data, back_params)

	residuals = np.ravel(data-psf_model-back_model)

	return residuals

def fit_multiple_stars(data_stamps, Y_data, X_data, psf_model = 'Moffat2D',background_model = 'Constant'):

	
	fits = []
	for stamp in data_stamps :
		

		fits.append(fit_star(stamp, Y_data, X_data, psf_model, background_model ))

	
	return fits	

	


def plot3d(x, y, z):
    '''
    Plots 3D data.
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(x, y, z, alpha=0.5)
    ax.plot_surface(x, y, z, alpha=0.2)
    cset = ax.contourf(x, y, z, zdir='z', offset=min(z.flatten()), cmap=cm.coolwarm, alpha=0.2)
    cset = ax.contourf(x, y, z, zdir='x', offset=min(x[0]), cmap=cm.coolwarm, alpha=0.3)
    cset = ax.contourf(x, y, z, zdir='y', offset=max(y[-1]), cmap=cm.coolwarm, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
