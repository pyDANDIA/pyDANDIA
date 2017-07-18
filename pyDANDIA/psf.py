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
		
	
	@abc.abstractproperty
	def psf_model(self, star_data, parameters):

		pass
	
class Moffat2D(PSFModel):
	
	def psf_type():

		return 'Moffat2D'
		
		

	def psf_model(self, Y_star, X_star, intensity, y_center, x_center, gamma, alpha, local_back):

		model = intensity * (1+((X_star-x_center)**2+(Y_star-y_center)**2)/gamma**2)**(-alpha) + local_back


		return model

	def psf_guess(self):
		
		#gamma, alpha
		return [2.0,2.0]


class Gaussian2D(PSFModel):
	
	def psf_type():

		return 'Gaussian2D'
		
		

	def psf_model(self, Y_star, X_star, intensity, y_center, x_center, width_y, width_x, local_back):

		model = intensity * np.exp(-(((X_star-x_center)/width_x)**2+((Y_star-y_center)/width_y)**2)/2) + local_back


		return model

	def psf_guess(self):
		
		#width_x,with_x
		return [1.0,1.0]

class Lorentzian2D(PSFModel):

        def psf_type():
	
	        return 'Lorentzian2D'
	
	
	def psf_model(self, Y_star, X_star, intensity, y_center, x_center, gamma, local_back):
	        
		 model = intensity * (gamma/((X_star-x_center)**2 + (Y_star-y_center)**2 + gamma**2 )**(1.5)) + local_back
		
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

	
def fit_psf(data, Y_data,X_data, psf_model = 'Moffat2D'):
	
	if psf_model == 'Moffat2D':

		psf_model = Moffat2D()

	if psf_model == 'Gaussian2D':

		psf_model = Gaussian2D()

	if psf_model == 'Lorentzian2D':

		psf_model = Lorentzian2D()


	guess = [data.max(), Y_data[len(Y_data[:,0])/2,0], X_data[0,len(Y_data[0,:])/2]] + psf_model.psf_guess()+[0]
	
	fit = optimize.leastsq(error_function, guess, args=(data, psf_model, Y_data, X_data), full_output=1)
	


	return fit


def error_function( psf_params, data, psf, Y_data, X_data):


	model = psf.psf_model(Y_data,X_data, *psf_params)

	residuals = np.ravel(data-model)

	return residuals
	


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
