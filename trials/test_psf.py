import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
sys.path.append('../pyDANDIA/')

import psf
import starfind


image = fits.open('./data/lsc1m005-fl15-20170614-0130-e91.fits')

image = image[0].data

stars = starfind.starfind('./data/lsc1m005-fl15-20170614-0130-e91.fits', plot_it=False, write_log=True)

LIMIT_PSF_PIXEL = 10


for star in stars:

	

	

	star_x = int(np.round(star['xcentroid']))
	star_y = int(np.round(star['ycentroid']))

	
	
	stamp = image[star_y-LIMIT_PSF_PIXEL:star_y+LIMIT_PSF_PIXEL,star_x-LIMIT_PSF_PIXEL:star_x+LIMIT_PSF_PIXEL]

	plt.subplot(321)
	plt.imshow(stamp)	
	plt.colorbar()

	X=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
	Y=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
	x,y = np.meshgrid(X,Y)



	
	fit = psf.fit_psf(stamp,y,x, 'Gaussian2D')
	
	fit_params = fit[0]
	fit_errors = fit[1].diagonal()**0.5
	gaussian = psf.Gaussian2D()
	fit_residuals = psf.error_function(fit_params, stamp,gaussian,y,x)
	fit_residuals = fit_residuals.reshape(stamp.shape)
	cov = fit[1]*np.sum(fit_residuals**2)/((LIMIT_PSF_PIXEL*2)**2-6)
	fit_errors = cov.diagonal()**0.5	
	print fit_params,fit_errors


	model = gaussian.psf_model(y,x, *fit_params)
	plt.subplot(323)
	plt.imshow(model)
	plt.colorbar()

	
	plt.subplot(325)	 
	plt.imshow(fit_residuals,interpolation='None')
	plt.colorbar()



	plt.subplot(322)
	plt.imshow(stamp)	
	plt.colorbar()

	



	
	fit = psf.fit_psf(stamp,y,x,'Moffat2D')

	fit_params = fit[0]
	 
	moffat = psf.Moffat2D()
	fit_residuals = psf.error_function(fit_params, stamp,moffat,y,x)
	fit_residuals = fit_residuals.reshape(stamp.shape)
	cov = fit[1]*np.sum(fit_residuals**2)/((LIMIT_PSF_PIXEL*2)**2-6)
	fit_errors = cov.diagonal()**0.5	
	print fit_params,fit_errors


	model = moffat.psf_model(y,x, *fit_params)
	plt.subplot(324)
	plt.imshow(model)
	plt.colorbar()

	
	plt.subplot(326)	 
	plt.imshow(fit_residuals,interpolation='None')
	plt.colorbar()
	plt.show()
	#import pdb; pdb.set_trace()

### Longer term idea, fit image not individual stars

data = np.zeros((4000,4000))

intensities = np.random.uniform(1000,50000,10000)
Y = np.random.uniform(0,4000,10000)
X = np.random.uniform(0,4000,10000)

params = [intensities,Y,X,2.1,2.5,10]

im = psf.Image(data,'Gaussian2D')
im.image_model(params,5000)



im.full_data = np.random.normal(im.model,im.model**0.5)
plt.imshow(im.full_data,interpolation='None')
plt.colorbar()
plt.show()

im.image_residuals(params,5000)
res = im.residuals
plt.imshow(res,interpolation='None')
plt.colorbar()
plt.show()
import pdb; pdb.set_trace()
