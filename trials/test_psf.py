import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
sys.path.append('../pyDANDIA/')
import time

import psf
import starfind


image = fits.open('./data/lsc1m005-fl15-20170614-0130-e91.fits')

image = image[0].data

stars = starfind.starfind('./data/lsc1m005-fl15-20170614-0130-e91.fits', plot_it=False, write_log=True)

LIMIT_PSF_PIXEL = 10

X=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
Y=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
x,y = np.meshgrid(X,Y)


stamps = []
start = time.time()
for star in stars[:50]:

	star_x = int(np.round(star['xcentroid']))
	star_y = int(np.round(star['ycentroid']))

	stamp = image[star_y-LIMIT_PSF_PIXEL:star_y+LIMIT_PSF_PIXEL,star_x-LIMIT_PSF_PIXEL:star_x+LIMIT_PSF_PIXEL]
	stamps.append(stamp)


aa = psf.fit_multiple_stars(stamps, y, x, psf_model='Moffat2D', background_model='Constant')
print time.time()-start
import pdb; pdb.set_trace()


for star in stars:

	

	

	star_x = int(np.round(star['xcentroid']))
	star_y = int(np.round(star['ycentroid']))

	
	
	stamp = image[star_y-LIMIT_PSF_PIXEL:star_y+LIMIT_PSF_PIXEL,star_x-LIMIT_PSF_PIXEL:star_x+LIMIT_PSF_PIXEL]

	plt.subplot(331)
	plt.imshow(np.log10(stamp),interpolation='None')	
	plt.colorbar()

	X=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
	Y=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
	x,y = np.meshgrid(X,Y)



	
	fit = psf.fit_star(stamp,y,x, 'Gaussian2D')
	
	fit_params = fit[0]
	fit_errors = fit[1].diagonal()**0.5

	


	gaussian = psf.Gaussian2D()
	back = psf.ConstantBackground()
	PSF = gaussian.get_FWHM(fit_params[-3], fit_params[-2],0.389)


	plt.title('Gaussian2D \n PSF = '+str(PSF)+' BACK = '+str(fit_params[-1]))
	
	fit_residuals = psf.error_star_fit_function(fit_params, stamp,gaussian,back, y,x)

	fit_residuals = fit_residuals.reshape(stamp.shape)
	cov = fit[1]*np.sum(fit_residuals**2)/((LIMIT_PSF_PIXEL*2)**2-6)
	fit_errors = cov.diagonal()**0.5	
	print fit_params,fit_errors

	model = gaussian.psf_model(y,x, fit_params)
	model += back.background_model(y,x,[fit_params[-1]])
	plt.subplot(334)
	plt.imshow(np.log10(model),interpolation='None')
	plt.colorbar()

	
	plt.subplot(337)	 
	plt.imshow(fit_residuals,interpolation='None')
	plt.colorbar()




	plt.subplot(332)
	plt.imshow(np.log10(stamp),interpolation='None')	
	plt.colorbar()

	X=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
	Y=np.arange(-LIMIT_PSF_PIXEL,LIMIT_PSF_PIXEL)
	x,y = np.meshgrid(X,Y)



	
	fit = psf.fit_star(stamp,y,x, 'Lorentzian2D')
	
	fit_params = fit[0]
	fit_errors = fit[1].diagonal()**0.5
	lorentz = psf.Lorentzian2D()

	PSF = lorentz.get_FWHM(fit_params[-2],0.389)


	plt.title('Lorentzian2D \n PSF = '+str(PSF)+' BACK = '+str(fit_params[-1]))
	
	fit_residuals = psf.error_star_fit_function(fit_params, stamp,lorentz,back,y,x)

	fit_residuals = fit_residuals.reshape(stamp.shape)
	cov = fit[1]*np.sum(fit_residuals**2)/((LIMIT_PSF_PIXEL*2)**2-6)
	fit_errors = cov.diagonal()**0.5	
	print fit_params,fit_errors


	model = lorentz.psf_model(y,x, fit_params)
	model += back.background_model(y,x,[fit_params[-1]])
	plt.subplot(335)
	plt.imshow(np.log10(model),interpolation='None')
	plt.colorbar()

	
	plt.subplot(338)	 
	plt.imshow(fit_residuals,interpolation='None')
	plt.colorbar()



	plt.subplot(333)
	plt.imshow(np.log10(stamp),interpolation='None')	
	plt.colorbar()

	



	
	fit = psf.fit_star(stamp,y,x,'Moffat2D')

	fit_params = fit[0]
	 
	moffat = psf.Moffat2D()

	PSF = moffat.get_FWHM(fit_params[-3], fit_params[-2],0.389)


	plt.title('Moffat2D \n PSF = '+str(PSF)+' BACK = '+str(fit_params[-1]))
	
	fit_residuals = psf.error_star_fit_function(fit_params, stamp,moffat,back,y,x)
	fit_residuals = fit_residuals.reshape(stamp.shape)

	
	cov = fit[1]*np.sum(fit_residuals**2)/((LIMIT_PSF_PIXEL*2)**2-6)
	fit_errors = cov.diagonal()**0.5	
	print fit_params,fit_errors


	model = moffat.psf_model(y,x, fit_params)
	model += back.background_model(y,x,[fit_params[-1]])
	
	plt.subplot(336)
	plt.imshow(np.log10(model),interpolation='None')
	plt.colorbar()

	
	plt.subplot(339)	 
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
