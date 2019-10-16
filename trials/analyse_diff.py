import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from astropy.stats import SigmaClip
from photutils import StdBackgroundRMS
from photutils import CircularAperture
from photutils.utils import calc_total_error
from photutils import aperture_photometry

def null_background(x,axis=None):

	return 0

names = [i for i in os.listdir('./diffim')]

cent1 = [604.68,486.6]
cent2 = [768.84,297.96]
cent3 = [462.436,215.249]

for nam in names:

	data = fits.open('./diffim/'+nam)[0].data

	stamp1 =  data[int(cent1[0])-10:int(cent1[0])+10,int(cent1[1])-10:int(cent1[1])+10]
	stamp2 =  data[int(cent2[0])-10:int(cent2[0])+10,int(cent2[1])-10:int(cent2[1])+10]
	stamp3 =  data[int(cent3[0])-10:int(cent3[0])+10,int(cent3[1])-10:int(cent3[1])+10]
	
	sigma_clip = SigmaClip(sigma=3.,cenfunc=null_background)	

	bkgrms = StdBackgroundRMS(sigma_clip)

	bkgrms_value1 = bkgrms.calc_background_rms(stamp1)
	bkgrms_value2 = bkgrms.calc_background_rms(stamp2)
	bkgrms_value3 = bkgrms.calc_background_rms(stamp3)

	fig,ax = plt.subplots(3,2)
	ax[0,0].imshow(stamp1,vmin=-1000,vmax=1000)
	ax[0,1].hist(stamp1.ravel(),50)

	ax[1,0].imshow(stamp2,vmin=-1000,vmax=1000)
	ax[1,1].hist(stamp2.ravel(),50)
	ax[2,0].imshow(stamp3,vmin=-1000,vmax=1000)
	ax[2,1].hist(stamp3.ravel(),50)
	print(np.std(stamp1.ravel()),np.std(stamp2.ravel()),np.std(stamp3.ravel()))
	print(bkgrms_value1,bkgrms_value2,bkgrms_value3)
	plt.show()
	apertures = CircularAperture([10,10], r=5)
	error = calc_total_error(stamp1, bkgrms_value1, 1)
	phot_table1 = aperture_photometry(stamp1, apertures, method='subpixel',
		                 error=error)

	error = calc_total_error(stamp1, np.std(stamp1.ravel()), 1)
	phot_table2 = aperture_photometry(stamp1, apertures, method='subpixel',
		                 error=error)
	
	import pdb; pdb.set_trace()
