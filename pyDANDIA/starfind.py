######################################################################
#                                                                   
# starfind.py - identify the stars in a given image.
#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+ 
#      scipy 0.15+
#      scikit-image 0.11+
#      scikit-learn 0.18+
#      matplotlib 1.3+
#      photutils 0.3.2+
#
# Developed by the Yiannis Tsapras
# as part of the ROME/REA LCO Key Project.
#
# version 0.1a (development)
#
# Last update: 17 Jul 2017
######################################################################

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch, AsymmetricPercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy import units as u
from photutils import background, detection, DAOStarFinder
from photutils import CircularAperture
import matplotlib.pyplot as plt

import time
from datetime import datetime
from sys import exit
import os

from config import read_config

def starfind(path_to_image, plot_it=False, write_log=True):
    '''
    The routine will quickly identify stars in a given image and return 
    a star list and image quality parameters. The output is to be used 
    to select suitable candidate images for constructing a template 
    reference image.
    '''
    t0 = time.time()
    im = fits.open(path_to_image)
    header = im[0].header
    scidata = im[0].data
    # Get size of image
    ymax, xmax = scidata.shape
    # Read configuration file and get saturation limit
    config = read_config('../Config/config.json')
    # If it is a large image, consider 250x250 pixel subregions and
    # choose the one with the fewest saturated pixels to evaluate stats
    saturation = config['maxval']['value']
    nr_sat_pix = 100000
    bestx1 = -1
    bestx2 = -1
    besty1 = -1
    besty2 = -1
    regionsx = np.arange(0, xmax, 250)
    regionsy = np.arange(0, ymax, 250)
    for i in regionsx[0:-1]:
        x1 = i
	x2 = i + 250
        for j in regionsy[0:-1]:
	    y1 = j
	    y2 = j + 250
	    nr_pix = len(scidata[y1:y2,x1:x2][np.where(scidata[y1:y2,x1:x2] > saturation)])
	    #print x1, x2, y1, y2, nr_pix
	    if nr_pix < nr_sat_pix:
	       nr_sat_pix = nr_pix
	       bestx1 = x1
	       bestx2 = x2
	       besty1 = y1
	       besty2 = y2
    
    #mean, median, std = sigma_clipped_stats(scidata[1:ymax, 1:xmax], sigma=3.0, iters=5)
    # Evaluate mean, median and standard deviation for the selected subregion
    mean, median, std = sigma_clipped_stats(scidata[besty1:besty2,bestx1:bestx2], sigma=3.0, iters=5)
    # Identify stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(scidata[besty1:besty2,bestx1:bestx2] - median)
    # Write steps to a log file
    if (write_log == True):
        logfile = open('starfind.log','a')
	logfile.write("Current system time: "+datetime.now().strftime("%Y:%m:%dT%H:%M:%S")+"\n")
        logfile.write("Identifying sources on image %s ...\n" % path_to_image.split('/')[-1])
        logfile.write("Found %s sources.\n" % str(len(sources)))
        if len(sources) == 0:
            logfile.write("Insufficient number of sources found. Stopping execution.\n")
	    exit('Could not detect enough sources on image.')
        elif (len(sources) > 0 and len(sources) <= 5):
            logfile.write("WARNING: Too few sources detected on image.\n")
        else:
            logfile.write("Using up to 30 best sources to determine FWHM.\n")
    
    # Discount saturated stars
    sources = sources[np.where(sources['peak'] < saturation)]
    sources.sort('peak')
    sources.reverse()
    # Keep only up to 100 stars
    sources = sources[0:100]
    
    sources_with_close_stars_ids = []
    # Discount stars with close neighbours (within r=5 pix)
    for i in np.arange(len(sources)):
	source_i = sources[i]
	for other_source in sources[i+1:]:
	    if (np.sqrt((source_i['xcentroid']-other_source['xcentroid'])**2 +
	             (source_i['ycentroid']-other_source['ycentroid'])**2) <= 5 ):
	        sources_with_close_stars_ids.append(i)
                #print source_i, other_source
		continue
    
    # Keep up to 30 isolated sources only (may be fewer)
    sources.remove_rows(sources_with_close_stars_ids)
    sources = sources[0:30]
    #return sources
    # Uncomment the following line to display source list in browser window:
    #sources.show_in_browser()
    
    # Fit a model to identified sources and estimate PSF shape parameters
    fwhm_arr = []
    fwhm_a_arr = []
    fwhm_b_arr = []
    ell_arr = []
    weights_arr = []
    i = 0
    while (i <= len(sources)):
        i_peak = sources[i]['peak']
	position = [sources[i]['xcentroid'], sources[i]['ycentroid']]
	stamp_size = (20,20) # in pixels
	cutout = Cutout2D(scidata, position, stamp_size)
	yc, xc = cutout.position_cutout
	y, x = np.mgrid[:20, :20]
	yy,xx = np.indices(cutout.data.shape)
	m_init = models.Moffat2D(amplitude=1, x_0=xc, y_0=yc)
	fit_m = fitting.LevMarLSQFitter()
	g = fit_m(m_init,xx,yy,cutout.data)
	plt.figure(figsize=(3,8))
	plt.subplot(3,1,1)
	plt.imshow(cutout.data, cmap='gray', origin='lower')
	plt.colorbar()
	plt.title("Data")
	plt.subplot(3,1,2)
	plt.imshow(g(xx,yy), cmap='gray', origin='lower')
	plt.colorbar()
	plt.title("Model")
	plt.subplot(3,1,3)
	plt.imshow(cutout.data-g(xx,yy), cmap='gray', origin='lower')
	plt.title("Residual")
	plt.colorbar()
        plt.show()
	i = i + 1
    
    # If plot_it is True, plot the sources found
    if plot_it == True:
        positions = (sources['xcentroid'], sources['ycentroid'])
	apertures = CircularAperture(positions, r=4.)	
	norm = ImageNormalize(interval=AsymmetricPercentileInterval(1,5), stretch=SqrtStretch())
	plt.imshow(scidata[besty1:besty2,bestx1:bestx2], cmap='gray', origin='lower', norm=norm)
	plt.title("Data")
	#plt.colorbar()
	apertures.plot(color='yellow', lw=1.2, alpha=0.5)
	plt.show()
	#plt.savefig('stars250x250.png')
    im.close()
    print "%.3f" % (time.time()-t0)
    return sources
    
def write_trendlog(path_to_trendlog_file, sources):
    if os.path.exists(path_to_trendlog_file):
        os.remove(path_to_trendlog_file)
    out = open(path_to_trendlog_file,'a')
    out.write("# Image : GJD (days) : HJD (days) : RA (deg) : Dec (deg) : Bias Level (ADU) : ")
    out.write("Readout Noise (ADU) : Bias Level And Readout Noise Tag (1 = Good : 0 = Bad) : ")
    out.write("Degree Of Polynomial Fitted To Overscan Region (-1 = No Fit) : Exptime (s) : ")
    out.write("Sky (ADU) : Sky Sigma (ADU) : FWHM (pix) : FWHM Major Axis (pix) : ")
    out.write("FWHM Minor Axis (pix) : Ellipticity : Number Of Peaks Used : ")
    out.write("Number Of Stars Detected At A Normalised Threshold Above The Sky Background\n")
    

#################################
# Command line section
if __name__ == '__main__':
    path_to_image = '/home/Tux/ytsapras/Programs/Workspace/development/pipeline_200x200_testdata/lsc1m005-fl15-20170614-0130-e91_cropped.fits'
    starfind(path_to_image, plot_it=True)
