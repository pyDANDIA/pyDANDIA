import numpy as np
import matplotlib.pyplot as plt
from pyDANDIA import stage4
import os
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import background, detection, DAOStarFinder
from astropy.table import Table, Column
from photutils import aperture_photometry
from photutils import CircularAperture,CircularAnnulus,RectangularAperture
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
from astropy.table import QTable
from astropy.stats import SigmaClip
from photutils.utils import calc_total_error
from pyDANDIA import psf
from photutils import Background2D, MedianBackground

import matplotlib
matplotlib.use('TkAgg')

def final_phot(times,flux,eflux,pscal,exptime):

    aligned_phot = []
 
    Time = np.array(times)
    Flux = np.array(flux)
    Eflux = np.array(eflux)
    
    for i in range(len(flux[0])):
    
        aligned_flux = pscal[:,0]*Flux[:,i]/exptime
        aligned_eflux = np.sqrt(Eflux[:,i]**2*pscal[:,0]**2+Flux[:,i]**2*pscal[:,1]**2)/exptime
        
        
        aligned_phot.append(np.c_[Time,aligned_flux,aligned_eflux])
       
        
    return np.array(aligned_phot)
    
        
    

def phot_scales(flux,exptime,refind=0,sub_catalog=None):

    photscales = []
    for i in range(len(flux)):
    
        
        mask = (~np.isnan(flux[i])) & (~np.isnan(flux[refind]))
        
        if sub_catalog is not None:
        
            mask = mask & ([True if i in sub_catalog else False for i in range(len(flux[i]))])        
        #a,b = np.polyfit(fluxes[i][mask],fluxes[refind][mask],1)
        #a,_,_,_=np.linalg.lstsq(fluxes[i][mask,None],fluxes[0][mask,None])
        a = np.nanmedian(flux[refind][mask]/flux[i][mask]*exptime[i]/exptime[refind])
        sig_a = np.nanmedian(np.abs(flux[refind][mask]/flux[i][mask]*exptime[i]/exptime[refind]-a))
        photscales.append([a,sig_a])
        #breakpoint()
    return np.array(photscales)
    
        
def aperture_phot(data,pos,radius=3):

    rad = 2*radius
    apertures = CircularAperture(pos, r=rad)
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50),  filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    #breakpoint()
    ron = 0
    gain = 1
    
    error = calc_total_error(np.abs(data), (bkg.background_rms**2)**0.5, gain)
    error = (error**2+ron**2/gain**2)**0.5
    phot_table = aperture_photometry(data-bkg.background, apertures, method='subpixel',error=error)
    
    return phot_table
    
def extract_catalog(data_image):


    mean_data, median_data, std_data = sigma_clipped_stats(
        data_image, sigma=3.0, maxiters=5)
 
    daofind2 = DAOStarFinder(fwhm=3, threshold=3. * std_data,
                             exclude_border=True)
    
    data_sources = daofind2.find_stars(data_image - median_data)

    data_sources = data_sources[data_sources['flux'].argsort()[::-1]]

    yy,xx = np.indices((20,20))
    fwhms = []
    for i in np.arange(100,200,1):
    
        try:
            X = int(data_sources[i]['xcentroid'])
            Y = int(data_sources[i]['ycentroid'])
            
            stamp = data_image[Y-10:Y+10,X-10:X+10]
            fit = psf.fit_star(stamp,yy,xx)
            gamma,alpha = fit[0][3],fit[0][4]
            fwhms.append(gamma * 2 * (2 ** (1 / alpha) - 1) ** 0.5)

#            maximum = stamp.max()
#            mask = stamp>maximum/2
#            
#            max_row,max_col = np.where(stamp==maximum)

#            
#            dist = (yy-max_row)**2+ (xx-max_col)**2
#            
#            fwhms.append(np.max(dist[mask])**0.5+1)
#        breakpoint()
        except:
            pass
    #breakpoint()
    return data_sources,np.nanmedian(fwhms)
    
    
    
images = [i for i in os.listdir('./g22dkv/')]


#ref = fits.open('./g22dkv/lsc1m009-fa04-20230514-0064-e91.fits.fz') #gp
ref = fits.open('./g22dkv/lsc1m009-fa04-20230514-0063-e91.fits.fz')
ref_cat,ref_fwhm = extract_catalog(ref[1].data)
#ref_cat2 = ref_cat[:200]

dist = (ref_cat['xcentroid']-2051)**2+(ref_cat['ycentroid']-2040)**2
target = dist.argmin()
#ref_ind = np.where(np.array(images)=='lsc1m009-fa04-20230514-0064-e91.fits.fz')[0][0]
ref_ind = np.where(np.array(images)=='lsc1m009-fa04-20230514-0063-e91.fits.fz')[0][0]

times = []
fluxes = []
efluxes = []
fluxes2 = []
efluxes2 = []
exptime = []
fwhms = []

for im in images[::1]:

    data =  fits.open('./g22dkv/'+im)
    
    if data[1].header['FILTER'] == 'ip':

        data_cat,data_fwhm = extract_catalog(data[1].data)
        
        refcat = np.c_[ref_cat['xcentroid'],ref_cat['ycentroid'],ref_cat['flux']]
        datacat = np.c_[data_cat['xcentroid'],data_cat['ycentroid'],data_cat['flux']]
        
        align = stage4.find_init_transform(ref[1].data, data[1].data,
                                                   refcat[:250],
                                                   datacat[:250])
        
        align_cat = ref_cat.copy()
        
        xx,yy,zz = np.dot(np.linalg.pinv(align[0]),np.r_[[ref_cat['xcentroid'],ref_cat['ycentroid'],[1]*len(ref_cat['flux'])]])
        
        #align_cat['ycentroid'] = yy
        #align_cat['xcentroid'] = xx
        
        phot_table = aperture_phot(data[1].data,np.c_[xx,yy],radius=data_fwhm)
        phot_table2 = aperture_phot(data[1].data,np.c_[xx,yy],radius=3)
        
        fluxes.append(phot_table['aperture_sum'].value)
        efluxes.append(phot_table['aperture_sum_err'].value)
                
        fluxes2.append(phot_table2['aperture_sum'].value)
        efluxes2.append(phot_table2['aperture_sum_err'].value)
        
        times.append(data[1].header['MJD-OBS'])
        exptime.append(data[1].header['EXPTIME'])
        fwhms.append(data_fwhm)
#        from photutils.psf import PSFPhotometry
#        from astropy.table import QTable

#        from photutils.psf import IntegratedGaussianPRF
#        psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
#        fit_shape = (5, 5)
#        psfphot = PSFPhotometry(psf_model, fit_shape,aperture_radius=4)
#        psf_model.x_0.fixed=True
#        psf_model.y_0.fixed=True
#        init_params = QTable()
#        init_params['x'] = xx
#        init_params['y'] = yy
#        mask = (xx>0) & (yy>0) & (xx<4000) & (yy<4000)

#        phot = psfphot(data[1].data, init_params=init_params[mask])


    
pscale = phot_scales(fluxes,exptime,refind=0,sub_catalog=np.arange(100,200))
pscale2 = phot_scales(fluxes2,exptime,refind=0,sub_catalog=np.arange(100,200))
phot = final_phot(times,fluxes,efluxes,pscale,exptime)
phot2 = final_phot(times,fluxes2,efluxes2,pscale2,exptime)
breakpoint()
