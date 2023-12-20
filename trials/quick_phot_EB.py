import numpy as np
from astropy.io import fits
import os
from skimage.registration import phase_cross_correlation
from skimage import transform as tf
import matplotlib.pyplot as plt
import scipy.signal as ss
from photutils import background, detection, DAOStarFinder
from skimage.measure import ransac
from pyDANDIA import stage4
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import scipy.optimize as so
import scipy.ndimage as snd
from ois import optimal_system
from pyDANDIA import psf
from pycpd import RigidRegistration,AffineRegistration

import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.spatial as ssp


def find_phot_scale_factor(raw_flux, exptime):

    mask = ~np.any((raw_flux<0) | ~np.isfinite(raw_flux),axis=1)
    good_flux=raw_flux[mask]

    medians_flux = np.median(good_flux/exptime,axis=1)
    
    rescale = [good_flux[i]/medians_flux[i]/exptime for i in range(len(good_flux))]
    pscale = np.median(rescale,axis=0)
    epscale = (np.percentile(rescale,84,axis=0)-np.percentile(rescale,16,axis=0))/2

    return pscale,epscale
    
    
def psf_phot(image, star_positions,radius=10):

    size = radius

    flu = []
    eflu = []
    
    for i in np.arange(0,len(star_positions)):
    
        try:
            x = int((star_positions[i][0]))
            y = int((star_positions[i][1]))

            sub_im = image[y-size:y+size+1,x-size:x+size+1]   
            Y,X = np.indices(sub_im.shape)
           
            psfit =   psf.fit_star(sub_im,Y,X,psf_model='Moffat2D')
      
            psfmo = psf.Moffat2D()
            mo = psfmo.psf_model(Y,X,psfit[0])
            flu.append(mo.sum())
           
            cov = psfit[1]
           
            try:
            
                sig =  cov.diagonal()[0]**0.5
            
            except:
            
                sig = psfit[0][0]
                
            eflu.append(mo.sum()* sig/psfit[0][0])
           
        except:
        
            flu.append(-99)
            eflu.append(-99)
            
    return np.array(flu),np.array(eflu) 
        
        
def aperture_phot(image,star_positions,radius=10):

    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.aperture import ApertureStats   
    from photutils.aperture import aperture_photometry

    aperture = CircularAperture(star_positions, r=radius)
    annulus_aperture = CircularAnnulus(star_positions, r_in=radius+5, r_out=radius+10)

    phot_table = aperture_photometry(image, aperture,error=np.abs(image)**0.5)
    aperstats = ApertureStats(image, annulus_aperture)
   
    return phot_table['aperture_sum']-aperstats.mean*aperture.area, phot_table['aperture_sum_err']


def build_mat(params,imshape):


    #rotate around image centers
    center = imshape
    
    mat_shift  = np.eye(3)
    mat_shift[0,2] = params[1]
    mat_shift[1,2] = params[2]
    
    mat_center = np.eye(3)
    
    mat_center[0,2] = -center[0]/2-0.5
    mat_center[1,2] = -center[1]/2-0.5
    
    mat_rota =  np.array([[np.cos(params[0]),-np.sin(params[0]),0],[np.sin(params[0]),np.cos(params[0]),0],[0,0,1]])
    
    mat_anticenter = np.eye(3)
    mat_anticenter[0,2] = +center[0]/2
    mat_anticenter[1,2] = +center[1]/2
    
    mat_tot = np.dot(mat_anticenter,np.dot(mat_shift,np.dot(mat_rota,mat_center)))

    return mat_tot

def fit_the_points(params,ref,im,imshape):


    mat = np.array([[np.cos(params[0]),-np.sin(params[0]),params[1]],[np.sin(params[0]),np.cos(params[0]),params[2]],[0,0,1]])
   

    mat_tot = build_mat(params,imshape)
    new_points = np.dot(mat_tot,np.c_[im,[1]*len(im)].T)
    
    dist = ssp.distance.cdist(ref,new_points.T[:,:2])
    
    return np.sum(np.min(dist,axis=0))


def find_init_transform(ref,im,source_ref,source_im):
    
    reso = so.minimize(fit_the_points,[0,0,0],args=(source_ref,source_im,ref.shape),method='Powell')

    recenter = tf.AffineTransform(matrix=build_mat(reso['x'],ref.shape))
    recentred = tf.warp(im.astype(float), inverse_map=recenter.inverse, output_shape=im1.shape, order=3, mode='constant', cval=0, clip=True,preserve_range=True)
    corr = np.corrcoef(ref.ravel(),recentred.ravel())[0,1]
    print(reso['x'],np.corrcoef(ref.ravel(),im.ravel())[0,1],corr)
 
    recenter = tf.AffineTransform(matrix=np.linalg.pinv(build_mat(reso['x'],ref.shape)))

    return recenter,corr
  


ref = fits.open('./data/cpt1m012-fa06-20220110-0068-e91.fits')[0].data

daofind = DAOStarFinder(fwhm=3.0, threshold=2.*ref.std())
sources_ref = daofind(ref)
sources_ref = sources_ref[sources_ref['flux'].argsort()[::-1],]
 
images = [i for i in os.listdir('./data/') if 'fits' in i]


times = []
exptime = []
raw_flux = [] 
raw_eflux = []


photometry='PSF'

for im in images[:100]:

    thefit = fits.open('./data/'+im)
    
    im1 = thefit[0].data
    
    times.append(thefit[0].header['MJD-OBS'])
    exptime.append(thefit[0].header['exptime'])
    
    
    daofind = DAOStarFinder(fwhm=3.0, threshold=2.*im1.std())
    sources_im1 = daofind(im1)

    sources_im1 = sources_im1[sources_im1['flux'].argsort()[::-1],]
    
    pref = np.c_[sources_ref['xcentroid'],sources_ref['ycentroid']]
    pim1 = np.c_[sources_im1['xcentroid'],sources_im1['ycentroid']]


    init_transform,corr = find_init_transform(ref,im1,pref,pim1)


    
    pts_data, pts_reference, e_pos_data,e_pos_ref = stage4.crossmatch_catalogs2(sources_ref, sources_im1, init_transform)

    pts_reference2 = np.copy(pts_reference)

    model_robust, inliers = ransac((pts_reference2[:5000, :2] , pts_data[:5000, :2] ), tf.AffineTransform,min_samples=min(50, int(0.1 * len(pts_data[:5000]))),residual_threshold=1, max_trials=1000)
                                   
###Potential refinement
#        A = stage4.polyfit2d(pts_reference2[:,0][:5000][inliers], pts_reference2[:,1][:5000][inliers], pts_data[:,0][:5000][inliers], order=2)#,errors=e_pos_ref[:5000][inliers]/2**0.5)

#        B = stage4.polyfit2d(pts_reference2[:,0][:5000][inliers], pts_reference2[:,1][:5000][inliers], pts_data[:,1][:5000][inliers], order=2)#,errors=e_pos_ref[:5000][inliers]/2**0.5)
#        C = tf.PolynomialTransform(np.r_[[A],[B]])
    #model_final = model_robust
    #warped = tf.warp(im1.astype(float), inverse_map=model_final, output_shape=im1.shape, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
    #print(np.corrcoef(ref.ravel(),warped.ravel())[0,1])

    positions = model_robust(pref)
    
    if photometry=='Aperture':
    
            flux,eflux = aperture_phot(im1,positions, radius = 10) 

         
    if photometry=='PSF': #Slower obviously

            flux,eflux = psf_phot(im1,positions, radius = 10) 
        
    raw_flux.append(flux)
    raw_eflux.append(eflux)
    
breakpoint()

raw_fluxes = np.array(raw_flux).T
raw_efluxes = np.array(raw_eflux).T

pscale,epscale = find_phot_scale_factor(raw_fluxes, exptime)

error_tot = np.sqrt(raw_efluxes**2/pscale**2+raw_fluxes**2/pscale**4*epscale**2)

#Gaia21ccu == stars 287

        
breakpoint()

