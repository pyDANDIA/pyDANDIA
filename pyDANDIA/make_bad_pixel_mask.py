# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 13:58:32 2018

@author: rstreet
"""
from os import path
from sys import argv
from astropy.io import fits
import numpy as np
from astropy.visualization import ZScaleInterval

def bad_pixel_analysis():
    """Driver function to analyse imaging data for bad pixels, and to create
    a mask of them which can be used in data reduction.
    """
    
    (image, header, params) = load_data()
    
    bpm = make_bad_pixel_mask(image, params)
    
    output_bpm(bpm, header, params)    

def load_data():
    """Function to gather the required input parameters"""
    
    params = {}
    
    if len(argv) == 1:
        
        skyflat = raw_input('Please enter the path to the fullframe flat field to analyse: ')
        params['option'] = raw_input('Which pixel selection method would you like to use? [zscale, sigma_clip]: ')
        
        if 'sigma_clip' in params['option']:
            
            params['nsigma_hi'] = float(raw_input('Please enter the factor of the image sigma to use as hi pixel threshold: '))
            params['nsigma_lo'] = float(raw_input('Please enter the factor of the image sigma to use as lo pixel threshold: '))
        
        else:
            
            params['percentage'] = float(raw_input('Please enter the percentage by which to scale the zscale interval: '))
            
    else:
        skyflat = argv[1]
        params['option'] = argv[2]
        
        if 'sigma_clip' in params['option']:
            
            params['nsigma_hi'] = float(argv[3])
            params['nsigma_lo'] = float(argv[4])
            
        else:
            
            params['percentage'] = float(argv[3])
            
    if path.isfile(skyflat):
        
        f = fits.open(skyflat)
        
        image = f[0].data
        header = f[0].header
        
        f.close()
        
    else:
        
        print('ERROR: Cannot find input flat field at '+skyflat)
        
        exit()
    
    params['data_dir'] = path.dirname(skyflat)
    
    return image, header, params
    
def make_bad_pixel_mask(image,params):
    """Function to identify pixels of lower sensitivity or artificially high
    count value, based on an array of image data.  
    This should normally be provided from a fullframe sky flat.
    """
    
    #image = adjust_quadrant_gains(image)
    
    if 'sigma_clip' in params['option']:
        
        print('Applying thresholds sigma cut limits')
        
        (mean,stddev) = calc_stddev_clip(image[50:image.shape[0]-50,50:image.shape[1]-50],
                                    1.0,1)
    
        print('Image mean and stddev: '+str(mean)+' '+str(stddev))
        
        thresh_hi = mean+float(params['nsigma_hi'])*stddev
        thresh_lo = mean-float(params['nsigma_lo'])*stddev
        
    else:
        
        print('Applying zscale limits as thresholds')
        
        interval = ZScaleInterval()
        (zthresh_lo, zthresh_hi) = interval.get_limits(image[50:image.shape[0]-50,50:image.shape[1]-50])
    
        print('Image zscale thresholds: '+str(zthresh_lo)+' '+str(zthresh_hi))
    
        thresh_hi = zthresh_hi * (1.0 + params['percentage']/100.0)
        thresh_lo = zthresh_lo * (1.0 - params['percentage']/100.0)
        
    print('Applying thresholds HI='+str(thresh_hi)+'ADU, LO='+str(thresh_lo)+'ADU')
        
    idx1 = np.where(image > thresh_hi)
    idx2 = np.where(image < thresh_lo)
    
    overscan_mask = mask_residual_overscan(image)
    
    # Generate FITS image of the Bad Pixel Mask:
    bpm = np.zeros([image.shape[0],image.shape[1]])
    bpm[idx1] = 1.0
    bpm[idx2] = 1.0

    idx = np.where(overscan_mask != 0.0)
    
    bpm[idx] = overscan_mask[idx]
    
    bpm = sanitize_bpm_edges(bpm)
    
    return bpm
    
def adjust_quadrant_gains(image):
    """Function to renormalize the median level of each quadrant, since the
    4-quadrant readout isn't always perfectly level.
    Code imported from SinistroToolKit code by R. Street.    
    """
    
    nsigma = 3.0
    niter = 3
    
    ymin = 40
    ymax = 4060
    xmin = 0
    xmax = 2048
    ylim = 2048
    xlim = 2048
    
    q1 = image[ymin:ylim,xmin:xmax]
    q2 = image[ymin:ylim,xlim:]
    q3 = image[ylim:ymax,xlim:]
    q4 = image[ylim:ymax,xmin:xlim]
    
    q1idx = np.where(q1 > 0.0)
    q2idx = np.where(q2 > 0.0)
    q3idx = np.where(q3 > 0.0)
    q4idx = np.where(q4 > 0.0)

    (q1mean,stdev) = calc_stddev_clip(q1[q1idx],nsigma,niter)
    (q2mean,stdev) = calc_stddev_clip(q2[q2idx],nsigma,niter)
    (q3mean,stdev) = calc_stddev_clip(q3[q3idx],nsigma,niter)
    (q4mean,stdev) = calc_stddev_clip(q4[q4idx],nsigma,niter)
    
    # Re-normalize Q2 relative to Q1: 
    q1 = image[0:2048,0:2048]
    q2 = image[0:2048,2048:]
    q3 = image[2048:,2048:]
    q4 = image[2048:,0:2048]
    
    #print q1[q1idx].mean(dtype='float64'),q2[q2idx].mean(dtype='float64')
    q2 = q2 * (q1mean/q2mean)
    image[0:2048,2048:] = q2
    
    # Re-normalize Q3 relative to Q4: 
    #print q4[q4idx].mean(dtype='float64'),q3[q3idx].mean(dtype='float64')
    q3 = q3 * (q4mean/q3mean)
    image[2048:,2048:] = q3
    
    # Re-normalize top row of quadrants relative to the bottom row:
    q12 = image[0:2048,:]
    q12idx = np.where(q12 > 0.0)
    (q12mean,stdev) = calc_stddev_clip(q12[q12idx],nsigma,niter)
    
    q34 = image[2048:,:]
    q34idx = np.where(q34 > 0.0)
    (q34mean,stdev) = calc_stddev_clip(q34[q34idx],nsigma,niter)
    
    q34 = q34 * (q12mean/q34mean)
    image[2048:,:] = q34
    
    return image

def calc_stddev_clip(data,nsigma,niter):
    """Function to calculate the mean and standard deviation of a 
    pixel values, given an image region data array, 
    using an iterative, sigma-clipping function
    """
    
    data = data.flatten()
    
    data = data[np.logical_not(np.isnan(data))]
    
    idx = np.where(data < 1e9)

    for it in range(0,niter,1):
        
        mean = data[idx].mean(dtype='float64')
        std = data[idx].std(dtype='float64')
        
        idx1 = np.where(data >= (mean-nsigma*std))
        idx2 = np.where(data <= (mean+nsigma*std))
        idx = np.intersect1d(idx1[0],idx2[0])

    mean = data[idx].mean(dtype='float64')
    std = data[idx].std(dtype='float64')
    
    return mean, std

def mask_residual_overscan(image):
    """Function to mask the residual overscans of Sinistro images.  
    This can occur at the top and bottom of reduced frames, where the
    pixel value is 1 or very close to it for all pixels in the row, 
    and which are not habitually trimmed by the BANZAI pipeline to avoid
    creating problems for downstream software such as the NRES pipeline.
    """
    
    bpm = np.zeros([image.shape[0],image.shape[1]])
    
    for a in [0, 1]:
        
        kdx = range(0,30,1) + range(image.shape[a]-30,image.shape[a],1)
        
        if a == 0:
            region = image[:,kdx]
        else:
            region = image[kdx,:]
        
        med = np.median(region,axis=a)
        
        idx1 = np.where(med > 0.9999)
        idx2 = np.where(med < 1.0001)
        
        idx = list(np.intersect1d(idx1[0],idx2[0]))
        
        for i in idx:
            if a == 0:
                bpm[:,kdx[i]] = 1.0
            else:
                bpm[kdx[i],:] = 1.0
    
    return bpm

def sanitize_bpm_edges(bpm):
    """Function to review the edges of the Bad Pixel Mask.
    The overscan mask covers the very edge of the image, but sometimes a gap
    occurs between this and the ragged edge.  The data in this region are 
    unsuitable for photometry, but not identified by the previous statistics. 
    This function is designed to identify and close the gaps around the edges
    of the mask."""
    
    pixel_regions = [ range(50,0,-1), range(bpm.shape[1]-50,bpm.shape[1],1) ]
    
    for region in pixel_regions:
        
        flag = False
        
        for i in region:
            
            med = np.median(bpm[i,:])
            stddev = bpm[i,:].std()
            
            idx = np.where(bpm[i,:] == 1.0)[0]
            
            frac = float(len(idx))/float(bpm.shape[1])
            
            if frac > 0.5:
                
                flag = True
            
            if flag:
                
                bpm[i,:] = 1.0
            
    return bpm
    
def output_bpm(bpm, header, params):
    """Function to output the Bad Pixel Mask as a FITS image.
    The header of the original image is retained for reference.
    """
    
    hdu = fits.PrimaryHDU(bpm, header=header)
    
    camera = header['INSTRUME']
    date = str(header['DATE-OBS']).split('T')[0].replace('-','')
    
    file_name = path.join(params['data_dir'],'bpm_'+camera+'_'+date+'.fits')
    
    hdu.writeto(file_name, overwrite=True)
    
    print('Bad pixel mask for '+camera+' written to '+file_name)


if __name__ == '__main__':
    
    bad_pixel_analysis()
    