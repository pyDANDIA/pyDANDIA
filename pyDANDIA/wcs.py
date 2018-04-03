# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:05:56 2017

@author: rstreet
"""
from os import path, getcwd
from sys import exit
from pyDANDIA import  logs
from pyDANDIA import  metadata
from astropy.io import fits
from astropy import wcs, coordinates, units, visualization
import subprocess
from astroquery.vizier import Vizier
from pyDANDIA import  catalog_utils
import numpy as np
import matplotlib.pyplot as plt


def reference_astrometry(setup,log,image_path,detected_sources,diagnostics=True):
    """Function to calculate the World Coordinate System (WCS) for an image"""
    
    log.info('Performing astrometry on the reference image')
    
    hdu = fits.open(image_path)
    
    image_wcs = wcs.WCS(hdu[0].header)
    log.info('Reference image initial WCS information:')
    log.info(image_wcs)
    
    wcs_image_path = path.join(setup.red_dir,'ref','ref_image_wcs.fits')
    catalog_file = path.join(setup.red_dir,'ref', 'star_catalog.fits')
    
    log.info('Querying ViZier for 2MASS sources within the field of view...')
    radius = hdu[0].header['NAXIS1']*image_wcs.wcs.cd[0,0]*60.0/2.0
    catalog_sources = search_vizier_for_2mass_sources(str(image_wcs.wcs.crval[0]), \
                                                      str(image_wcs.wcs.crval[1]), \
                                                     radius)
    log.info('ViZier returned '+str(len(catalog_sources))+' within the field of view')
    
    catalog_sources_xy = calc_image_coordinates(catalog_sources,image_wcs)
    log.info('Calculated image coordinates of 2MASS catalog sources')

    matched_stars = match_stars(detected_sources,catalog_sources_xy)
    log.info('Matched '+str(len(matched_stars))+' detected objects to stars in the 2MASS catalogue')
    
    offset_x = np.median(matched_stars[:,5]-matched_stars[:,2])
    offset_y = np.median(matched_stars[:,4]-matched_stars[:,1])
    log.info('Refining reference image WCS')
    log.info('Calculated x, y offsets: '+str(offset_x)+', '+str(offset_y)+' pix')
    
    if diagnostics == True:
        diagnostic_plots(path.join(setup.red_dir,'ref'),hdu,image_wcs,
                         detected_sources,
                         catalog_sources_xy,
                         matched_stars,offset_x,offset_y)
        log.info('-> Output astrometry diagnostic plots')

    (image_wcs,hdu) = update_image_wcs(hdu,image_wcs,offset_x,offset_y)
    hdu.writeto(wcs_image_path,overwrite=True)
    log.info('-> Output reference image with updated WCS:')
    log.info(image_wcs)
    
    ref_source_catalog = build_ref_source_catalog(detected_sources,\
                                                    catalog_sources,\
                                                    matched_stars,image_wcs)
    log.info('Build reference image source catalogue of '+str(len(ref_source_catalog))+' objects')
    
    catalog_utils.output_ref_catalog_file(catalog_file,ref_source_catalog)
    log.info('-> Output reference source catalogue')
    
    log.info('Completed astrometry of reference image')
    
    return ref_source_catalog
    
def run_imwcs(detected_sources,catalog_sources,input_image_path,output_image_path):
    """Function to run wcstools.imwcs to match detected to catalog objects and
    re-compute the WCS for an image
    
    THIS FUNCTION IS NOT IN USE - RETAINED FOR POSSIBLE FUTURE USE
    
    In testing, imwcs repeatedly reported errors in reading a pre-written catalog file
    ultimately, it simply didn't read the file, but can only query a
    catalog if given at a specific location on disk. 
    
    Although the pipeline could (and may in future) be designed to use standard
    catalogues (e.g. 2MASS PSC, USNO-B1.0 etc) stored locally, and recompute the
    WCS fit using imwcs' built-in querys, the astroquery 
    package currently provides the needed catalogue output, and tests show
    that refining the WCS fit from the image data is sufficient for our 
    purposes without introducing a dependency on locally-held catalogues.  
    This is open to future revision.
    """
    
    n_wcs_stars = 200
    n_param = 6
    
    path = 'imwcs'
    command = path
    args = ['-wvd '+detected_sources+\
                ' -c '+catalog_sources+\
                ' -h '+str(n_wcs_stars)+\
                ' -n '+str(n_param)+\
                ' -q irs '+input_image_path+\
                ' -o '+output_image_path]
    print( command, args)
    
    p = subprocess.Popen([command,args], stdout=subprocess.PIPE)
    p.wait()
    
def search_vizier_for_2mass_sources(ra, dec, radius):
    """Function to perform online query of the 2MASS catalogue and return
    a catalogue of known objects within the field of view"""
    
    v = Vizier(columns=['_RAJ2000', '_DEJ2000', 'Jmag', 'e_Jmag', \
                                    'Hmag', 'e_Hmag','Kmag', 'e_Kmag'],\
                column_filters={'Jmag':'<20'})
    v.ROW_LIMIT = 5000
    c = coordinates.SkyCoord(ra+' '+dec, frame='icrs', unit=(units.deg, units.deg))
    r = radius * units.arcminute
    result=v.query_region(c,radius=r,catalog=['2MASS'])
    
    return result[0]

def calc_image_coordinates(catalog_sources,image_wcs,verbose=False):
    """Function to calculate the x,y pixel coordinates of a set of stars
    specified by their RA, Dec positions, by applying the WCS from a FITS
    image header"""
    
    positions = []
    for star in catalog_sources['_RAJ2000','_DEJ2000'].as_array():
        positions.append( [star[0],star[1]] )
        if verbose == True:
            s = coordinates.SkyCoord(str(star[0])+' '+str(star[1]), frame='icrs', unit=(units.deg, units.deg))
            print( star, s.to_string('hmsdms'))
    positions = np.array(positions)
    
    return image_wcs.wcs_world2pix(positions,1)

def match_stars(detected_sources,catalog_sources_xy,verbose=False):
    """Function to match stars between the objects detected in an image
    and those extracted from a catalog, using image pixel postions."""
    
    matched_stars = []
    for i in range(0,len(catalog_sources_xy),1):
        cat_x = catalog_sources_xy[i,0]
        cat_y = catalog_sources_xy[i,1]
        
        dx = detected_sources[:,1]-cat_x
        dy = detected_sources[:,2]-cat_y
        sep = np.sqrt(dx*dx + dy*dy)
        
        idx = sep.argsort()

        matched_stars.append([idx[0],detected_sources[idx[0],1],detected_sources[idx[0],2],\
                            i, cat_x, cat_y, sep[idx[0]]])
        if verbose == True:
            print( matched_stars[-1])
            
    return np.array(matched_stars)
    
def update_image_wcs(hdu,image_wcs,offset_x,offset_y):
    """Function to update the WCS of an image given offsets in X, Y pixel
    position"""
    
    image_wcs.wcs.crpix[0] = image_wcs.wcs.crpix[0] + offset_x
    image_wcs.wcs.crpix[1] = image_wcs.wcs.crpix[1] + offset_y
    hdu[0].header['CRPIX1'] = image_wcs.wcs.crpix[0]
    hdu[0].header['CRPIX2'] = image_wcs.wcs.crpix[1]
    
    return image_wcs,hdu
    
def diagnostic_plots(output_dir,hdu,image_wcs,detected_sources,
                     catalog_sources_xy,matched_stars,offset_x,offset_y):
    """Function to output plots used to assess and debug the astrometry
    performed on the reference image"""

    norm = visualization.ImageNormalize(hdu[0].data, \
                        interval=visualization.ZScaleInterval())

    fig = plt.figure(1)
    plt.imshow(hdu[0].data, origin='lower', cmap=plt.cm.viridis, norm=norm)
    plt.plot(detected_sources[:,1],detected_sources[:,2],'o',markersize=2,\
             markeredgewidth=1,markeredgecolor='b',markerfacecolor='None')
    plt.plot(catalog_sources_xy[:,0],catalog_sources_xy[:,1],'r+')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig(path.join(output_dir,'reference_detected_sources_pixels.png'))
    plt.close(1)

    fig = plt.figure(1)
    fig.add_subplot(111, projection=image_wcs)
    plt.imshow(hdu[0].data, origin='lower', cmap=plt.cm.viridis, norm=norm)
    plt.plot(detected_sources[:,1],detected_sources[:,2],'o',markersize=2,\
             markeredgewidth=1,markeredgecolor='b',markerfacecolor='None')
    plt.plot(catalog_sources_xy[:,0],catalog_sources_xy[:,1],'r+')
    plt.xlabel('RA [J2000]')
    plt.ylabel('Dec [J2000]')
    plt.savefig(path.join(output_dir,'reference_detected_sources_world.png'))
    plt.close(1)

    fig = plt.figure(2)
    plt.subplot(211)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.95,
                wspace=0.1, hspace=0.3)
    plt.hist((matched_stars[:,5]-matched_stars[:,2]),50)
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.plot([offset_x,offset_x],[ymin,ymax],'r-')
    plt.xlabel('(Detected-catalog) X pixel')
    plt.ylabel('Frequency')
    plt.subplot(212)
    plt.hist((matched_stars[:,4]-matched_stars[:,1]),50)
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.plot([offset_y,offset_y],[ymin,ymax],'r-')
    plt.xlabel('(Detected-catalog) Y pixel')
    plt.ylabel('Frequency')
    plt.savefig(path.join(output_dir,'astrometry_separations.png'))
    plt.close(1)

def build_ref_source_catalog(detected_sources,catalog_sources,\
                            matched_stars,image_wcs):
    """Function to calculate the positions of all objects detected in the
    reference image in world coordinates, combining this catalogue with
    information from the 2MASS Point Source Catalogue where available.
    
    Output catalog is in numpy array format with columns:
    0   1  2  3   4     5        6             7         8        9   10   11 12   13   14   15
    idx x  y  ra  dec  ref_flux  ref_flux_err ref_mag ref_mag_err J  Jerr  H Herr   K   Kerr psf_star
    
    J, H, K magnitudes and their photometric errors are added if a given 
    star has been matched with the 2MASS PSC.  Similarly, instrumental 
    magnitudes and magnitude errors are given where available.  
    If no entry is available for these quantities, -99.999 values are given, 
    so that the catalog can be output to a valid FITS table. 
    """
    
    def validate_entry(cat_entry):
        """Function to intercept the NaNs used by Vizier to fill in data 
        missing from the catalog and replace them with the FITS-compatible
        null default used in the pipeline"""
    
        if str(cat_entry) == '--' or np.isnan(cat_entry) == True:
            cat_entry = -99.999
            
        return cat_entry
        
    world_coords = image_wcs.wcs_pix2world(detected_sources[:,1:3], 1)
    
    data = np.zeros([len(detected_sources),16])
        
    data[:,0] = detected_sources[:,0]   # Index
    data[:,1] = detected_sources[:,1]   # X
    data[:,2] = detected_sources[:,2]   # Y
    data[:,3] = world_coords[:,0]       # RA
    data[:,4] = world_coords[:,1]       # Dec
    data[:,7] = detected_sources[:,10]  # instrumental mag
    data[:,8] = -99.999                 # instrumental mag error (null)
    data[:,15] = 0                      # PSF star switch
    
    for j in range(0,len(matched_stars),1):
        idx = int(matched_stars[j,0])
        data[int(matched_stars[j,0]),9] = validate_entry(catalog_sources[int(matched_stars[j,3])][2])
        data[int(matched_stars[j,0]),10] = validate_entry(catalog_sources[int(matched_stars[j,3])][3])
        data[int(matched_stars[j,0]),11] = validate_entry(catalog_sources[int(matched_stars[j,3])][4])
        data[int(matched_stars[j,0]),12] = validate_entry(catalog_sources[int(matched_stars[j,3])][5])
        data[int(matched_stars[j,0]),13] = validate_entry(catalog_sources[int(matched_stars[j,3])][6])
        data[int(matched_stars[j,0]),14] = validate_entry(catalog_sources[int(matched_stars[j,3])][7])
        
    return data
    
