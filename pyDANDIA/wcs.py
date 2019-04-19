# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:05:56 2017

@author: rstreet
"""
from os import path, getcwd
from sys import exit
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  vizier_tools
from pyDANDIA import  match_utils
from astropy.io import fits
from astropy import wcs, coordinates, units, visualization, table
import subprocess
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from pyDANDIA import  catalog_utils
from pyDANDIA import shortest_string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize

def reference_astrometry(setup,log,image_path,detected_sources,diagnostics=True):
    """Function to calculate the World Coordinate System (WCS) for an image"""
    
    log.info('Performing astrometry on the reference image')
    
    hdu = fits.open(image_path)
    
    image_wcs = wcs.WCS(hdu[0].header)
    field = hdu[0].header['OBJECT']
    
    log.info('Reference image initial WCS information:')
    log.info(image_wcs)
    
    wcs_image_path = path.join(setup.red_dir,'ref','ref_image_wcs.fits')
    catalog_file = path.join(setup.red_dir,'ref', 'star_catalog.fits')
    
    gaia_sources = fetch_catalog_sources_for_field(setup,field,hdu[0].header,
                                                      image_wcs,log,'Gaia')
    
    gaia_sources_xy = calc_image_coordinates(gaia_sources,image_wcs,log)
    
    (detected_subregion, gaia_subregion) = extract_central_region_from_catalogs(detected_sources, 
                                                                                gaia_sources_xy,
                                                                                log)
                                                    
    transform = shortest_string.find_xy_offset(detected_subregion, 
                                               gaia_subregion, log=log)
                                               
    image_wcs = update_wcs(image_wcs,transform,log)
    
    world_coords = calc_world_coordinates(detected_sources,image_wcs,log)
    
    if diagnostics:
        plot_overlaid_sources(path.join(setup.red_dir,'ref'),
                          world_coords,gaia_sources)
    
    matched_stars = match_stars_world_coords(world_coords,gaia_sources,log,
                                             verbose=True)
    
    analyze_coord_residuals(matched_stars,world_coords,gaia_sources,
                            path.join(setup.red_dir,'ref'))
    
    #(matched_stars,image_wcs) = refine_wcs(detected_sources, gaia_sources_xy, 
    #                               image_wcs, log, path.join(setup.red_dir,'ref'))

#    if diagnostics == True:
#        diagnostic_plots(path.join(setup.red_dir,'ref'),hdu,image_wcs,
#                         detected_sources,
#                         gaia_sources_xy)
#        log.info('-> Output astrometry diagnostic plots')

    (image_wcs,hdu) = update_image_wcs(hdu,image_wcs)
    hdu.writeto(wcs_image_path,overwrite=True)
    log.info('-> Output reference image with updated WCS:')
    log.info(image_wcs)
    
    ref_source_catalog = build_ref_source_catalog(detected_sources,\
                                                    gaia_sources,\
                                                    matched_stars,image_wcs)

    log.info('Build reference image source catalogue of '+\
             str(len(ref_source_catalog))+' objects')
    
    catalog_utils.output_ref_catalog_file(catalog_file,ref_source_catalog)
    log.info('-> Output reference source catalogue')
    
    log.info('Completed astrometry of reference image')
    
    return ref_source_catalog

def fetch_catalog_sources_for_field(setup,field,header,image_wcs,log,
                                    catalog_name):
    """Function to read or fetch a catalogue of sources for the field.
    catalog_name indicates the data origin, one of 'Gaia' or 'VPHAS'
    """
    
    catalog_file = path.join(setup.pipeline_config_dir,
                             field+'_'+catalog_name+'_catalog.fits')
    
    catalog_sources = catalog_utils.read_vizier_catalog(catalog_file,catalog_name)
    
    if catalog_sources != None:
        
        log.info('Read data for '+str(len(catalog_sources))+\
                 ' '+catalog_name+' stars from stored catalog for field '+field)
        
    else:
        
        log.info('Querying ViZier for '+catalog_name+' sources within the field of view...')
    
        radius = header['NAXIS1']*header['PIXSCALE']/60.0/2.0
        
        ra = image_wcs.wcs.crval[0]
        dec = image_wcs.wcs.crval[1]
        
        if catalog_name in ['VPHAS', '2MASS']:
            
            catalog_sources = vizier_tools.search_vizier_for_sources(ra, dec, 
                                                                     radius, 
                                                                     catalog_name, 
                                                                     row_limit=2000)
            
        else:
            
            catalog_sources = vizier_tools.search_vizier_for_gaia_sources(str(ra), \
                                                                          str(dec), 
                                                                          radius)
        
        log.info('ViZier returned '+str(len(catalog_sources))+\
                 ' within the field of view')
        
        catalog_utils.output_vizier_catalog(catalog_file, catalog_sources, 
                                            catalog_name)
    
    return catalog_sources

def extract_central_region_from_catalogs(detected_sources, gaia_sources_xy,log):
    """Function to extract the positions of stars in the central region of an
    image"""
    
    log.info('Extracting sub-catalogs of detected and catalog sources within the central region of the field of view')
    
    mid_x = int((detected_sources[:,1].max() - detected_sources[:,1].min())/2.0)
    mid_y = int((detected_sources[:,2].max() - detected_sources[:,2].min())/2.0)
    
    dx = 100
    dy = 100
    
    detected_subregion = np.zeros(1)
    gaia_subregion = np.zeros(1)
    it = 0
    max_it = 3
    cont = True
    
    while cont:
        
        log.info(' -> Trying subsection dx='+str(dx)+', dy='+str(dy))
        
        xmin = max(mid_x-dx,detected_sources[:,1].min())
        xmax = min(mid_x+dx,detected_sources[:,1].max())
        ymin = max(mid_y-dy,detected_sources[:,2].min())
        ymax = min(mid_y+dy,detected_sources[:,2].max())
        
        idx1 = np.where(detected_sources[:,1] >= xmin)[0]
        idx2 = np.where(detected_sources[:,1] <= xmax)[0]
        idx3 = np.where(detected_sources[:,2] >= ymin)[0]
        idx4 = np.where(detected_sources[:,2] <= ymax)[0]
        idx = set(idx1).intersection(set(idx2))
        idx = set(idx).intersection(set(idx3))
        idx = list(set(idx).intersection(set(idx4)))
        
        detected_subregion = detected_sources[idx,:]
        detected_subregion = detected_subregion[:,[1,2]]
        
        log.info(' -> '+str(len(detected_subregion))+\
                 ' detected stars')
        
        idx1 = np.where(gaia_sources_xy[:,0] >= xmin)[0]
        idx2 = np.where(gaia_sources_xy[:,0] <= xmax)[0]
        idx3 = np.where(gaia_sources_xy[:,1] >= ymin)[0]
        idx4 = np.where(gaia_sources_xy[:,1] <= ymax)[0]
        idx = set(idx1).intersection(set(idx2))
        idx = set(idx).intersection(set(idx3))
        idx = list(set(idx).intersection(set(idx4)))
        
        gaia_subregion = gaia_sources_xy[idx,:]
        gaia_subregion = gaia_subregion[:,[0,1]]
    
        log.info(' -> '+str(len(gaia_subregion))+\
                 ' stars from the Gaia catalog')
                 
        if (len(detected_subregion) < 100 or len(gaia_subregion) < 100) and \
            (it <= max_it):
            
            dx *= 2.0
            dy *= 2.0
            
            log.info(' => Increasing the sub-region size')
            
        elif (len(detected_subregion) > 1000 or len(gaia_subregion) > 1000) and \
            (it <= max_it):
            
            dx /= 2.0
            dy /= 2.0
            
            log.info(' => Reducing the sub-region size')
            
        else:
            
            cont = False
        
        it += 1
             
    return detected_subregion, gaia_subregion

def analyze_coord_residuals(matched_stars,world_coords,gaia_sources,output_dir):
    """Function to analyse the residuals between the RA, Dec positions
    calculated for stars detected in the image, and Gaia RA, Dec positions
    for matched stars."""
    
    dra = np.array(matched_stars.cat1_ra) - np.array(matched_stars.cat2_ra)
    ddec = np.array(matched_stars.cat1_dec) - np.array(matched_stars.cat2_dec)

    plot_astrometry(output_dir,matched_stars,pfit=None)
    
    
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
    print( command, args )
    
    p = subprocess.Popen([command,args], stdout=subprocess.PIPE)
    p.wait()
    
def search_vizier_for_vphas_sources(params,log):
    """Function to extract the objects from the VPHAS+ catalogue within the
    field of view of the reference image, based on the metadata information."""
    
    params['radius'] = (np.sqrt(params['fov'])/2.0)*60.0
    
    log.info('Search radius: '+str(params['radius'])+' arcmin')
    
    vphas_cat = vizier_tools.search_vizier_for_sources(params['ra'], 
                                                       params['dec'], 
                                                        params['radius'], 
                                                        'VPHAS+')
        
    log.info('VPHAS+ search returned '+str(len(vphas_cat))+' entries')
    
    return vphas_cat
    
def calc_image_coordinates(catalog_sources,image_wcs,log,verbose=False):
    """Function to calculate the x,y pixel coordinates of a set of stars
    specified by their RA, Dec positions, by applying the WCS from a FITS
    image header"""
    
    log.info('Calculating the predicted image pixel positions for all catalog stars')
    
    positions = []
    
    for star in catalog_sources['ra','dec'].as_array():

        positions.append( [star[0],star[1]] )

        if verbose == True:
            s = coordinates.SkyCoord(str(star[0])+' '+str(star[1]), 
                                     frame='icrs', unit=(units.deg, units.deg))
            print( star, s.to_string('hmsdms'))

    positions = np.array(positions)
    
    log.info('Completed calculation of image coordinates')
    
    return image_wcs.wcs_world2pix(positions,1)


def calc_world_coordinates(detected_sources,image_wcs,log):
    """Function to calculate the RA, Dec positions of an array of image
    pixel positions"""
    
    log.info('Calculating the world coordinates of all detected stars')
    
    world_coords = image_wcs.wcs_pix2world(detected_sources[:,1:3], 1)
    
    table_data = [ table.Column(name='ra', data=world_coords[:,0]),
                   table.Column(name='dec', data=world_coords[:,1]),
                   table.Column(name='x', data=detected_sources[:,1]),
                   table.Column(name='y', data=detected_sources[:,2]),
                    table.Column(name='index', data=detected_sources[:,0]) ]
    
    coords_table = table.Table(data=table_data)
    
    log.info('Completed calculation of world coordinates')
    
    return coords_table
    
    
def match_stars_world_coords(detected_sources,catalog_sources,log,
                             verbose=False):
    """Function to match stars between the objects detected in an image
    and those extracted from a catalog, using image pixel postions."""
    
    log.info('Matching detected and catalog sources via their world coordinates')
    
    tol = 1.0/3600.0
    dra = 30.0/3600.0
    ddec = 30.0/3600.0
    
    det_sources = coordinates.SkyCoord(detected_sources['ra'], 
                                       detected_sources['dec'], 
                                       frame='icrs', 
                                       unit=(units.deg, units.deg))

    matched_stars = match_utils.StarMatchIndex()
    
    jincr = int(float(len(catalog_sources))*0.01)
    
    for j in range(0,len(catalog_sources),1):
        
        c = coordinates.SkyCoord(catalog_sources['ra'][j], 
                                 catalog_sources['dec'][j], 
                                 frame='icrs', unit=(units.deg, units.deg))
        
        kdx1 = np.where(detected_sources['ra'] >= (c.ra.value-dra))[0]
        kdx2 = np.where(detected_sources['ra'] <= (c.ra.value+dra))[0]
        kdx3 = np.where(detected_sources['dec'] >= (c.dec.value-ddec))[0]
        kdx4 = np.where(detected_sources['dec'] <= (c.dec.value+ddec))[0]
        kdx = set(kdx1).intersection(set(kdx2))
        kdx = kdx.intersection(set(kdx3))
        kdx = list(kdx.intersection(set(kdx4)))
        
        (idx, d2d, d3d) = c.match_to_catalog_sky(det_sources[kdx])
        i = int(idx)
        
        if d2d.value < tol:
            
            p = {'cat1_index': detected_sources['index'][kdx[i]],
                 'cat1_ra': detected_sources['ra'][kdx[i]],
                 'cat1_dec': detected_sources['dec'][kdx[i]],
                 'cat1_x': detected_sources['x'][kdx[i]],
                 'cat1_y': detected_sources['y'][kdx[i]],
                 'cat2_index': j, 
                 'cat2_ra': catalog_sources['ra'][j], 
                 'cat2_dec': catalog_sources['dec'][j], \
                 'separation': d2d.value[0]}
                 
            matched_stars.add_match(p)
            
            if verbose:
                log.info(matched_stars.summarize_last())
            
            if j%jincr == 0:
                percentage = round((float(j)/float(len(catalog_sources)))*100.0,0)
                log.info(' -> Completed cross-match of '+str(percentage)+\
                            '% ('+str(j)+' of catalog stars out of '+\
                            str(len(catalog_sources))+')')

    log.info(' -> Matched '+str(matched_stars.n_match)+' stars')

    log.info('Completed star match in world coordinates')
    
    return matched_stars


def refine_wcs(detected_sources,catalog_sources_xy,image_wcs,
               log,output_dir,verbose=False):
    """Function to iteratively refine the WCS of the reference frame by 
    matching the detected stars against the catalog and calculating the 
    offset and RMS"""
    
    log.info('Refining reference image WCS')
    
    image_positions = detected_sources[:,[0,1,2]]
    
    i = 0
    max_iter = 3
    cont = True
    
    pinit = [0.0, 0.0]
    
    sigma_old = 1e6
    
    while i <= max_iter and cont:
        
        matched_stars = match_stars(image_positions,catalog_sources_xy)
        log.info('Iteration '+str(i)+' matched '+str(len(matched_stars))+\
                ' detected objects to stars in the catalogue')
        
        (pfit,sigma) = fit_coordinate_transform(pinit, image_positions, catalog_sources_xy, 
                             matched_stars)
                             
        log.info('Parameters of coordinate transform fit: '+repr(pfit)+\
                    ' sigma = '+str(sigma))

        plot_astrometry(output_dir,matched_stars,pfit)
        
        (xprime,yprime) = transform_coords(pfit,image_positions[:,1],detected_sources[:,2])
        
        image_positions[:,1] = xprime
        image_positions[:,2] = yprime
        
        image_wcs = update_wcs(image_wcs,pfit,log)
        
        if abs(sigma-sigma_old) < 1e-3 or pfit.all() == 0.0:
            cont = False
        
        i += 1
        sigma_old = sigma
    
    return matched_stars,image_wcs
    
def transform_coords(p,x,y):
    
    if len(p) == 2:
        
        xprime = x + p[0]
        yprime = y + p[1]
        
    elif len(p) == 4:
        
        xprime = x + p[0] + p[1]*x
        yprime = y + p[2] + p[3]*y
        
    elif len(p) == 6:
        
        xprime = x + p[0] + p[1]*x + p[2]*y
        yprime = y + p[3] + p[4]*x + p[5]*y
    
    return xprime, yprime
    
def errfunc(p,x,y,xc,yc):
    """Function to calculate the residuals on the photometric transform"""
    
    (xprime,yprime) = transform_coords(p,x,y)
    
    return np.sqrt( (xc-xprime)**2 + (yc-yprime)**2 )
    
def fit_coordinate_transform(pinit, detected_sources, catalog_sources_xy, 
                             matched_stars):
    """Function to calculate the photometric transformation between a set
    of catalogue magnitudes and the instrumental magnitudes for the same stars
    """
    
    cat_x = matched_stars[:,4]
    cat_y = matched_stars[:,5]
    
    det_x = matched_stars[:,1]
    det_y = matched_stars[:,2]
    
    (pfit,iexec) = optimize.leastsq(errfunc,pinit,
                                    args=(det_x,det_y,cat_x,cat_y))
    
    sigma = calc_transform_uncertainty(pfit,matched_stars)
    
    return pfit,sigma
    

def calc_transform_uncertainty(pfit,matched_stars):
    
    (xprime,yprime) = transform_coords(pfit,matched_stars[:,1],matched_stars[:,2])
    
    dx = matched_stars[:,4]-xprime
    dy = matched_stars[:,5]-yprime
    
    sep = np.sqrt( (dx*dx) + (dy*dy) )
    
    return sep.std()

def update_wcs(image_wcs,transform,log):
    """Function to update the WCS object for an image"""
    
    (xprime,yprime) = transform_coords(transform,image_wcs.wcs.crpix[1],
                                                image_wcs.wcs.crpix[0])
        
    image_wcs.wcs.crpix[0] = yprime
    image_wcs.wcs.crpix[1] = xprime
    
    log.info('Updated image WCS information')
    
    return image_wcs
    
def update_image_wcs(hdu,image_wcs):
    """Function to update the WCS of an image given offsets in X, Y pixel
    position"""
    
    hdu[0].header['CRPIX1'] = image_wcs.wcs.crpix[0]
    hdu[0].header['CRPIX2'] = image_wcs.wcs.crpix[1]
    
    return image_wcs,hdu
    
def diagnostic_plots(output_dir,hdu,image_wcs,detected_sources,
                     catalog_sources_xy):
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

def plot_overlaid_sources(output_dir,detected_sources_world,gaia_sources_world):
    matplotlib.use('TkAgg')
    
    fig = plt.figure(1)
    plt.plot(detected_sources_world['ra'], detected_sources_world['dec'], 'r+',
             alpha=0.5,label='Detected sources')
    plt.plot(gaia_sources_world['ra'], gaia_sources_world['dec'], 'bo',
             fillstyle='none', label='Gaia sources')
    plt.xlabel('RA deg')
    plt.ylabel('Dec deg')
    plt.legend()
    
    plt.savefig(path.join(output_dir,'detected_sources_world_overlay.png'))
    plt.close(1)
    
def plot_astrometry(output_dir,matched_stars,pfit=None):

    dra = np.array(matched_stars.cat2_ra)-np.array(matched_stars.cat1_ra)
    ddec = np.array(matched_stars.cat2_dec)-np.array(matched_stars.cat1_dec)
    
    fig = plt.figure(1)
    
    plt.subplot(211)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.95,
                wspace=0.1, hspace=0.3)

    plt.hist(dra,50)

    (xmin,xmax,ymin,ymax) = plt.axis()
        
    plt.xlabel('(Catalog-detected) RA')
    plt.ylabel('Frequency')

    plt.subplot(212)

    plt.hist(ddec,50)

    (xmin,xmax,ymin,ymax) = plt.axis()
    
    plt.xlabel('(Catalog-detected) Dec')
    plt.ylabel('Frequency')

    plt.savefig(path.join(output_dir,'astrometry_separations.png'))
    plt.close(1)

    fig = plt.figure(2,(10,10))

    plt.subplot(221)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.95,
                wspace=0.5, hspace=0.3)

    plt.plot(np.array(matched_stars.cat1_x),dra,
             'b.',markersize=1)

    plt.xlabel('X pixel')
    plt.ylabel('$\\Delta$ RA')

    plt.subplot(222)
    
    plt.plot(np.array(matched_stars.cat1_y),dra,
             'b.',markersize=1)
             
    plt.xlabel('Y pixel')
    plt.ylabel('$\\Delta$ RA')

    plt.subplot(223)

    plt.plot(np.array(matched_stars.cat1_x),ddec,
             'm.',markersize=1)

    plt.xlabel('X pixel')
    plt.ylabel('$\\Delta$ Dec')
    
    plt.subplot(224)

    plt.plot(np.array(matched_stars.cat1_y),ddec,
             'm.',markersize=1)

    plt.xlabel('Y pixel')
    plt.ylabel('$\\Delta$ Dec')
    
    plt.savefig(path.join(output_dir,'detected_catalog_positions.png'))
    plt.close(2)

def build_ref_source_catalog(detected_sources,catalog_sources,\
                            matched_stars,image_wcs):
    """Function to calculate the positions of all objects detected in the
    reference image in world coordinates, combining this catalogue with
    information from the 2MASS Point Source Catalogue where available.
    
    Output catalog is in numpy array format with columns (2MASS):
    0   1  2  3   4     5        6             7         8        9   10   11 12   13   14   15-19   20
    idx x  y  ra  dec  ref_flux  ref_flux_err ref_mag ref_mag_err J  Jerr  H Herr   K   Kerr null  psf_star
    
    Output catalog is in numpy array format with columns (Gaia):
    0   1  2  3   4     5        6             7         8            9   
    idx x  y  ra  dec  ref_flux  ref_flux_err ref_mag ref_mag_err gaia_source_id 
    10      11           12       13            14              15
    ra    ra_error      dec     dec_error   phot_g_mean_flux phot_g_mean_flux_error 
    16                      17    
    phot_bp_mean_flux phot_bp_mean_flux_error 
    18                  19                          20
    phot_rp_mean_flux phot_rp_mean_flux_error    psf_star
    
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
    
    cat_source = 'Gaia'
    
    world_coords = image_wcs.wcs_pix2world(detected_sources[:,1:3], 1)
    
    data = np.zeros([len(detected_sources),21])
        
    data[:,0] = detected_sources[:,0]   # Index
    data[:,1] = detected_sources[:,1]   # X
    data[:,2] = detected_sources[:,2]   # Y
    data[:,3] = world_coords[:,0]       # RA
    data[:,4] = world_coords[:,1]       # Dec
    data[:,7] = detected_sources[:,10]  # instrumental mag
    data[:,8] = -99.999                 # instrumental mag error (null)
    data[:,20] = 0                      # PSF star switch
    
    for j in range(0,len(matched_stars.cat1_index),1):
        
        idx1 = int(matched_stars.cat1_index[j])
        idx2 = int(matched_stars.cat2_index[j])
        
        if cat_source == '2MASS':
            data[idx1,9] = validate_entry(catalog_sources[matched_stars.cat2_index[j],2])
            data[idx1,10] = validate_entry(catalog_sources[matched_stars.cat2_index[j],3])
            data[idx1,11] = validate_entry(catalog_sources[matched_stars.cat2_index[j],4])
            data[idx1,12] = validate_entry(catalog_sources[matched_stars.cat2_index[j],5])
            data[idx1,13] = validate_entry(catalog_sources[matched_stars.cat2_index[j],6])
            data[idx1,14] = validate_entry(catalog_sources[matched_stars.cat2_index[j],7])
        
        elif cat_source == 'Gaia':
            data[idx1,9] = validate_entry(catalog_sources['source_id'][idx2])
            data[idx1,10] = validate_entry(catalog_sources['ra'][idx2])
            data[idx1,11] = validate_entry(catalog_sources['ra_error'][idx2])
            data[idx1,12] = validate_entry(catalog_sources['dec'][idx2])
            data[idx1,13] = validate_entry(catalog_sources['dec_error'][idx2])
            data[idx1,14] = validate_entry(catalog_sources['phot_g_mean_flux'][idx2])
            data[idx1,15] = validate_entry(catalog_sources['phot_g_mean_flux_error'][idx2])
            data[idx1,16] = validate_entry(catalog_sources['phot_bp_mean_flux'][idx2])
            data[idx1,17] = validate_entry(catalog_sources['phot_bp_mean_flux_error'][idx2])
            data[idx1,18] = validate_entry(catalog_sources['phot_rp_mean_flux'][idx2])
            data[idx1,19] = validate_entry(catalog_sources['phot_rp_mean_flux_error'][idx2])
        
        else:
            raise IOError('Unrecognized catalog source '+catalog_source)
            exit()
            
    return data
    
