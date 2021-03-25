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
from pyDANDIA import  utilities
from astropy.io import fits
from astropy import wcs, coordinates, units, visualization, table
import subprocess
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from pyDANDIA import  catalog_utils
from pyDANDIA import  shortest_string
from pyDANDIA import  calc_coord_offsets
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize


def reference_astrometry(setup,log,image_path,detected_objects,
                         find_transform=False,
                         diagnostics=True):
    """Function to calculate the World Coordinate System (WCS) for an image"""

    log.info('Performing astrometry on the reference image')

    detected_sources = build_detect_source_catalog(detected_objects)

    header = fits.getheader(image_path)

    image_wcs = wcs.WCS(header)

    field = header['OBJECT']

    log.info('Reference image initial WCS information:')
    log.info(image_wcs)

    wcs_image_path = path.join(setup.red_dir,'ref','ref_image_wcs.fits')
    catalog_file = path.join(setup.red_dir,'ref', 'star_catalog.fits')
    det_catalog_file = path.join(setup.red_dir,'ref', 'detected_stars_full.reg')
    cat_catalog_file = path.join(setup.red_dir,'ref', 'catalog_stars_full.reg')

    detected_sources = calc_world_coordinates_astropy(setup,image_wcs,
                                              detected_sources,log)

    gaia_sources = fetch_catalog_sources_for_field(setup,field,header,
                                                      image_wcs,log,'Gaia')

    gaia_sources = calc_image_coordinates_astropy(setup, image_wcs, gaia_sources,log)

    catalog_utils.output_ds9_overlay_from_table(detected_sources,det_catalog_file,colour='green')
    catalog_utils.output_ds9_overlay_from_table(gaia_sources,cat_catalog_file,colour='blue')

    (bright_central_detected_stars, bright_central_gaia_stars) = extract_bright_central_stars(setup,detected_sources,
                        gaia_sources, image_wcs, log, radius=0.05)

    if find_transform:

        bright_central_detected_pix = shortest_string.extract_pixel_catalog(bright_central_detected_stars)
        bright_central_gaia_pix = shortest_string.extract_pixel_catalog(bright_central_gaia_stars)

        transform = shortest_string.find_xy_offset(bright_central_detected_pix,
                                                   bright_central_gaia_pix, log=log)

        #transform = calc_coord_offsets.calc_offset_hist2d(setup,bright_central_detected_stars,
        #                                                  bright_central_gaia_stars,
        #                                                  image_wcs.wcs.crval[0],
        #                                                  image_wcs.wcs.crval[1],
        #                                                  log,
        #                                                  diagnostics=True)

        image_wcs = update_wcs(image_wcs,transform,header['PIXSCALE'],log,
                               transform_type='pixels')

        detected_sources = calc_world_coordinates_astropy(setup,image_wcs,
                                                      detected_sources,log)

        gaia_sources = calc_image_coordinates_astropy(setup, image_wcs, gaia_sources,log)

        cat_catalog_file = path.join(setup.red_dir,'ref', 'catalog_stars_full_updated.reg')
        catalog_utils.output_ds9_overlay_from_table(gaia_sources,cat_catalog_file,colour='red')

    if diagnostics:
        plot_overlaid_sources(path.join(setup.red_dir,'ref'),
                          detected_sources,gaia_sources, interactive=False)

    matched_stars = match_stars_world_coords(detected_sources,gaia_sources,log,'Gaia',
                                             verbose=True)

    analyze_coord_residuals(matched_stars,detected_sources,gaia_sources,
                            path.join(setup.red_dir,'ref'))

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
                             str(field).replace(' ','-').replace('/','-')+'_'+catalog_name+'_catalog.fits')

    catalog_sources = catalog_utils.read_vizier_catalog(catalog_file,catalog_name)

    if catalog_sources != None:

        log.info('Read data for '+str(len(catalog_sources))+\
                 ' '+catalog_name+' stars from stored catalog for field '+field+', '+\
                 catalog_file)

    else:

        log.info('Querying ViZier for '+catalog_name+' sources within the field of view...')

        diagonal = np.sqrt(header['NAXIS1']*header['NAXIS1'] + header['NAXIS2']*header['NAXIS2'])
        radius = diagonal*header['PIXSCALE']/60.0/2.0 #arcminutes

        ra = image_wcs.wcs.crval[0]
        dec = image_wcs.wcs.crval[1]

        if catalog_name in ['VPHAS', '2MASS', 'Gaia-DR2']:

            catalog_sources = vizier_tools.search_vizier_for_sources(ra, dec,
                                                                     radius,
                                                                     catalog_name,
                                                                     row_limit=-1,
                                                                     coords='degrees')

        else:

            #catalog_sources = vizier_tools.search_vizier_for_gaia_sources(str(ra), \
            #                                                              str(dec),
            #                                                              radius,
            #                                                              log=log)

            log.info('ERROR: Attempt to query unsupported catalog '+catalog_name)
            raise IOError('ERROR: Attempt to query unsupported catalog '+catalog_name)

        log.info('ViZier returned '+str(len(catalog_sources))+\
                 ' within the field of view')

        catalog_utils.output_vizier_catalog(catalog_file, catalog_sources,
                                            catalog_name)

    return catalog_sources

def extract_bright_central_stars(setup, detected_sources, catalog_sources,
                                 image_wcs, log, radius=0.1):
    """Function to extract the bright stars from the central regions of
    both detected and catalog starlists

    radius float degrees
    """

    log.info('Selecting bright, central stars within '+str(radius)+\
             ' deg of the image centre to calculate the transformation')

    central_detected_stars = calc_coord_offsets.extract_nearby_stars(detected_sources,
                                                                     image_wcs.wcs.crval[0],
                                                                     image_wcs.wcs.crval[1],
                                                                     radius)

    central_catalog_stars = calc_coord_offsets.extract_nearby_stars(catalog_sources,
                                                                     image_wcs.wcs.crval[0],
                                                                     image_wcs.wcs.crval[1],
                                                                     radius)

    idx = np.argsort(central_detected_stars['ref_flux'])

    imin = int(len(central_detected_stars)*0.75)
    imax = int(len(central_detected_stars)*0.98)

    bright_central_detected_stars = central_detected_stars[idx][imin:imax]

    log.info('Selected '+str(len(bright_central_detected_stars))+\
             ' bright detected stars close to the centre of the image')

    jdx = []

    for i,flux in enumerate(central_catalog_stars['phot_rp_mean_flux']):
        if np.isfinite(flux):
            jdx.append(i)

    idx = np.argsort(central_catalog_stars['phot_rp_mean_flux'][jdx])

    imin = int(len(central_catalog_stars[jdx])*0.75)
    imax = int(len(central_catalog_stars[jdx])*0.98)

    bright_central_catalog_stars = central_catalog_stars[jdx][idx[imin:imax]]

    if 'x1' not in bright_central_catalog_stars.colnames:
        bright_central_catalog_stars.add_column( table.Column(name='x1', data=np.copy(bright_central_catalog_stars['x'])) )
        bright_central_catalog_stars.add_column( table.Column(name='y1', data=np.copy(bright_central_catalog_stars['y'])) )

    log.info('Selected '+str(len(bright_central_catalog_stars))+\
             ' bright detected stars close to the centre of the image')

    det_catalog_file = path.join(setup.red_dir,'ref', 'bright_detected_stars.reg')
    cat_catalog_file = path.join(setup.red_dir,'ref', 'bright_catalog_stars.reg')
    catalog_utils.output_ds9_overlay_from_table(bright_central_detected_stars,
                                                det_catalog_file,colour='yellow')
    catalog_utils.output_ds9_overlay_from_table(bright_central_catalog_stars,
                                                cat_catalog_file,colour='magenta')
    #print(radius)

    if (len(bright_central_detected_stars)==0) | (len(bright_central_catalog_stars)==0):

        return extract_bright_central_stars(setup, detected_sources, catalog_sources, image_wcs, log, 2*radius)

    else:

        return bright_central_detected_stars, bright_central_catalog_stars, radius

def extract_central_region_from_catalogs(detected_sources, gaia_sources, log):
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

def calc_image_coordinates(setup, image_path, catalog_sources,log):
    """Function to calculate the x,y pixel coordinates of a set of stars
    specified by their RA, Dec positions, by applying the WCS from a FITS
    image header"""

    log.info('Calculating the predicted image pixel positions for all catalog stars')


    coords_file = path.join(setup.red_dir,'ref','catalog_stars.txt')

    coords_list = open(coords_file,'w')

    for star in catalog_sources['ra','dec'].as_array():

        s = coordinates.SkyCoord(str(star[0])+' '+str(star[1]),
                                     frame='icrs', unit=(units.deg, units.deg))

        coords_list.write(s.to_string('hmsdms',sep=':')+'\n')

    coords_list.close()

    # Note that this does NOT apply the CD distortion matrix, so this
    # replaced with a call to wcstools:
    #positions =  image_wcs.wcs_world2pix(np.array(positions),1)

    cmd = 'sky2xy'
    args = [cmd, image_path, '@'+coords_file]

    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    (output,err) = p.communicate()

    positions = []
    for entry in (output.decode('ascii')).split('\n'):

        if len(entry) > 0:
            x = entry.split()[4]
            y = entry.split()[5]

            positions.append([x,y])

    positions = np.array(positions)

    catalog_sources.add_column( table.Column(name='x', data=positions[:,0]) )
    catalog_sources.add_column( table.Column(name='y', data=positions[:,1]) )

    log.info('Completed calculation of image coordinates')

    return catalog_sources

def calc_image_coordinates_astropy(setup, image_wcs, catalog_sources,log,
                                    stellar_density,
                                    rotate_wcs, kwargs,
                                    stellar_density_threshold,
                                    radius=None):

    def transform_positions(dpositions, image_wcs, log):

        log.info('Applying WCS CD transform parameters')

        dpositions[:,0] = dpositions[:,0] - image_wcs.wcs.crpix[0]
        dpositions[:,1] = dpositions[:,1] - image_wcs.wcs.crpix[1]

        theta = np.pi
        R = np.zeros((2,2))
        R[0,0] = np.cos(theta)
        R[0,1] = -np.sin(theta)
        R[1,0] = np.sin(theta)
        R[1,1] = np.cos(theta)

        positions = np.dot(dpositions, R)

        positions[:,0] += image_wcs.wcs.crpix[0]
        positions[:,1] += image_wcs.wcs.crpix[1]

        return positions

    log.info('Calculating the predicted image pixel positions for catalog stars')
    log.info('Current image_wcs: '+repr(image_wcs))

    positions = []
    jidx = None
    for j,star in enumerate(catalog_sources['ra','dec'].as_array()):

        s = coordinates.SkyCoord(str(star[0])+' '+str(star[1]),
                                     frame='icrs', unit=(units.deg, units.deg))

        positions.append([s.ra.deg, s.dec.deg])

    positions = np.array(positions)
    dpositions =  image_wcs.wcs_world2pix(np.array(positions),1)

    if kwargs['force_rotate_ref']:
        log.info('Rotation of the reference image WCS FORCED')
        positions = transform_positions(dpositions, image_wcs, log)

    else:
        if stellar_density < stellar_density_threshold:

            log.info('Since stellar density is low (< '+\
                str(utilities.stellar_density_threshold())+\
                '), assuming WCS in correct orientation')
            positions = dpositions

        elif kwargs['trust_wcs']:
            log.info('Trust WCS flag set, assuming WCS in correct orientation')
            positions = dpositions

        else:

            if rotate_wcs:

                log.info('Since stellar density is high (> '+\
                            str(utilities.stellar_density_threshold())+\
                            '), and the image rotation flag is set to TRUE in the pipeline configuration, applying image rotation.')

                if image_wcs.wcs.cd[0,0] < 0:

                    positions = transform_positions(dpositions, image_wcs, log)

                else:

                    log.info('The image WCS parameters suggest this image has already been rotated appropriately, so NO futher rotation will be made')

                    positions = dpositions

            else:

                log.info('The stellar density is high (> '+\
                            str(utilities.stellar_density_threshold())+\
                            '), but the image rotation flag is switched OFF in the pipeline configuration, so not applying image rotation.')

                positions = dpositions

    if 'x' in catalog_sources.colnames:
        catalog_sources['x'] = positions[:,0]
        catalog_sources['y'] = positions[:,1]

    else:
        catalog_sources.add_column( table.Column(name='x', data=positions[:,0]) )
        catalog_sources.add_column( table.Column(name='y', data=positions[:,1]) )

    log.info('Completed calculation of image coordinates')

    return catalog_sources

def calc_world_coordinates(setup,image_path,detected_sources,log):
    """Function to calculate the RA, Dec positions of an array of image
    pixel positions"""

    log.info('Calculating the world coordinates of all detected stars')

    coords_file = path.join(setup.red_dir,'ref','detected_stars.txt')

    coords_list = open(coords_file,'w')

    for j in range(0,len(detected_sources),1):

        coords_list.write(str(detected_sources[j,1])+' '+str(detected_sources[j,2])+'\n')

    coords_list.close()

    # REPLACED USE OF WCS module because it doesn't handle image distortions
    #world_coords = image_wcs.wcs_pix2world(detected_sources[:,1:3], 1)

    cmd = 'xy2sky'
    args = [cmd, image_path, '@'+coords_file]

    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    (output,err) = p.communicate()

    world_coords = []
    for entry in (output.decode('ascii')).split('\n'):

        if len(entry) > 0:
            ra = entry.split()[0]
            dec = entry.split()[1]

            c = coordinates.SkyCoord(ra, dec, frame='icrs', unit=(units.hourangle, units.deg))

            world_coords.append([c.ra.deg, c.dec.deg])

    world_coords = np.array(world_coords)

    if len(world_coords) == len(detected_sources):
        table_data = [ table.Column(name='ra', data=world_coords[:,0]),
                       table.Column(name='dec', data=world_coords[:,1]),
                       table.Column(name='x', data=detected_sources[:,1]),
                       table.Column(name='y', data=detected_sources[:,2]),
                       table.Column(name='index', data=detected_sources[:,0]),
                       table.Column(name='flux', data=detected_sources[:,9]) ]

        coords_table = table.Table(data=table_data)

        log.info('Completed calculation of world coordinates')

    else:

        raise IOError('Could not properly convert all detected source pixel positions to RA, Dec')

    return coords_table

def calc_world_coordinates_astropy(setup,image_wcs,detected_sources,log,
                                   rotate=False, verbose=False):
    """Function to calculate the RA, Dec positions of an array of image
    pixel positions"""

    log.info('Calculating the world coordinates of detected stars')

    # Now apply the known image rotation
    if rotate:
        dpositions = np.zeros([len(detected_sources),2])
        dpositions[:,0] = detected_sources['x'].data
        dpositions[:,1] = detected_sources['y'].data

        dpositions[:,0] = dpositions[:,0] - image_wcs.wcs.crpix[0]
        dpositions[:,1] = dpositions[:,1] - image_wcs.wcs.crpix[1]

        theta = -np.pi
        R = np.zeros((2,2))
        R[0,0] = np.cos(theta)
        R[0,1] = -np.sin(theta)
        R[1,0] = np.sin(theta)
        R[1,1] = np.cos(theta)

        positions = np.dot(dpositions, R)

        positions[:,0] += image_wcs.wcs.crpix[0]
        positions[:,1] += image_wcs.wcs.crpix[1]

    else:
        positions = np.zeros([len(detected_sources),2])
        positions[:,0] = detected_sources['x'].data
        positions[:,1] = detected_sources['y'].data


    world_coords = image_wcs.wcs_pix2world(positions, 1)

    detected_sources['ra'] = world_coords[:,0]
    detected_sources['dec'] = world_coords[:,1]

    if verbose:
        for j in range(0,len(detected_sources),1):
            log.info(detected_sources['x'][j],detected_sources['y'][j],' -> ',
                  detected_sources['ra'][j],detected_sources['dec'][j])

    log.info('Completed calculation of world coordinates')

    return detected_sources

def match_stars_world_coords(detected_sources,catalog_sources,log,catalog_name,
                             radius=None, ra_centre=None, dec_centre=None,
                             verbose=False, max_radius=None):
    """Function to match stars between the objects detected in an image
    and those extracted from a catalog, using image world postions."""

    log.info('Matching detected and '+catalog_name+' catalog sources via their world coordinates')


    if catalog_sources is None:

        return None

    tol = 1.0/3600.0
    dra = 30.0/3600.0
    ddec = 30.0/3600.0

    det_sources = coordinates.SkyCoord(detected_sources['ra'],
                                       detected_sources['dec'],
                                       frame='icrs',
                                       unit=(units.deg, units.deg))

    cat_sources = coordinates.SkyCoord(catalog_sources['ra'],
                                       catalog_sources['dec'],
                                       frame='icrs',
                                       unit=(units.deg, units.deg))

    if radius != None:

        if max_radius == None:

            centre = coordinates.SkyCoord(ra_centre, dec_centre,
                                      frame='icrs', unit=(units.deg, units.deg))

            separations = centre.separation(cat_sources)

            jdx = np.where(abs(separations.deg) <= radius)[0]

            log.info('Selected '+str(len(jdx))+' catalog stars centred around '+\
                 str(ra_centre)+', '+str(dec_centre)+' within radius '+str(radius)+'deg')

        else:

            nstars = 0

            while nstars < 1000 and radius < max_radius:
                centre = coordinates.SkyCoord(ra_centre, dec_centre,
                                      frame='icrs', unit=(units.deg, units.deg))

                separations = centre.separation(cat_sources)

                jdx = np.where(abs(separations.deg) <= radius)[0]

                nstars = len(jdx)

                log.info('Selected '+str(nstars)+' catalog stars centred around '+\
                 str(ra_centre)+', '+str(dec_centre)+' within radius '+str(radius)+'deg')

                if nstars == 0 and radius < max_radius:
                    radius = min(radius * 1.1, max_radius)

                    log.info(' -> Insufficient stars selected, so increased search radius to '+str(radius)+'deg')

    else:

        jdx = np.arange(0,len(cat_sources),1)

        log.info('All catalog stars selected')

    matched_stars = match_utils.StarMatchIndex()

    jincr = int(float(len(catalog_sources))*0.01)

    nm = 0

    for j in jdx:

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

        if len(kdx) > 0:

            (idx, d2d, d3d) = c.match_to_catalog_sky(det_sources[kdx])
            i = int(idx)

            if d2d.value < tol:

                add_star = True

                # Check for any pre-existing matches to this detected objects,
                # and replace the entry if the current catalog star is a
                # better match.
                if kdx[i] in matched_stars.cat1_index:

                    kk = matched_stars.cat1_index.index(kdx[i])

                    if d2d.value[0] < matched_stars.separation[kk]:

                        matched_stars.remove_match(kk)

                        nm -= 1

                        add_star = True

                    else:

                        add_star = False

                if add_star:

                    p = {'cat1_index': kdx[i],
                         'cat1_ra': detected_sources['ra'][kdx[i]],
                         'cat1_dec': detected_sources['dec'][kdx[i]],
                         'cat1_x': detected_sources['x'][kdx[i]],
                         'cat1_y': detected_sources['y'][kdx[i]],
                         'cat2_index': j,
                         'cat2_ra': catalog_sources['ra'][j],
                         'cat2_dec': catalog_sources['dec'][j],
                         'separation': d2d.value[0]}

                    matched_stars.add_match(p)
                    nm += 1

                    if verbose:
                        log.info(matched_stars.summarize_last(units='deg'))

            if j%jincr == 0:
                percentage = round((float(j)/float(len(catalog_sources)))*100.0,0)
                log.info(' -> Completed cross-match of '+str(percentage)+\
                            '% ('+str(j)+' of catalog stars out of '+\
                            str(len(catalog_sources))+')')

    log.info(' -> Matched '+str(matched_stars.n_match)+' stars')
    log.info(' -> Sanity check matched star count: '+str(nm))

    check_sanity = False
    if check_sanity:
        calc_coord_offsets.identify_inlying_matched_stars(matched_stars, log)

    log.info('Completed star match in world coordinates')

    return matched_stars

def cross_match_star_catalogs(detected_sources, catalog_sources, star_index, log,
                                dra=30.0, ddec=30.0, tol=1.0):
    """Function to identify those stars which are present in both detected
    and catalog source Tables based on their RA, Dec positions, while eliminating
     duplicate entries.

     Inputs:
     detected_sources  Table   Table of stars detected in working dataset
     catalog_sources   Table   Table of stars from reference catalog
     star_index        int array Array of Table indices to use in cross-match NOT star IDs
     log               logger  Open logging object
     dra               float   Search box width around each star [arcsec]
     ddec              float   Search box height around each star [arcsec]
     tol               float   Match tolerance in arcsec
     """

    # Convert search and match parameters to decimal degrees
    tol = tol/3600.0
    dra = dra/3600.0
    ddec = ddec/3600.0

    det_sources = coordinates.SkyCoord(detected_sources['ra'],
                                       detected_sources['dec'],
                                       frame='icrs',
                                       unit=(units.deg, units.deg))

    matched_stars = match_utils.StarMatchIndex()

    jincr = int(float(len(catalog_sources))*0.01)

    for j in star_index:
        c = coordinates.SkyCoord(catalog_sources['ra'][j],
                                 catalog_sources['dec'][j],
                                 frame='icrs', unit=(units.deg, units.deg))

        # Returns star array indices in the detected_sources array
        nearest_stars_index = select_nearest_stars_in_catalog(catalog_sources, detected_sources,
                                            c,dra,ddec)

        matched_stars = match_star_without_duplication(c,j,det_sources,nearest_stars_index,
                                            detected_sources, catalog_sources,
                                            tol,matched_stars,log,verbose=True)

        if jincr > 0 and j%jincr == 0:
            percentage = round((float(j)/float(len(catalog_sources)))*100.0,0)
            log.info(' -> Completed cross-match of '+str(percentage)+\
                        '% ('+str(j)+' of catalog stars out of '+\
                        str(len(catalog_sources))+')')

    log.info(' -> Matched '+str(matched_stars.n_match)+' stars after first pass')

    # The function above replaces matched_stars entries if a closer match
    # is found.  This can result in star entries being dropped if they happen
    # to lie close to more than one potential match.  Here we loop through
    # the list of remaining unmatched stars again, to see if other matches
    # are possible.
    if matched_stars.n_match < len(star_index):

        for j in star_index:

            if j not in matched_stars.cat2_index:

                c = coordinates.SkyCoord(catalog_sources['ra'][j],
                                         catalog_sources['dec'][j],
                                         frame='icrs', unit=(units.deg, units.deg))

                nearest_stars_index = select_nearest_stars_in_catalog(catalog_sources, detected_sources,
                                                    c,dra,ddec)

                # Remove from the list of potential matches all stars that have
                # already been matched
                revised_nearest_stars = []
                for jj in nearest_stars_index:
                    if jj not in matched_stars.cat2_index:
                        revised_nearest_stars.append(jj)

                matched_stars = match_star_without_duplication(c,j,
                                                    det_sources,revised_nearest_stars,
                                                    detected_sources, catalog_sources,
                                                    tol,matched_stars,log,verbose=True)

    log.info(' -> Matched '+str(matched_stars.n_match)+' stars after second pass')

    return matched_stars

def select_nearest_stars_in_catalog(catalog_sources, detected_sources,
                                    catalog_star,dra,ddec):
    """Function to identify the detected_source array indices of stars close
    to catalog_sources star j
    Returns a list of array indices of nearby stars in detected_sources
    """

    kdx1 = np.where(detected_sources['ra'] >= (catalog_star.ra.value-dra))[0]
    kdx2 = np.where(detected_sources['ra'] <= (catalog_star.ra.value+dra))[0]
    kdx3 = np.where(detected_sources['dec'] >= (catalog_star.dec.value-ddec))[0]
    kdx4 = np.where(detected_sources['dec'] <= (catalog_star.dec.value+ddec))[0]
    kdx = set(kdx1).intersection(set(kdx2))
    kdx = kdx.intersection(set(kdx3))
    nearest_stars_index = list(kdx.intersection(set(kdx4)))

    return nearest_stars_index

def match_star_without_duplication(catalog_star,cat_idx,det_sources,nearest_stars_index,
                                    detected_sources, catalog_sources,
                                    tol,matched_stars,log,verbose=True):

    if len(nearest_stars_index) > 0:
        (idx, d2d, d3d) = catalog_star.match_to_catalog_sky(det_sources[nearest_stars_index])
        i = int(idx)
        match_star_id = detected_sources['star_id'][nearest_stars_index[i]]

        if d2d.value < tol:

            add_star = True

            # Check for any pre-existing matches to this detected objects,
            # and replace the entry if the current catalog star is a
            # better match.
            #if nearest_stars_index[i] in matched_stars.cat1_index:
            if match_star_id in matched_stars.cat1_index:

                kk = matched_stars.cat1_index.index(match_star_id)

                if matched_stars.cat2_index[kk] != catalog_sources['star_id'][cat_idx]:
                    if d2d.value[0] < matched_stars.separation[kk]:

                        matched_stars.remove_match(kk)

                        add_star = True

                        if verbose:
                            log.info('Replacing previous match')

                    else:

                        add_star = False

                        if verbose:
                            log.info('Existing match at a smaller separation:')
                            log.info(' -> Catalog 2 star '+\
                                        str(matched_stars.cat2_index[kk])+' at '+\
                                        str(matched_stars.separation[kk])+' - retaining')
                            log.info(' -> compared with Catalog 2 star '+\
                                        str(catalog_sources['star_id'][cat_idx])+' at '+\
                                        str(d2d.value[0])+'\n')

                else:
                    if verbose:
                        log.info('Existing match of same star (catalog 2 star '+\
                                    str(matched_stars.cat2_index[kk])+' at '+\
                                    str(matched_stars.separation[kk])+'), retaining\n')

            if add_star:

                # Records star array indices NOT star ID
                p = {'cat1_index': match_star_id,
                     'cat1_ra': detected_sources['ra'][nearest_stars_index[i]],
                     'cat1_dec': detected_sources['dec'][nearest_stars_index[i]],
                     'cat1_x': detected_sources['x'][nearest_stars_index[i]],
                     'cat1_y': detected_sources['y'][nearest_stars_index[i]],
                     'cat2_index': catalog_sources['star_id'][cat_idx],
                     'cat2_ra': catalog_sources['ra'][cat_idx],
                     'cat2_dec': catalog_sources['dec'][cat_idx],
                     'cat2_x': catalog_sources['x'][cat_idx],
                     'cat2_y': catalog_sources['y'][cat_idx],
                     'separation': d2d.value[0]}

                matched_stars.add_match(p)

                if verbose:
                    log.info(matched_stars.summarize_last(units='deg'))

        else:
            if verbose:
                log.info('Catalog 2 star array index '+str(cat_idx)+': Nearest match outside tolerance')

    else:
        if verbose:
            log.info('Catalog 2 star array index '+str(cat_idx)+': No nearby catalog stars to match to')

    return matched_stars

def match_stars_pixel_coords(detected_sources,catalog_sources,log,
                             radius=None, x_centre=None, y_centre=None,
                             verbose=False, tol=1.5):
    """Function to match stars between the objects detected in an image
    and those extracted from a catalog, using image pixel postions."""

    log.info('Matching detected and catalog sources via their pixel coordinates')

    #tol = 1.5
    dpix = 10.0

    if radius != None:

        dx = catalog_sources['x1'].data - x_centre
        dy = catalog_sources['y1'].data - y_centre
        separations = np.sqrt( dx*dx + dy*dy )

        jdx = np.where(abs(separations) <= radius)[0]

        log.info('Selected '+str(len(jdx))+' catalog stars centred around '+\
                 str(x_centre)+', '+str(y_centre))
    else:

        jdx = np.arange(0,len(catalog_sources),1)

        log.info('All catalog stars selected')

    matched_stars = match_utils.StarMatchIndex()

    jincr = int(float(len(catalog_sources))*0.01)

    for j in jdx:
        cat_x = catalog_sources['x1'][j]    # Transformed coordinates
        cat_y = catalog_sources['y1'][j]

        kdx1 = np.where(detected_sources['x'] >= (cat_x-dpix))[0]
        kdx2 = np.where(detected_sources['x'] <= (cat_x+dpix))[0]
        kdx3 = np.where(detected_sources['y'] >= (cat_y-dpix))[0]
        kdx4 = np.where(detected_sources['y'] <= (cat_y+dpix))[0]
        kdx = set(kdx1).intersection(set(kdx2))
        kdx = kdx.intersection(set(kdx3))
        kdx = list(kdx.intersection(set(kdx4)))

        if len(kdx) > 0:

            dx = detected_sources['x'][kdx].data - cat_x
            dy = detected_sources['y'][kdx].data - cat_y
            separations = np.sqrt( dx*dx + dy*dy )

            if separations.min() <= tol:
                i = np.where(separations == separations.min())[0][0]

                # Note: stores untransformed coordinates for catalogue stars
                p = {'cat1_index': kdx[i],
                     'cat1_ra': detected_sources['ra'][kdx[i]],
                     'cat1_dec': detected_sources['dec'][kdx[i]],
                     'cat1_x': detected_sources['x'][kdx[i]],
                     'cat1_y': detected_sources['y'][kdx[i]],
                     'cat2_index': j,
                     'cat2_ra': catalog_sources['ra'][j],
                     'cat2_dec': catalog_sources['dec'][j], \
                     'cat2_x': catalog_sources['x'][j],
                     'cat2_y': catalog_sources['y'][j], \
                     'separation': separations.min()}

                matched_stars.add_match(p)

                if verbose:
                    log.info(matched_stars.summarize_last(units='pixels'))

                if j%jincr == 0:
                    percentage = round((float(j)/float(len(catalog_sources)))*100.0,0)
                    log.info(' -> Completed cross-match of '+str(percentage)+\
                                '% ('+str(j)+' of catalog stars out of '+\
                                str(len(catalog_sources))+')')

    log.info(' -> Matched '+str(matched_stars.n_match)+' stars')

    log.info('Completed star match in pixel coordinates')

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

def update_wcs(image_wcs,transform,pixscale,log,transform_type='pixels'):
    """Function to update the WCS object for an image"""

    if transform_type == 'sky':
        dx = (transform[0]*3600.0) / pixscale
        dy = (transform[1]*3600.0) / pixscale
    else:
        if image_wcs.wcs.cd[0][0] > 0.0:
            dx = -transform[0]
            dy = -transform[1]
            log.info('Applying pre-2019 image orientation')
        else:
            log.info('Applying post-2019 image orientation')
            dx = transform[0]
            dy = transform[1]

    image_wcs.wcs.crpix[0] += dx
    image_wcs.wcs.crpix[1] += dy

    log.info('Updated image WCS information')
    log.info(image_wcs)

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

def plot_overlaid_sources(output_dir,detected_sources_world,gaia_sources_world,
                          interactive=False):

    #matplotlib.use('TkAgg')

    fig = plt.figure(1,(20,10))

    plt.subplot(211)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.95,
                wspace=0.5, hspace=0.3)

    plt.plot(detected_sources_world['ra'], detected_sources_world['dec'], 'r+',
             alpha=0.5,label='Detected sources')
    plt.plot(gaia_sources_world['ra'], gaia_sources_world['dec'], 'bo',
             fillstyle='none', label='Gaia sources')

    plt.xlabel('RA deg')
    plt.ylabel('Dec deg')

    plt.legend()
    plt.grid()

    plt.subplot(212)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.95,
                wspace=0.5, hspace=0.3)

    plt.plot(detected_sources_world['x'], detected_sources_world['y'], 'r+',
             alpha=0.5,label='Detected sources')
    plt.plot(gaia_sources_world['x'], gaia_sources_world['y'], 'bo',
             fillstyle='none', label='Gaia sources')

    plt.xlabel('X pixels')
    plt.ylabel('Y pixels')

    plt.legend()
    plt.grid()

    if interactive:
        plt.show()

    else:
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

def build_detect_source_catalog(detected_objects):
    """Function to build an astroph Table with all the columns necessary"""

    detected_data = [ table.Column(name='index', data=detected_objects['index']),
                      table.Column(name='x', data=detected_objects['x']),
                      table.Column(name='y', data=detected_objects['y']),
                      table.Column(name='ra', data=np.zeros(len(detected_objects))),
                      table.Column(name='dec', data=np.zeros(len(detected_objects))),
                      table.Column(name='ref_flux', data=detected_objects['ref_flux']),
                      table.Column(name='ref_flux_err', data=np.zeros(len(detected_objects))),
                      table.Column(name='ref_mag', data=np.zeros(len(detected_objects))),
                      table.Column(name='ref_mag_err', data=np.zeros(len(detected_objects))) ]

    detected_sources = table.Table(data=detected_data)

    return detected_sources

def build_ref_source_catalog(detected_sources,gaia_sources,vphas_sources,\
                            matched_stars_gaia,matched_stars_vphas,image_wcs):
    """Function to calculate the positions of all objects detected in the
    reference image in world coordinates, combining this catalogue with
    information from the Gaia and VPHAS+ Catalogues where available.

    Output catalog is in numpy array format with columns:
    0   1  2  3   4     5        6             7         8            9                 10              11                  12
    idx x  y  ra  dec  ref_flux  ref_flux_err ref_mag ref_mag_err   cal_ref_mag   cal_ref_mag_error  cal_ref_flux       cal_ref_flux_error
    13
    gaia_source_id
    14      15           16       17            18              19
    ra    ra_error      dec     dec_error   phot_g_mean_flux phot_g_mean_flux_error
    20                      21
    phot_bp_mean_flux phot_bp_mean_flux_error
    22                  23
    phot_rp_mean_flux phot_rp_mean_flux_error
    24                  25              26      27      28
    vphas_source_id     vphas_ra    vphas_dec  gmag    gmag_error
    29          30          31          32          33
    rmag     rmag_error     imag     imag_error     clean
    34
    psf_star

    Catalogue positions and magnitudes and their errors are added if a given
    star has been matched with the Gaia and VPHAS catalogues.  Similarly, instrumental
    magnitudes and magnitude errors are given where available.
    If no entry is available for these quantities, -99.999 values are given,
    so that the catalog can be output to a valid FITS table.
    """

    def validate_entry(cat_entry):
        """Function to intercept the NaNs used by Vizier to fill in data
        missing from the catalog and replace them with the FITS-compatible
        null default used in the pipeline"""

        if str(cat_entry) == '--' or np.isnan(cat_entry) == True:
            cat_entry = -9999.999

        return cat_entry

    def validate_entry_string(cat_entry):
        """Function to intercept the NaNs used by Vizier to fill in data
        missing from the catalog and replace them with the FITS-compatible
        null default used in the pipeline"""

        if '--' in str(cat_entry):
            cat_entry = -99.999

        else:
            cat_entry = float(cat_entry)

        return cat_entry

    data = np.zeros([len(detected_sources),19])
    source_ids = np.empty([len(detected_sources),2], dtype='S30')
    gaia_source_ids = [ 'None' ] * len(detected_sources)
    vphas_source_ids = [ 'None' ] * len(detected_sources)

    for j in range(0,len(matched_stars_gaia.cat1_index),1):

        idx1 = int(matched_stars_gaia.cat1_index[j])
        idx2 = int(matched_stars_gaia.cat2_index[j])

        #source_ids[idx1,0] = gaia_sources['source_id'][idx2]
        gaia_source_ids[idx1] = gaia_sources['source_id'][idx2]

        data[idx1,0] = validate_entry(gaia_sources['ra'][idx2])
        data[idx1,1] = validate_entry(gaia_sources['ra_error'][idx2])
        data[idx1,2] = validate_entry(gaia_sources['dec'][idx2])
        data[idx1,3] = validate_entry(gaia_sources['dec_error'][idx2])
        data[idx1,4] = validate_entry(gaia_sources['phot_g_mean_flux'][idx2])
        data[idx1,5] = validate_entry(gaia_sources['phot_g_mean_flux_error'][idx2])
        data[idx1,6] = validate_entry(gaia_sources['phot_bp_mean_flux'][idx2])
        data[idx1,7] = validate_entry(gaia_sources['phot_bp_mean_flux_error'][idx2])
        data[idx1,8] = validate_entry(gaia_sources['phot_rp_mean_flux'][idx2])
        data[idx1,9] = validate_entry(gaia_sources['phot_rp_mean_flux_error'][idx2])

    for j in range(0,len(matched_stars_vphas.cat1_index),1):

        jdx1 = int(matched_stars_vphas.cat1_index[j])
        jdx2 = int(matched_stars_vphas.cat2_index[j])

        #source_ids[jdx1,1] = vphas_sources['source_id'][jdx2]
        vphas_source_ids[jdx1] = vphas_sources['source_id'][jdx2]

        data[jdx1,10] = validate_entry(vphas_sources['ra'][jdx2])
        data[jdx1,11] = validate_entry(vphas_sources['dec'][jdx2])
        data[jdx1,12] = validate_entry(vphas_sources['gmag'][jdx2])
        data[jdx1,13] = validate_entry(vphas_sources['gmag_error'][jdx2])
        data[jdx1,14] = validate_entry(vphas_sources['rmag'][jdx2])
        data[jdx1,15] = validate_entry(vphas_sources['rmag_error'][jdx2])
        data[jdx1,16] = validate_entry(vphas_sources['imag'][jdx2])
        data[jdx1,17] = validate_entry(vphas_sources['imag_error'][jdx2])
        data[jdx1,18] = validate_entry(vphas_sources['clean'][jdx2])


    table_data = [ table.Column(name='index', data=detected_sources['index'].data),
                      table.Column(name='x', data=detected_sources['x'].data),
                      table.Column(name='y', data=detected_sources['y'].data),
                      table.Column(name='ra', data=detected_sources['ra'].data),
                      table.Column(name='dec', data=detected_sources['dec'].data),
                      table.Column(name='ref_flux', data=detected_sources['ref_flux'].data),
                      table.Column(name='ref_flux_error', data=detected_sources['ref_flux_err'].data),
                      table.Column(name='ref_mag', data=detected_sources['ref_mag'].data),
                      table.Column(name='ref_mag_error', data=detected_sources['ref_mag_err'].data),
                      table.Column(name='cal_ref_mag', data=np.zeros(len(detected_sources))),
                      table.Column(name='cal_ref_mag_error', data=np.zeros(len(detected_sources))),
                      table.Column(name='cal_ref_flux', data=np.zeros(len(detected_sources))),
                      table.Column(name='cal_ref_flux_error', data=np.zeros(len(detected_sources))),
                      table.Column(name='sky_background', data=np.zeros(len(detected_sources))),
                      table.Column(name='sky_background_error', data=np.zeros(len(detected_sources))),
                      table.Column(name='gaia_source_id', data=np.array(gaia_source_ids)),
                      table.Column(name='gaia_ra', data=data[:,0]),
                      table.Column(name='gaia_ra_error', data=data[:,1]),
                      table.Column(name='gaia_dec', data=data[:,2]),
                      table.Column(name='gaia_dec_error', data=data[:,3]),
                      table.Column(name='phot_g_mean_flux', data=data[:,4]),
                      table.Column(name='phot_g_mean_flux_error', data=data[:,5]),
                      table.Column(name='phot_bp_mean_flux', data=data[:,6]),
                      table.Column(name='phot_bp_mean_flux_error', data=data[:,7]),
                      table.Column(name='phot_rp_mean_flux', data=data[:,8]),
                      table.Column(name='phot_rp_mean_flux_error', data=data[:,9]),
                      table.Column(name='vphas_source_id', data=np.array(vphas_source_ids)),
                      table.Column(name='vphas_ra', data=data[:,10]),
                      table.Column(name='vphas_dec', data=data[:,11]),
                      table.Column(name='gmag', data=data[:,12]),
                      table.Column(name='gmag_error', data=data[:,13]),
                      table.Column(name='rmag', data=data[:,14]),
                      table.Column(name='rmag_error', data=data[:,15]),
                      table.Column(name='imag', data=data[:,16]),
                      table.Column(name='imag_error', data=data[:,17]),
                      table.Column(name='clean', data=data[:,18]),
                      table.Column(name='psf_star', data=np.zeros(len(detected_sources))),
                    ]

    combined_table = table.Table(data=table_data)

    return combined_table
