# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 08:10:51 2019

@author: rstreet
"""
from os import path
from sys import argv
from pyDANDIA import metadata
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy import table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

def map_astrometry_residuals(params,star_catalog=None):

    if star_catalog == None:
        star_catalog = load_meta_data(params)

    idx1 = np.where(star_catalog['ra_cat'] > 0.0)[0]
    idx2 = np.where(star_catalog['dec_cat'] != 0.0)[0]
    idx = list(set(idx1).intersection(set(idx2)))

    dra = star_catalog['ra_det'][idx] - star_catalog['ra_cat'][idx]
    ddec = star_catalog['dec_det'][idx] - star_catalog['dec_cat'][idx]

    (avgdra,mad_dra) = calc_median_mad(dra)
    (avgddec,mad_ddec) = calc_median_mad(ddec)

    print('Deviations over all cross-matched stars:')
    print('Median, MAD dRA = '+str(avgdra)+' '+str(mad_dra)+' deg')
    print('Median, MAD dRA = '+str(avgdra*3600.0)+' '+str(mad_dra*3600.0)+' arcsec')
    print('Median, MAD dRA = '+str(avgdra*3600.0/params['pixscale'])+' '+\
                                str(mad_dra*3600.0/params['pixscale'])+' pix\n')

    print('Median, MAD dDec = '+str(avgddec)+' '+str(mad_ddec)+' deg')
    print('Median, MAD dDec = '+str(avgddec*3600.0)+' '+str(mad_ddec*3600.0)+' arcsec')
    print('Median, MAD dDec = '+str(avgddec*3600.0/params['pixscale'])+' '+\
                                 str(mad_ddec*3600.0/params['pixscale'])+' pix')

    coords = np.zeros((len(idx),2))
    coords[:,0] = star_catalog['ra_cat'][idx].data
    coords[:,1] = star_catalog['dec_cat'][idx].data

    catalog = SkyCoord(coords, frame='icrs', unit=(u.deg, u.deg))

    target = SkyCoord(params['target_ra']+' '+params['target_dec'],
                       frame='icrs',unit=(u.hourangle, u.deg))

    sep = target.separation(catalog)

    tol = 2.0
    jdx = np.where(sep < tol*u.arcmin)[0]

    (avgdra,mad_dra) = calc_median_mad(dra[jdx])
    (avgddec,mad_ddec) = calc_median_mad(ddec[jdx])

    print('\nDeviations within '+str(round(tol,2))+' arcmin of target')
    print('Median, MAD dRA = '+str(avgdra)+' '+str(mad_dra)+' deg')
    print('Median, MAD dRA = '+str(avgdra*3600.0)+' '+str(mad_dra*3600.0)+' arcsec')
    print('Median, MAD dRA = '+str(avgdra*3600.0/params['pixscale'])+' '+\
                                str(mad_dra*3600.0/params['pixscale'])+' pix\n')

    print('Median, MAD dDec = '+str(avgddec)+' '+str(mad_ddec)+' deg')
    print('Median, MAD dDec = '+str(avgddec*3600.0)+' '+str(mad_ddec*3600.0)+' arcsec')
    print('Median, MAD dDec = '+str(avgddec*3600.0/params['pixscale'])+' '+\
                                 str(mad_ddec*3600.0/params['pixscale'])+' pix')

    fig = plt.figure(1,(10,10))

    plt.subplot(221)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.8, top=0.95,
                wspace=0.4, hspace=0.2)

    plt.plot(star_catalog['x'][idx],dra*3600.0,'k.',markersize=1)
    plt.plot(star_catalog['x'][idx][jdx],dra[jdx]*3600.0,'r.',markersize=1)
    plt.xlabel('X pixel')
    plt.ylabel('$\\Delta$ RA [arcsec]')

    plt.subplot(222)
    plt.plot(star_catalog['y'][idx],dra*3600.0,'k.',markersize=1)
    plt.plot(star_catalog['y'][idx][jdx],dra[jdx]*3600.0,'r.',markersize=1)
    plt.xlabel('Y pixel')
    plt.ylabel('$\\Delta$ RA [arcsec]')

    plt.subplot(223)
    plt.plot(star_catalog['x'][idx],ddec*3600.0,'k.',markersize=1)
    plt.plot(star_catalog['x'][idx][jdx],ddec[jdx]*3600.0,'r.',markersize=1)
    plt.xlabel('X pixel')
    plt.ylabel('$\\Delta$ Dec [arcsec]')

    plt.subplot(224)
    plt.plot(star_catalog['y'][idx],ddec*3600.0,'k.',markersize=1)
    plt.plot(star_catalog['y'][idx][jdx],ddec[jdx]*3600.0,'r.',markersize=1)
    plt.xlabel('Y pixel')
    plt.ylabel('$\\Delta$ Dec [arcsec]')

    plt.savefig(path.join(params['red_dir'],'astrometry_residuals.pdf'),
                bbox_inches='tight')

    plt.close(1)

    sep_image = np.zeros((params['n_pix_x'],params['n_pix_y']))
    norm = np.zeros((params['n_pix_x'],params['n_pix_y']))

    for j in jdx:

        x = int(round(star_catalog['x'][idx][j],0))
        y = int(round(star_catalog['y'][idx][j],0))

        sep_image[y,x] += sep[j].value
        norm[y,x] += 1.0

    kdx = np.where(norm > 0.0)
    sep_image[kdx] = sep_image[kdx]/norm[kdx]

    fig = plt.figure(2,(10,10))

    plt.imshow(sep_image)

    xmin = (star_catalog['x'][idx][jdx]).min()
    xmax = (star_catalog['x'][idx][jdx]).max()
    ymin = (star_catalog['y'][idx][jdx]).min()
    ymax = (star_catalog['y'][idx][jdx]).max()

    plt.axis([xmin,xmax,ymin,ymax])

    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')

    plt.colorbar()

    plt.savefig(path.join(params['red_dir'],'astrometry_residual_separations.pdf'),
                bbox_inches='tight')
    plt.close(2)

def calc_median_mad(a):

    med = np.median(a)

    mad = np.median(a - med)

    return med, mad

def load_meta_data(params):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( params['red_dir'],
                                               params['meta_file'],
                                              'star_catalog' )
    reduction_metadata.load_a_layer_from_file( params['red_dir'],
                                               params['meta_file'],
                                              'phot_calib' )
    reduction_metadata.load_a_layer_from_file( params['red_dir'],
                                               params['meta_file'],
                                              'reduction_parameters' )

    star_catalog = table.Table()
    star_catalog['star_index'] = reduction_metadata.star_catalog[1]['star_index']
    star_catalog['x'] = reduction_metadata.star_catalog[1]['x_pixel']
    star_catalog['y'] = reduction_metadata.star_catalog[1]['y_pixel']
    star_catalog['ra_det'] = reduction_metadata.star_catalog[1]['RA_J2000']
    star_catalog['dec_det'] = reduction_metadata.star_catalog[1]['DEC_J2000']
    star_catalog['ra_cat'] = reduction_metadata.phot_calib[1]['_RAJ2000']
    star_catalog['dec_cat'] = reduction_metadata.phot_calib[1]['_DEJ2000']

    params['pixscale'] = reduction_metadata.reduction_parameters[1]['PIX_SCALE'][0]
    params['n_pix_x'] = reduction_metadata.reduction_parameters[1]['IMAGEX2'][0]
    params['n_pix_y'] = reduction_metadata.reduction_parameters[1]['IMAGEY2'][0]

    return star_catalog

def get_args():

    params = {}

    if len(argv) > 1:

        params['red_dir'] = argv[1]
        params['target_ra'] = argv[2]
        params['target_dec'] = argv[3]

    else:

        params['red_dir'] = input('Please enter the path to the metadata file: ')
        params['target_ra'] = input('Please enter the target RA [sexigesimal]: ')
        params['target_dec'] = input('Please enter the target Dec[sexigesimal]: ')

    params['meta_file'] = 'pyDANDIA_metadata.fits'

    return params


if __name__ == '__main__':

    params = get_args()

    map_astrometry_residuals(params)
