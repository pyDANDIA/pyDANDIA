from os import path
from sys import argv
from pyDANDIA import vizier_tools
from astropy.io import fits
from astropy import wcs as aWCS
from astropy import table
import numpy as np

def extract_VPHAS_catalog():

    header = { 'NAXIS1': 4096, 'NAXIS2': 4096,
               'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
               'CRPIX1': 2048.0, 'CRPIX2': 2048.0,
               'CRVAL1': 270.83666667, 'CRVAL2': -28.8431583, # ROME-20
             #  'CRVAL1': 269.5637194, 'CRVAL2': -28.4422329, # ROME-13
               'CUNIT1': 'deg', 'CUNIT2': 'deg',
               'CD1_1': 0.0001081, 'CD1_2': 0.0,
               'CD2_1': 0.0, 'CD2_2': -0.0001081,
               'PIXSCALE': 0.389 }
    image_wcs = aWCS.WCS(header)

    diagonal = np.sqrt(header['NAXIS1']*header['NAXIS1'] + header['NAXIS2']*header['NAXIS2'])
    radius = diagonal*header['PIXSCALE']/60.0/2.0 #arcminutes

    vphas_sources = vizier_tools.search_vizier_for_sources(header['CRVAL1'], header['CRVAL2'], radius, 'VPHAS+', coords='degrees')
    print(vphas_sources)

    table_data = [ table.Column(name='source_id', data=vphas_sources['sourceID'].data),
                  table.Column(name='ra', data=vphas_sources['_RAJ2000'].data),
                  table.Column(name='dec', data=vphas_sources['_DEJ2000'].data),
                  table.Column(name='gmag', data=vphas_sources['gmag'].data),
                  table.Column(name='gmag_error', data=vphas_sources['e_gmag'].data),
                  table.Column(name='rmag', data=vphas_sources['rmag'].data),
                  table.Column(name='rmag_error', data=vphas_sources['e_rmag'].data),
                  table.Column(name='imag', data=vphas_sources['imag'].data),
                  table.Column(name='imag_error', data=vphas_sources['e_imag'].data),
                  table.Column(name='clean', data=vphas_sources['clean'].data),
                  ]
    vphas_sources = table.Table(data=table_data)

    print(vphas_sources)

    world_coords = np.zeros((len(vphas_sources),2))
    world_coords[:,0] = vphas_sources['ra'].data
    world_coords[:,1] = vphas_sources['dec'].data

    pixel_coords = image_wcs.wcs_world2pix(world_coords,1)

    f = open('vphas_catalog_stars.reg','w')

    for j in range(0,len(pixel_coords),1):
        f.write('point '+str(pixel_coords[j,0])+' '+str(pixel_coords[j,1])+' # color=blue\n')

    f.close()

if __name__ == '__main__':
    extract_VPHAS_catalog()
