from sys import argv
from scipy.stats import binned_statistic_2d
from pyDANDIA import metadata
from pyDANDIA import hd5_utils
from pyDANDIA import crossmatch
import matplotlib.pyplot as plt
import numpy as np
from os import path, sep
import glob
import h5py
#from scipy import signal
#Col 1: hjd
#Col 4: calibrated_mag
#Col 5: calibrated_mag_err
                         

def rms_spatial_distribution(reduction_dir, dataset_code, n_samples):
    """Extract RMS for random selection of stars"""

    field_name = path.normpath(reduction_dir).split(sep)[-1]
    xmatch_file = path.join(reduction_dir,field_name+'_field_crossmatch.fits' )

    photometry_h5files = [path.join(reduction_dir,field_name+'_quad1_photometry.hdf5'),\
                          path.join(reduction_dir,field_name+'_quad2_photometry.hdf5'),\
                          path.join(reduction_dir,field_name+'_quad3_photometry.hdf5'),\
                          path.join(reduction_dir,field_name+'_quad4_photometry.hdf5')]
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(xmatch_file)    
    ra, dec, quadrant_idx, rms = [], [], np.array([]), []

    nxbins = 16
    nybins = 16
    for quadrant in [1,2,3,4]:
        idx = np.where((xmatch.field_index['quadrant'] == quadrant) &\
                       (xmatch.field_index[dataset_code+'_index'] != 0))[0]
        nstars = len(xmatch.field_index['ra'][idx]) 
        random_index_array = (np.random.random(n_samples)*nstars).astype(np.int)
        random_index_array.sort()
        ra_select = np.array(xmatch.field_index['ra'][idx])[random_index_array]
        dec_select = np.array(xmatch.field_index['dec'][idx])[random_index_array]
        #ra = np.append(ra, xmatch.field_index['ra'][idx]) 
        #dec = np.append(dec, xmatch.field_index['dec'][idx])
        #quadrant_idx = np.append(quadrant_idx, xmatch.field_index['quadrant_id'][idx])

        quadrant_phot_index =  np.array(xmatch.field_index['quadrant_id'][idx] - 1 )[random_index_array].astype(np.int)
        photometry_h5file = photometry_h5files[quadrant - 1]
        print('Open ',photometry_h5file)
        phot_data = hd5_utils.read_phot_from_hd5_file(photometry_h5file)
        for entry_idx in range(len(quadrant_phot_index)):
            lc_idx = quadrant_phot_index[entry_idx]
            dataset_idx = np.where(xmatch.images['dataset_code'] == dataset_code)[0] 
            lc = phot_data[lc_idx,dataset_idx,:]
            good = (lc[:,0]>0) & (lc[:,2]>0.)
            if len(lc[:,3][good])>0:
                ra.append(ra_select[entry_idx])
                dec.append(dec_select[entry_idx])
                rms.append(np.std(lc[:,3]))
                
    ra = np.array(ra)
    dec = np.array(dec)
    rms = np.array(rms)
    binned_stat = binned_statistic_2d(ra, dec,
                                      rms,
                                      statistic='median',
                                      bins = [nxbins,nybins],
                                      range=[[ra.min(), ra.max()],
                                            [dec.min(), dec.max()]])
    np.save(path.join(reduction_dir,'rms_spatial.npy'),binned_stat)
    fig, ax1 = plt.subplots(1,1)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)
    im = ax1.imshow(binned_stat.statistic.T,
                    cmap = 'gist_rainbow', origin='bottom',
                    extent=(0, nxbins, 0, nybins),
                    vmin = 0.0, vmax = rms.max())

    nticks = 5
    bin_incr = float(nxbins)/float(nticks)
    xincr = (ra.max() - ra.min())/nticks
    yincr = (dec.max() - dec.min())/nticks
    xticks = []
    xlabels = []
    yticks = []
    ylabels = []
    for i in range(0,nticks,1):
        xticks.append(i*bin_incr)
        xlabels.append(str(np.round((ra.min()+i*xincr),3)))
        yticks.append(i*bin_incr)
        ylabels.append(str(np.round((dec.min()+i*yincr),3)))

    plt.xticks(xticks, xlabels, rotation=45.0)
    plt.yticks(yticks, ylabels, rotation=45.0)
    #ax1.set_xticks(binned_stat.x_edge,3)
    #ax1.set_xticklabels(np.round(binned_stat.x_edge,3))
    #ax1.set_yticklabels(np.round(binned_stat.y_edge,3))

    cb = fig.colorbar(im, ax = ax1, label = 'RMS')
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    plt.savefig(path.join(reduction_dir,'rms_spatial.png'))
    
if __name__ == '__main__':
    if len(argv) == 4:
        reduction_dir = argv[1]
        dataset_code = argv[2]
        n_samples = int(argv[3])
    else:
        reduction_dir = input('Please enter the path to the field reduction directory, e.g. /data/ROME-FIELD-06: ')
        dataset_code = input('Please enter the dataset_code, e.g. ROME-FIELD-06_lsc-doma-1m0-05-fa15_ip: ')
        n_samples = int(input('Please enter the number of stars to be used per quadrant, e.g. 500: '))

    rms_spatial_distribution(reduction_dir, dataset_code, n_samples)

