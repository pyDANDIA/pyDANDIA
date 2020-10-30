import os, glob
import h5py
import numpy as np
from astropy.stats import mad_std,sigma_clipped_stats


def append_hdf5_stats(filename,mean_arr, medi_arr, std_arr, rep_arr, len_arr, ps, ps_err, bkg, bkg_err,sites):
    file_handle = h5py.File(filename,"r")
    if 'coj' in filename:
        site = 'coj'
    if 'cpt' in filename:
        site = 'cpt'
    if 'lsc' in filename:
        site = 'lsc'
    
    phot = file_handle['dataset_photometry']             
    nstars = phot.shape[0]
    random_index_array = (np.random.random(5000)*nstars).astype(np.int)
    random_index_array.sort()


    for index in random_index_array:
        msk = (phot[index,:,13] > 0) & (np.isfinite(phot[index,:,13]))
        mean, medi, std = sigma_clipped_stats(phot[index,:,13][msk])
        if np.isfinite(medi):
            mean_arr.append(mean)
            medi_arr.append(medi)
            std_arr.append(std)
            rep_arr.append(np.median(phot[index,:,14][msk]))
            len_arr.append(len(phot[index,:,14][msk]))
            ps.append(np.median(phot[index,:,19][msk]))
            ps_err.append(np.median(phot[index,:,20][msk]))
            bkg.append(np.median(phot[index,:,21][msk]))
            bkg_err.append(np.median(phot[index,:,22][msk]))
            sites.append(site)

    file_handle.close()

mean_arr, medi_arr, std_arr, rep_arr, len_arr, ps, pserr, bkg, bkgerr, sites = [], [], [], [], [], [], [], [], [], []
flist = glob.glob('field2/*/*.hdf5')

for entry in flist:
    print(entry)

    append_hdf5_stats(entry,mean_arr, medi_arr, std_arr, rep_arr, len_arr, ps, pserr, bkg, bkgerr,sites)

np.savetxt('data_quality.dat', np.c_[np.array(mean_arr),np.array(medi_arr),np.array(std_arr),np.array(rep_arr),np.array(len_arr),np.array(ps), np.array(pserr), np.array(bkg), np.array(bkgerr), np.array(sites)], fmt="%s")


