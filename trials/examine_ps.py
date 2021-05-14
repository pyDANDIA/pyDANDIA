from os import path
from sys import argv
from pyDANDIA import hd5_utils
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
import numpy as np
import matplotlib.pyplot as plt

def examine_ps_qc(params):
    setup = pipeline_setup.PipelineSetup()
    setup.red_dir = params['red_dir']

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(params['red_dir'], 'pyDANDIA_metadata.fits')
    exptimes = reduction_metadata.headers_summary[1]['EXPKEY']
    exptimes = np.array(exptimes, dtype='float')

    dset = hd5_utils.read_phot_hd5(setup)
    photometry = np.array(dset[:])
    ps_data = photometry[:,:,19]

    mask = np.empty(ps_data.shape)
    mask.fill(False)
    invalid = np.where(ps_data == 0.0)
    mask[invalid] = True
    ps_data = np.ma.masked_array(ps_data, mask=mask)

    frames_list = reduction_metadata.headers_summary[1]['IMAGES']
    image_index = np.arange(0,len(frames_list),1)
    qc_ps = ps_data.mean(axis=0)/exptimes

    fig = plt.figure(1,(10,10))
    plt.plot(image_index, qc_ps, 'k.')
    plt.xlabel('Image')
    plt.ylabel('p-scale/exptime [s^-1]')
    plt.grid()
    plt.xticks(image_index, frames_list, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path.join(setup.red_dir,'pscale_qc.png'))

if __name__ == '__main__':
    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
    else:
        params['red_dir'] = argv[1]

    examine_ps_qc(params)
