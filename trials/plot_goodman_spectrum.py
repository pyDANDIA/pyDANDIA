from os import path
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def plot_extracted_spectrum():

    params = get_args()

    data = read_goodman_spectrum_file(params)

    plot_spectrum(data,params)

def plot_spectrum(data,params):

    plot_params = {'axes.labelsize': 18,
                  'axes.titlesize': 18,
                  'xtick.labelsize':18,
                  'ytick.labelsize':18}
    plt.rcParams.update(plot_params)

    fig = plt.figure(1,(10,10))

    plt.plot(data[:,0], data[:,1], 'k-')

    plt.xlabel('Wavelength [$\\AA$]')
    plt.ylabel('Flux')
    plt.grid()

    if params['interactive']:
        plt.show()
    else:
        plt.savefig(params['output_plot'], bbox_inches='tight')

def read_goodman_spectrum_file(params):

    if not path.isfile(params['input_file']):
        raise IOError('Cannot find input spectrum file '+params['input_file'])

    hdu = fits.open(params['input_file'])
    header = hdu[0].header

    data = np.zeros( (len(hdu[0].data),2) )

    data[:,1] = hdu[0].data

    l1 = header['CRVAL1']
    l2 = header['CRVAL1'] + header['NAXIS1']*header['CDELT1']
    try:
        data[:,0] = np.arange(l1, l2, header['CDELT1'])
    except ValueError:
        l2 = header['CRVAL1'] + (header['NAXIS1']-1)*header['CDELT1']
        data[:,0] = np.arange(l1, l2, header['CDELT1'])
    return data

def get_args():

    params = {'interactive': False}

    if len(argv) == 1:
        params['input_file'] = input('Please enter the path to the extracted 1D spectrum FITS file: ')
        params['output_plot'] = input('Please enter the path to the output plot: ')
    else:
        params['input_file'] = argv[1]
        params['output_plot'] = argv[2]

    if len(argv) > 3:
        for a in argv[2:]:
            if '--i' in a:
                params['interactive'] = True

    return params


if __name__ == '__main__':
    plot_extracted_spectrum()
