from os import path
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def plot_extracted_spectrum():

    params = get_args()

    data = read_goodman_spectrum_file(params)

    plot_spectrum(data,params)

    save_spectrum_data_as_text_file(data, params)

def plot_spectrum(data,params):

    plot_params = {'axes.labelsize': 18,
                  'axes.titlesize': 18,
                  'xtick.labelsize':18,
                  'ytick.labelsize':18}
    plt.rcParams.update(plot_params)

    fig = plt.figure(1,(10,10))

    plt.plot(data[:,0], data[:,1], 'k-')

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    plt.grid()

    if params['interactive']:
        plt.show()
    else:
        plt.savefig(params['output_plot'], bbox_inches='tight')

def save_spectrum_data_as_text_file(data, params):

    file_path = path.splitext(params['output_plot'])[0]+'.txt'

    f = open(file_path, 'w')
    f.write('# Wavelength [nm]   Flux   Flux uncertainty\n')
    for i in range(0,data.shape[0],1):
        err = np.sqrt(data[i,1])
        f.write(str(data[i,0])+'  '+str(data[i,1])+' '+str(err)+'\n')
    f.close()

def read_goodman_spectrum_file(params):

    if not path.isfile(params['input_file']):
        raise IOError('Cannot find input spectrum file '+params['input_file'])

    hdu = fits.open(params['input_file'])
    header = hdu[0].header

    data = np.zeros( (len(hdu[0].data),2) )

    data[:,1] = hdu[0].data

    l1 = header['CRVAL1']
    l2 = header['CRVAL1'] + header['NAXIS1']*header['CDELT1']

    # Better to do this than use arange because rounding errors in CDELT1
    # cause an unpredictable offset in the indexing
    # Note conversion from Angstroms to nm
    for i in range(0,len(data[:,0]),1):
        data[i,0] = (header['CRVAL1'] + (float(i)*header['CDELT1']))/10.0

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
