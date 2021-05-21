from os import path
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob

def examine_weather_data(params):

    image_list = glob.glob(path.join(params['red_dir'],'data','*.fits'))
    image_index = np.arange(0,len(image_list),1)
    frames_list = []
    for image in image_list:
        frames_list.append(path.basename(image))

    weather_data = []
    for image in image_list:
        hdr = fits.getheader(image)
        weather_data.append([ hdr['AIRMASS'], hdr['MOONDIST'], hdr['MOONFRAC'], \
                                hdr['WMSCLOUD'], hdr['WMSSKYBR'] ])
    weather_data = np.array(weather_data)

    for i,key in [[0,'airmass'],[1,'moondist'],[2,'moonfrac'],[3,'wmscloud'],[4,'wmsskybr']]:
        fig = plt.figure(1,(10,10))
        plt.plot(image_index, weather_data[:,i], 'k.')
        plt.xlabel('Image')
        plt.ylabel(key)
        plt.grid()
        plt.xticks(image_index, frames_list, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(path.join(params['red_dir'],key+'.png'))
        plt.close(fig)

    for i,key in [[3,'wmscloud'],[4,'wmsskybr']]:
        fig = plt.figure(1,(10,10))
        for site,fmt in {'cpt':'kD','lsc':'m*','coj':'gs'}.items():
            for j in image_index:
                if site in frames_list[j]:
                    plt.plot(j, weather_data[j,i], fmt)
        plt.xlabel('Image')
        plt.ylabel(key)
        plt.grid()
        plt.xticks(image_index, frames_list, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(path.join(params['red_dir'],key+'.png'))
        plt.close(fig)

if __name__ == '__main__':
    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
    else:
        params['red_dir'] = argv[1]

    examine_weather_data(params)
