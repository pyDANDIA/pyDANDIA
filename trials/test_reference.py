# -*- coding: utf-8 -*-
"""
@author: mpgh
"""
from os import path
from os import listdir
from os import path
from astropy.io import fits
from astropy.table import Table
from operator import itemgetter
import numpy as np


class Reference:
    """Class defining the data structure produced by the pyDANDIA pipeline
    to hold metadata regarding the reduction of a single dataset, including
    reduction configuration parameters, the data inventory and key measured
    parameters from each stage.
    """

    def __init__(self):
        """Class containing the configuration for selecting the best reference
           image(s) (to be obtained from metadata)
        """
        self.target_magnitude = 17.
        self.reference = None
        self.configuration = None
        self.signal_to_noise_metric = []
        self.camera = 'Sinistro'
        self.thresholds = []
        self.noise_variance = None

    def moon_brightness_header(self, header):
        '''
        Based on header information, determine if image was
        taken with bright/gray or dark moon
        roughly following the ESO definitions
        https://www.eso.org/sci/observing/phase2/ObsConditions.html
        '''
        if float(header['MOONFRAC']) < 0.4:
            return 'dark'
        else:
            if float(header['MOONFRAC']) > 0.4 and float(header['MOONFRAC']) < 0.7 and float(header['MOONDIST']) > 90.0:
                return 'gray'
            else:
                return 'bright'

    def electrons_per_second_sinistro(self, mag):
        return 1.47371235e+09 * 10.**(-0.4 * mag)

    def check_header_thresholds(self, header):
        """Check for images that cannot be used as reference image regardless
        of their average FWHM.
        """
        print self.moon_brightness_header(header)
        criteria = [self.moon_brightness_header != 'bright',int(header['NPFWHM'])>25,float(header['ELLIP'])>0.9]

        return all(criteria)

    def add_header_rank(self, image_name, header):
        """Check for images that cannot be used as reference image regardless
        of their average FWHM.
        """

        if self.camera == 'Sinistro':
            if self.check_header_thresholds(header)==True:
                self.signal_electrons = 10.0**(-0.4 * self.electrons_per_second_sinistro(
                    self.target_magnitude)) * header['EXPTIME'] * header['GAIN']
                self.sky = header['L1MEDIAN']
                self.sky_electrons = self.sky * header['GAIN']
                # Simplified number of pixels for optimal aperture photometry
                self.npix = np.pi * \
                    header['PIXSCALE'] * (0.67 * header['L1FWHM'])**2
                self.readout_noise = header['RDNOISE']
                self.signal_to_noise_metric.append([image_name, (self.signal_electrons) / (
                    self.signal_electrons + self.npix * (self.sky_electrons + self.readout_noise**2 ))**0.5])

    def single_best(self):
        """Sort and return the best reference image
        """

        print self.signal_to_noise_metric
        best_image = sorted(self.signal_to_noise_metric, key=itemgetter(1))
        return best_image


# Command line section
if __name__ == '__main__':

    event_reference_image = Reference()
    path_to_header = '../trials/data/headers'
    dirlist = listdir(path.join(path_to_header))
    
    for entry in dirlist:

        headerlist = fits.open(path.join(path_to_header,entry))
        print entry,headerlist[1].header['NPFWHM']
        event_reference_image.add_header_rank(entry, headerlist[1].header)

    print event_reference_image.single_best()
