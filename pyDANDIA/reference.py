'''
Reference selector

this stage is supposed to select a single sharpest image
in agreement with the thresholds.

It is inspired by the reference frame selection in DANDIA
http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:0802.1273

It does not implement a Gaussian smoothed combinations of the
sharpest N images. In the new framework this requires a full
kernel solution, adapting consecutive images to the worst
constituent and risks systematical effects.
'''
# READ json parameters
import numpy as np


class Reference():
    def __init__(self):
        '''Class containing the pipeline configuration
        for a single dataset (filter,site,etc.)'''

        self.reference = None
        self.moon_status = None
        self.configuration = None
        self.reference_magnitude = 17.
        self.sky = header['L1MEDIAN']
        self.sky_electrons = self.sky * header['GAIN']
        self.signal_electrons = 10.0**(-0.4 * adu_per_second_sbig()) * \
            header['EXPTIME'] * header['GAIN']
        self.npix = np.pi * header['PIXSCALE'] * header['L1FWHM']**2
        self.readout_noise = header['RDNOISE']
        self.noise_variance = None
        self.twilight_status = None

    def apply_thresholds(self):
        '''
        The standard reference frame selection
        requires metadata containing measurements of
        fwhm, ellipticity, sky background, number of
        detections. Independent of pre-processing,
        Images need to be discarded as potential
        reference whenever they are contaminated by
        transient phenomena (moon, bad weather).
        In addition, assymetric PSFs and defocus
        need to be avoided. Iterates over all image
        metadata entries and selects those fit for
        reference frame selection.
        '''

    def moon_brightness(self, image):
        '''
        Based on header information, determine if image was
        taken with bright/gray or dark moon
        roughly following the ESO definitions
        https://www.eso.org/sci/observing/phase2/ObsConditions.html
        '''
        if header['MOONFRAC'] < 0.4:
            self.moon_status = 'dark'
        else:
            if header['MOONFRAC'] > 0.4 and header['MOONFRAC'] < 0.7 and header['MOONDIST'] > 90.0:
                self.moon_status = 'gray'
            else:
                self.moon_status = 'bright'

    def twilight(self):
        if header['SUNALT'] < -18.:
            self.twilight_status = True
        else:
            self.twilight_status = False

    def adu_per_second_sbig(self):
        return 10.0**(-0.374462 * self.reference_magnitude + 8.24499)

    def signal_noise_ratio_header(self):
        # READOUTNOISE IS COUNTED 2x (DIFFERENCE FRAME!)
        self.noise_variance = 2.0 * self.npix * \
            self.sky_electrons + self.npix * self.readout_noise**2
