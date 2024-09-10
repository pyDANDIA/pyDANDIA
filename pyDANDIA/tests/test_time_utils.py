# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:46:21 2019

@author: rstreet
"""
import os
import sys

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, '../'))
import time_utils
import slalib_time_utils
from astropy import units as u
import numpy as np

def test_calc_hjd():
    """Test of time calculate for conversion from UTC -> HJD, which
    is the standard in the microlensing field.

    The main pipeline's time_utils.calc_hjd function is based on the Astropy
    (v6.0.0) recipe found here:
    https://docs.astropy.org/en/stable/time/

    This unittest therefore needs an independent reference with which to compare
    the output of the pipeline's function.

    Several potential references were considered:
    Jason Eastman's online calculators will convert from UTC->BJD
    and HJD->BJD, but not UTC->HJD
    https://astroutils.astronomy.osu.edu/time/
    The software itself is installable and will calculate HJD,
    but it is IDL based and less easily wrapped
    as a Python unittest.

    HEASARC has a time conversion tool (based on the Eastman code)
    but it doesn't output HJDs directly either
    https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl

    There are also HJD calculators at Texas A&M:
    https://doncarona.tamu.edu/apps/jd/
    and the British Astronomical Society calculator at
    https://britastro.org/computing/applets_dt.html
    Tests indicate that the output of these codes agree with each other
    but not with Astropy, possibly because they don't accept the location of
    the observatory.  The underlying assumptions they are applying are not clear.

    For these reasons, the pySLAlib library was adopted as a well established and
    documented package for calculating time and coordinate systems, which is
    accessible through its Python wrapped library.
    """

    # Test case definition
    dateobs = '2024-03-06T17:48:06.492'
    RA = '05:55:10.305'
    Dec = '+07:24:25.430'
    exp_time = 0.0
    #tel_code = 'ogg-clma-2m0a'
    tel_code = 'geocenter'

    print('\nUsing Astropy routines:')
    hjd, ltt_helio = time_utils.calc_hjd(
        dateobs,
        RA, Dec,
        tel_code,
        exp_time,
        debug=True
    )
    print('Calculated HJD=' + str(hjd) + ' helio time correction=' + str(ltt_helio.to(u.s)))

    print('\nUsing SLAlib routines:')

    shjd, stcorr = slalib_time_utils.calc_hjd_slalib(
        dateobs,
        RA, Dec,
        exp_time,
        debug=True
    )
    print('SLAlib HJD=' + str(shjd) + ' helio time correction=' + str(stcorr))

    # Calculate the difference:
    dhjd = (hjd - shjd) * 24.0 * 60.0 * 60.0
    print('Difference between Astropy and SLAlib calculated timestamps: ' + str(dhjd) + 's')

    # The stated precision of the ECOR SLAlib routine used for checking here
    # is "~0.02"
    np.testing.assert_almost_equal(hjd, shjd, 3.5e-7)


if __name__ == '__main__':
    test_calc_hjd()