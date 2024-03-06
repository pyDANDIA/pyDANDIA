# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:46:21 2019

@author: rstreet
"""
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import time_utils
from datetime import datetime,timedelta
import pyslalib.slalib as S
import math

def test_calc_hjd():
    """Test of time code against Texas A&M's online
    HJD calculator at:
    https://doncarona.tamu.edu/apps/jd/
    and the British Astronomical Society calculator at
    https://britastro.org/computing/applets_dt.html

    HEASARC has a time conversion tool but it doesn't handle HJDs
    https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl

    Jason Eastman's online calculators will convert from UTC->BJD
    and HJD->BJD, but not UTC->HJD
    https://astroutils.astronomy.osu.edu/time/
    """
    
    dateobs = '2024-03-06T17:48:06.492'
    RA = '05:55:10.305'
    Dec = '+07:24:25.430'
    exp_time = 0.0
    tel_code = 'ogg-clma-2m0a'
    output1 = 2460376.2429285203        # Texas A&M
    output2 = 2460376.24292             # British Astro Society

    hjd = time_utils.calc_hjd(dateobs, RA, Dec, tel_code, exp_time)

    print('Calculated HJD=' + str(hjd) + ' reference value=' + str(output1))

    assert hjd == output1
    
if __name__ == '__main__':
    
    test_calc_hjd()
    