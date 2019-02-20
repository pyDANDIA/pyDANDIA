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
    """Test of time code against the British Astronomical Society's online 
    HJD calculator at:
    http://britastro.org/computing/applets_dt.html
    """
    
    dateobs = '2019-02-02T12:34:00.0'
    ra = '17:59:27.05'
    dec = '-28:36:37.0'
    exptime = 60.0
    
    hjd = time_utils.calc_hjd(dateobs, ra, dec, exptime, debug=True)
    
    assert hjd == 2458517.020578858

    
if __name__ == '__main__':
    
    test_calc_hjd()
    