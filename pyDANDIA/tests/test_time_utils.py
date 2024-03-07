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
from astropy.time import Time, TimeDelta

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
    tel_code = 'ogg-clma-2m0a'


    hjd = time_utils.calc_hjd(dateobs, RA, Dec, tel_code, exp_time)

    print('Calculated HJD=' + str(hjd) + ' reference value=' + str(output1))

    assert hjd == output1


def calc_hjd_slalib(dateobs, RA, Dec, exptime, debug=False):
    """Function to calculate the Heliocentric Julian Date from the parameters
    in a typical image header, using SLAlib routines as an independent check
    on Astropy.

    :params string dateobs: DATE-OBS, Exposure start time in UTC,
                            %Y-%m-%dT%H:%M:%S format
    :params float exptime:  Exposure time in seconds
    :params string RA:      RA in sexigesimal hours format, mean J2000.0
    :params string Dec:     Dec in sexigesimal degrees format, mean J2000.0

    Returns:

    :params float HJD:      HJD
    """

    # Convert RA, Dec (mean position) to radians:
    dRA = sexig2dec(RA)
    dRA = dRA * 15.0 * math.pi / 180.0
    dDec = sexig2dec(Dec)
    dDec = dDec * math.pi / 180.0
    if debug:
        print('RA ' + RA + ' -> decimal radians ' + str(dRA))
        print('Dec ' + Dec + ' -> decimal radians ' + str(dDec))

    # Convert the timestamp into a DateTime object:
    if 'T' in dateobs:
        try:
            dt = datetime.strptime(dateobs, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            dt = datetime.strptime(dateobs, "%Y-%m-%dT%H:%M:%S")
    else:
        try:
            dt = datetime.strptime(dateobs, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            dt = datetime.strptime(dateobs, "%Y-%m-%d %H:%M:%S")

    # Convert the exposure time into a TimeDelta object and add half of it
    # to the time to get the exposure mid-point:
    expt = timedelta(seconds=exptime)

    dt = dt + expt / 2.0

    if debug:
        print('DATE-OBS = ' + str(dateobs))
        print('Exposure time = ' + str(expt))
        print('Mid-point of exposure = ' + dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))

        at = Time(dateobs, format='isot', scale='utc')
        aexpt = TimeDelta(exptime, format='sec')

        adt = at + aexpt / 2.0
        print('Astropy: mid-point of exposure = ' + adt.value)

    # Calculate the MJD (UTC) timestamp:
    mjd_utc = datetime2mjd_utc(dt)
    if debug:
        print('MJD_UTC = ' + str(mjd_utc))
        print('Astropy MJD_UTC = ' + str(adt.mjd))

    # Correct the MJD to TT:
    mjd_tt = mjd_utc2mjd_tt(mjd_utc)
    if debug:
        print('MJD_TT = ' + str(mjd_tt))

        att = adt.tt
        print('Astropy MJD_TT = ' + str(att.mjd))

    # Calculating MJD of 1st January that year: sla_clyd XXXX convert to TT?
    (mjd_jan1, iexec) = S.sla_cldj(dt.year, 1, 1)

    if debug:
        print('MJD of Jan 1, ' + str(dt.year) + ' = ' + str(mjd_jan1))

        at_jan1 = Time(str(dt.year) + '-01-01T00:00:00.0', format='isot', scale='utc')
        print('Astropy MJD of Jan 1, ' + str(dt.year) + ' = ' + str(at_jan1.mjd))

    # Calculating the MJD difference between the DateObs and Jan 1 of the same year:
    tdiff = mjd_tt - mjd_jan1
    if debug:
        print('Time difference from Jan 1 - dateobs, ' + \
              str(dt.year) + ' = ' + str(tdiff))

        atdiff = att.mjd - at_jan1.mjd
        print('Astropy time difference = ' + str(atdiff))

    # Calculating the RV and time corrections to the Sun: XX Year could change
    print(dRA, dDec, dt.year, int(tdiff), (tdiff - int(tdiff)))
    (rv, tcorr) = S.sla_ecor(dRA, dDec, dt.year, int(tdiff), (tdiff - int(tdiff)))
    if debug:
        print('Time correction to the Sun = ' + str(tcorr))

    # Calculating the HJD:
    hjd = mjd_tt + tcorr / 86400.0 + 2400000.5
    if debug:
        print('HJD = ' + str(hjd))

    iy, im, id, fd, stat = S.sla_djcl(mjd_tt)
    print(iy, im, id, fd, stat)

def sexig2dec(coord):
    """Function to convert a sexigesimal coordinate string into a decimal float,
    returning a value in the same units as the string passed to it."""

    # Ensure that the string is separated by ':':
    coord = coord.lstrip().rstrip().replace(' ',':')

    # Strip the sign, if any, from the first character, making a note if its negative:
    if coord[0:1] == '-':
      	sign = -1.0
      	coord = coord[1:]
    else:
      	sign = 1.0

    # Split the CoordStr on the ':':
    coord_list = coord.split(':')

    # Assuming this presents us with a 3-item list, the last two items of which will
    # be handled as minutes and seconds:
    try:
        decimal = float(coord_list[0]) + (float(coord_list[1])/60.0) + \
                            (float(coord_list[2])/3600.0)
        decimal = sign*decimal

    except:
        decimal = 0

    # Return with the decimal float:
    return decimal

def datetime2mjd_utc(d):
    """Function to calculate the Modified Julian Date in UTC based on a
    datetime object, d.  This should be in ICRS and UTC.
    Output: MJD_UTC as a float
    """

    # Converts Gregorian calendar date to Modified Julian Date
    (mjd, status) = S.sla_cldj(d.year, d.month, d.day)
    if status != 0:
        return None

    # Converts a datetime from hours, minutes, seconds to a fractional day in double precision
    (fday, status ) = S.sla_dtf2d(d.hour, d.minute, d.second+(d.microsecond/1e6))
    if status != 0:
        return None

    mjd_utc = mjd + fday

    return mjd_utc


def mjd_utc2mjd_tt(mjd_utc, dbg=False):
    '''Converts a MJD in UTC (MJD_UTC) to a MJD in TT (Terrestial Time) which is
    needed for any position/ephemeris-based calculations.
    UTC->TT consists of: UTC->TAI = 10s offset + 24 leapseconds (last one 2009 Jan 1.)
    	    	    	 TAI->TT  = 32.184s fixed offset'''
# UTC->TT offset
    tt_utc = S.sla_dtt(mjd_utc)
    if dbg:
        print('TT-UTC(s)='+str(tt_utc))

# Correct MJD to MJD(TT)
    mjd_tt = mjd_utc + (tt_utc/86400.0)
    if dbg:
        print('MJD(TT)  =  '+str(mjd_tt))

    return mjd_tt

def datetime2mjd_tdb(date, obsvr_long, obsvr_lat, obsvr_hgt, dbg=False):

    auinkm = 149597870.691
# Compute MJD_UTC from passed datetime
    mjd_utc = datetime2mjd_utc(date)
    if mjd_utc == None: return None

# Compute MJD_TT
    mjd_tt = mjd_utc2mjd_tt(mjd_utc, dbg)

# Compute TT->TDB

# Convert geodetic position to geocentric distance from spin axis (r) and from
# equatorial plane (z)
    (r, z) = S.sla_geoc(obsvr_lat, obsvr_hgt)

    ut1 = compute_ut1(mjd_utc, dbg)
    if dbg:
        print("UT1="+str(ut1))

# Compute relativistic clock correction TDB->TT
    tdb_tt = S.sla_rcc(mjd_tt, ut1, -obsvr_long, r*auinkm, z*auinkm)
    if dbg: print("(TDB-TT)="+str(tdb_tt))
    if dbg: print("(CT-UT)="+str(S.sla_dtt(mjd_utc)+tdb_tt))

    mjd_tdb = mjd_tt + (tdb_tt/86400.0)

    return mjd_tdb


if __name__ == '__main__':
    
    test_calc_hjd()
    