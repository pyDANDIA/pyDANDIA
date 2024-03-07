# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:58:24 2019

@author: rstreet
Code adapted from T. Lister's code from NEOExchange
"""

from sys import argv
from datetime import datetime,timedelta
import pyslalib.slalib as S
import math
from astropy.time import Time, TimeDelta


def calc_hjd(dateobs,RA,Dec,exptime,debug=False):
    """Function to calculate the Heliocentric Julian Date from the parameters
    in a typical image header:

    :params string dateobs: DATE-OBS, Exposure start time in UTC,
                            %Y-%m-%dT%H:%M:%S format
    :params float exptime:  Exposure time in seconds
    :params string RA:      RA in sexigesimal hours format, J2000.0
    :params string Dec:     Dec in sexigesimal degrees format, J2000.0

    Returns:

    :params float HJD:      HJD
    """

    # Convert RA, Dec to radians:
    dRA = sexig2dec(RA)
    dRA = dRA * 15.0 * math.pi / 180.0
    dDec = sexig2dec(Dec)
    dDec = dDec * math.pi / 180.0
    if debug:
        print('RA '+RA+' -> decimal radians '+str(dRA))
        print('Dec '+Dec+' -> decimal radians '+str(dDec))

    # Convert the timestamp into a DateTime object:
    if 'T' in dateobs:
        try:
            dt = datetime.strptime(dateobs,"%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            dt = datetime.strptime(dateobs,"%Y-%m-%dT%H:%M:%S")
    else:
        try:
            dt = datetime.strptime(dateobs,"%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            dt = datetime.strptime(dateobs,"%Y-%m-%d %H:%M:%S")

    # Convert the exposure time into a TimeDelta object and add half of it
    # to the time to get the exposure mid-point:
    expt = timedelta(seconds=exptime)

    dt = dt + expt/2.0

    if debug:
        print('DATE-OBS = '+str(dateobs))
        print('Exposure time = '+str(expt))
        print('Mid-point of exposure = '+dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))

        at = Time(dateobs,format='isot', scale='utc')
        aexpt = TimeDelta(exptime,format='sec')

        adt = at + aexpt/2.0
        print('Astropy: mid-point of exposure = '+adt.value)

    # Calculate the MJD (UTC) timestamp:
    mjd_utc = datetime2mjd_utc(dt)
    if debug:
        print('MJD_UTC = '+str(mjd_utc))
        print('Astropy MJD_UTC = '+str(adt.mjd))

    # Correct the MJD to TT:
    mjd_tt = mjd_utc2mjd_tt(mjd_utc)
    if debug:
        print('MJD_TT = '+str(mjd_tt))

        att = adt.tt
        print('Astropy MJD_TT = '+str(att.mjd))


    # Convert the mean RA, Dec (presumed to be J2000.0) to the geocentric
    # apparent RA, Dec, accounting for precession, nutation and abberation.
    # Inputs are:
    # RM,DM mean [α,δ] (radians)
    # PR,PD proper motions: [α,δ] changes per Julian year
    # PX parallax (arcsec)
    # RV radial velocity (km s−1, +ve if receding)
    # EQ epoch and equinox of star data (Julian)
    # DATE TDB for apparent place (JD−2400000.5)
    # Docs note that the distinction between the required TDB and TT is always negligible
    (aRA, aDec) = S.sla_map(
        dRA, dDec,
        0.0, 0.0,
        0.0, 0.0,
        2000.0, mjd_tt)
    if debug:
        print('Converted to geocentric apparent coordinates: ', aRA, aDec)

    # Calculating MJD of 1st January that year: sla_clyd XXXX convert to TT?
    (mjd_jan1,iexec) = S.sla_cldj(dt.year,1,1)
    if debug:
        print('MJD of Jan 1, '+str(dt.year)+' = '+str(mjd_jan1))

        at_jan1 = Time(str(dt.year)+'-01-01T00:00:00.0',format='isot', scale='utc')
        print('Astropy MJD of Jan 1, '+str(dt.year)+' = '+str(at_jan1.mjd))

    # Calculating the MJD difference between the DateObs and Jan 1 of the same year:
    tdiff = mjd_tt - mjd_jan1
    if debug:
        print('Time difference from Jan 1 - dateobs, '+\
                str(dt.year)+' = '+str(tdiff))

        atdiff = att.mjd - at_jan1.mjd
        print('Astropy time difference = '+str(atdiff))

    # Calculating the RV and time corrections to the Sun: XX Year could change
    # This function requires the following input (from SLAlib documentation):
    # RM,DM = mean [α,δ] of date (radians)
    # IY = year
    # ID = day in year (1 = Jan 1st)
    # FD = fraction of day
    # where the date and time is TDB (loosely ET) in a Julian calendar which has
    # been aligned to the ordinary Gregorian calendar for the interval 1900 March 1 to
    # 2100 February 28. The year and day can be obtained by calling sla_CALYD or sla_CLYD.
    (iy, im, id, fd, stat) = S.sla_djcl(mjd_tt)
    (ny, nd, stat) = S.sla_clyd(iy, im, id)
    print(ny, nd)
    (rv,tcorr) = S.sla_ecor(
        aRA,
        aDec,
        ny, nd, fd
    )
    if debug:
        print('Time correction to the Sun = '+str(tcorr))

    # Calculating the HJD:
    hjd = mjd_tt + tcorr/86400.0 + 2400000.5
    if debug:
        print('HJD = '+str(hjd))

    return hjd

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

# Compute MJD for UTC
    (mjd, status) = S.sla_cldj(d.year, d.month, d.day)
    if status != 0:
        return None
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

if __name__ == '__main__':

    if len(argv) == 1:
        print('Call sequence: python calctime.py DateObsStr RA Dec exptime[s]')
        exit()
    else:
        dateobs = argv[1]
        RA = argv[2]
        Dec = argv[3]
        exptime = float(argv[4])

    hjd = calc_hjd(dateobs,RA,Dec,exptime,debug=True)
