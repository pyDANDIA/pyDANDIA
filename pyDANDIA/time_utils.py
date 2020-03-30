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
    :params string RA:      RA in sexigesimal hours format
    :params string Dec:     Dec in sexigesimal degrees format

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


    # Calculating MJD of 1st January that year:
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

    # Calculating the RV and time corrections to the Sun:
    (rv,tcorr) = S.sla_ecor(dRA,dDec,dt.year,int(tdiff),(tdiff-int(tdiff)))
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

def ut1_minus_utc(mjd_utc, dbg=False):
    '''Compute UT1-UTC (in seconds), needed for tasks that require the Earth's orientation.
    UT1-UTC can be had from IERS Bulletin A (http://maia.usno.navy.mil/ser7/ser7.dat)
    but only for a short timespan and in arrears requiring continual downloading.
    Really need to get and read ftp://maia.usno.navy.mil/ser7/finals.all
    to get accurate UT1 value. Exercise for the reader...
    Currently we fake it by asuming 0.0. This will be wrong by at most +/- 0.9s
    until they do away with leapseconds.'''

    dut = 0.0
    return dut

def compute_ut1(mjd_utc, dbg=False):
    '''Compute UT1 (as fraction of a day), needed for tasks that require the Earth's orientation.
    Currently we fake it by taking the fractional part of the day. This is good
    to +/- 0.9s until they do away with leapseconds.'''

    dut = ut1_minus_utc(mjd_utc)
    if dbg: print("DUT="+str(dut))
    ut1 = (mjd_utc - int(mjd_utc)) + (dut/86400.0)

    return ut1

def parse_neocp_date(neocp_datestr, dbg=False):
    '''Parse dates from the NEOCP (e.g. '(Nov. 16.81 UT)' ) into a datetime object and
    return this. No sanity checking of the input is done'''
    month_map = { 'Jan' : 1,
    	    	  'Feb' : 2,
		  'Mar' : 3,
		  'Apr' : 4,
		  'May' : 5,
		  'Jun' : 6,
		  'Jul' : 7,
		  'Aug' : 8,
		  'Sep' : 9,
		  'Oct' : 10,
		  'Nov' : 11,
		  'Dec' : 12 }

    chunks = neocp_datestr.split(' ')
    if dbg: print(chunks)
    if len(chunks) != 3: return None
    month_str = chunks[0].replace('(', '').replace('.', '')
    day_chunks = chunks[1].split('.')
    if dbg: print(day_chunks)
    neocp_datetime = datetime(year=datetime.utcnow().year, month=month_map[month_str[0:3]],
    	day=int(day_chunks[0]))

    decimal_day = float('0.' + day_chunks[1].split()[0])
    neocp_datetime = neocp_datetime + timedelta(days=decimal_day)

    return neocp_datetime

def round_datetime(date_to_round, round_mins=10, round_up=False):
    '''Rounds the passed datetime object, <date_to_round>, to the
    'floor' (default) or the 'ceiling' (if [roundup=True]) of
    the nearest passed amount (which defaults to 10min)'''

    correct_mins = 0
    if round_up:
        correct_mins = round_mins
    date_to_round = date_to_round - timedelta(minutes=(date_to_round.minute % round_mins)-correct_mins,
                	seconds=date_to_round.second,
                	microseconds=date_to_round.microsecond)

    return date_to_round


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
