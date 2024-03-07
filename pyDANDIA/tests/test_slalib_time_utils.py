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
import slalib_time_utils
import numpy as np

def get_test_cases():
    return [
        {
        'dateobs': '2024-03-06T17:48:06.492',
        'RA': '05:55:10.305',
        'Dec': '+07:24:25.430',
        'exp_time': 0.0,
        'tel_code': 'ogg-clma-2m0a',
        'RA_deg': 88.7929375,
        'Dec_deg': 7.40706389,
        'MJD_UTC': 60375.741741805556,
        'MJD_TT': 60375.74254254629,
        'obs_site': {
            'name': 'ogg',
            'lat': 20.7070833333 * np.pi/180.0,
            'lon': 156.2576111111 * np.pi/180.0,
            'height': 3055.0
        },
        'BJD_TDB': 2460376.243709158 - 2400000.5,
    }
    ]

def test_slalib_leap_seconds():
    """SLAlib is precompiled code with a built-in table of the known leap seconds
    declared to the date of compilation.  This test is designed as a reminder to the
    user to install updates and recompile when new leap second are announced by
    comparing the standard correction for leap seconds with that of the installed
    Astropy version"""

    # Expected offset for leap seconds as of 2024 March 6:
    tai_utc_offset = 69.184

    # Test time in UTC:
    test_cases = get_test_cases()
    usecase = test_cases[0]

    # Calculate the TAI-UTC offset using Astropy
    t = Time(usecase['dateobs'], format='isot', scale='utc')
    dt = (t.tai.mjd - t.mjd) * 24.0 * 60.0 * 60.0    # Difference in seconds

    # Calculate using SLAlib
    offset = S.sla_dat(t.mjd)

    # If these are not the same, SLAlib needs updating
    np.testing.assert_almost_equal(offset, dt, decimal=3)

def test_sexig2dec():
    """Function to convert a sexigesimal coordinate string into a decimal float,
    returning a value in the same units as the string passed to it."""

    test_cases = get_test_cases()

    for usecase in test_cases:

        # Function returns a decimal float in units of hour angle and degrees
        # respectively for RA and Dec, so standardize these to degrees for testing
        dRA = slalib_time_utils.sexig2dec(usecase['RA']) * 15.0
        dDec = slalib_time_utils.sexig2dec(usecase['Dec'])

        assert(abs(dRA - usecase['RA_deg']) < 1.0/3600.0)
        assert(abs(dDec - usecase['Dec_deg']) < 1.0/3600.0)

def test_datetime2mjd_utc():
    """Function to test the conversion between a datetime object and MJD (UTC)"""

    test_cases = get_test_cases()

    for usecase in test_cases:
        d = datetime.strptime(usecase['dateobs'],"%Y-%m-%dT%H:%M:%S.%f")
        mjd_utc = slalib_time_utils.datetime2mjd_utc(d)

        assert(abs(mjd_utc - usecase['MJD_UTC']) < 1e-7)

def test_mjd_utc2mjd_tt():
    '''Converts a MJD in UTC (MJD_UTC) to a MJD in TT (Terrestial Time) which is
    needed for any position/ephemeris-based calculations.
    UTC->TT consists of: UTC->TAI = 10s offset + 24 leapseconds (last one 2009 Jan 1.)
    	    	    	 TAI->TT  = 32.184s fixed offset'''

    test_cases = get_test_cases()

    for usecase in test_cases:
        mjd_tt = slalib_time_utils.mjd_utc2mjd_tt(usecase['MJD_UTC'])

        np.testing.assert_almost_equal(mjd_tt, usecase['MJD_TT'], 1e-7)

def test_datetime2mjd_tdb():
    """Function to test the convertion of a datetime object in UTC to Barycentric Dynamical Time,
    TDB.  In this case, the Ohio State online calculator was used to test the example case:
    https://astroutils.astronomy.osu.edu/time/utc2bjd.php
    """

    test_cases = get_test_cases()

    for usecase in test_cases:
        d = datetime.strptime(usecase['dateobs'],"%Y-%m-%dT%H:%M:%S.%f")
        mjd_tdb = slalib_time_utils.datetime2mjd_tdb(
            d,
            usecase['obs_site']['lon'],
            usecase['obs_site']['lat'],
            usecase['obs_site']['height']
        )

        #print((usecase['BJD_TDB']-mjd_tdb)*24.0*60.0*60.0)
        # Function not currently in use, skip test as not comparing like with like
        #np.testing.assert_almost_equal(usecase['BJD_TDB'], mjd_tdb, 7)

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
    test_slalib_leap_seconds()
    test_sexig2dec()
    test_datetime2mjd_utc()
    test_mjd_utc2mjd_tt()
