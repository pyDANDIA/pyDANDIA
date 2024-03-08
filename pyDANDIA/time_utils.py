# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:58:24 2019

@author: rstreet
"""

import argparse
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u
from astropy import constants

def calc_hjd(dateobs, RA, Dec, tel_code, exp_time, debug=False):
    """Function to calculate the Heliocentric Julian Date of the center of a image,
    for the mid-point of an exposure

    Parameters:
        dateobs  str  Datetime in UTC of the exposure start, in ISOT format %Y-%m-%dT%H:%M:%S
        RA       str  RA of the target or frame center in sexigesimal format, J2000.0
        Dec      str  Dec of the target or frame center in sexigesimal format, J2000.0
        tel_code str  Identifer code of the observatory site where the data were taken;
                      used with a look-up table to find the geographic location
        exp_time  float Exposure time in seconds

    Returns:
        hjd      float Calculated HJD

    This function is based on a recipe from Astropy's documentation:
    https://docs.astropy.org/en/stable/time/
    """

    # Get the geographic location of the telescope
    observatory_site = fetch_observatory_location(tel_code)

    # Convert the timestamp string to a time object with this location, then
    # add half the exposure time to arrive at the mid-point of the exposure
    t = Time(dateobs, format='isot', scale='utc', location=observatory_site)
    dt = TimeDelta(exp_time/2.0, format='sec')
    t = t + dt
    if debug:
        print('Calculating HJD for ' + tel_code)
        print('DATE-OBS = ' + str(dateobs))
        print('Exposure time = ' + str(exp_time))
        print('Mid-point of exposure (UTC) = ' + str(t.mjd))
        print('Mid-point of exposure (TT) = ' + str(t.tt.mjd))

    # Produce a SkyCoord object for the target coordinates, frame ICRS
    star = SkyCoord(RA, Dec, unit=(u.hourangle, u.degree), frame='icrs')
    if debug: print('Star position: ' + repr(star))

    # Calculate the light travel time from the observatory to the Sun heliocenter
    ltt_helio = t.light_travel_time(star, 'heliocentric', ephemeris='jpl')
    if debug:
        print('Light travel time correction: ' + str(ltt_helio.to(u.s)))

    # Apply the correction to arrive at the HJD
    hjd = t.utc + ltt_helio
    if debug:
        print('HJD (astropy) = ' + str(hjd.jd))

    # Calculate radial velocity, helicentric correction
    heliocorr = star.radial_velocity_correction(
        'heliocentric',
        obstime=t,
    )
    if debug: print('Heliocentric RV correction: ' + str(heliocorr.to(u.km / u.s)))

    return hjd.jd, ltt_helio

def fetch_observatory_location(tel_code):
    lco_facilities = {
        'ogg-clma-2m0a': EarthLocation(lat='20d42m25.5sN', lon='156d15m27.4sW', height=3055.0 * u.m),
        'ogg-clma-0m4b': EarthLocation(lat='20d42m25.1sN', lon='156d15m27.11sW', height=3037.0 * u.m),
        'ogg-clma-0m4c': EarthLocation(lat='20d42m25.1sN', lon='156d15m27.12sW', height=3037.0 * u.m),
        'coj-clma-2m0a': EarthLocation(lat='31d16m23.4sS', lon='149d4m13.0sE', height=1111.8 * u.m),
        'coj-doma-1m0a': EarthLocation(lat='31d16m22.56sS', lon='149d4m14.33sE', height=1168.0 * u.m),
        'coj-domb-1m0a': EarthLocation(lat='31d16m22.89sS', lon='149d4m14.75sE', height=1168.0 * u.m),
        'coj-clma-0m4a': EarthLocation(lat='31d16m22.38sS', lon='149d4m15.05sE', height=1191.0 * u.m),
        'coj-clma-0m4b': EarthLocation(lat='31d16m22.48sS', lon='149d4m14.91sE', height=1191.0 * u.m),
        'elp-doma-1m0a': EarthLocation(lat='30d40m47.53sN', lon='104d0m54.63sW', height=2010.0 * u.m),
        'elp-domb-1m0a': EarthLocation(lat='30d40m48.00sN', lon='104d0m55.74sW', height=2029.4 * u.m),
        'elp-aqwa-0m4a': EarthLocation(lat='30d40m48.15sN', lon='104d0m54.24sW', height=2027.0 * u.m),
        'lsc-doma-1m0a': EarthLocation(lat='30d10m2.58sS', lon='70d48m17.24sW', height=2201.0 * u.m),
        'lsc-domb-1m0a': EarthLocation(lat='30d10m2.39sS', lon='70d48m16.78sW', height=2201.0 * u.m),
        'lsc-domc-1m0a': EarthLocation(lat='30d10m2.81sS', lon='70d48m16.85sW', height=2201.0 * u.m),
        'lsc-aqwa-0m4a': EarthLocation(lat='30d10m3.79sS', lon='70d48m16.88sW', height=2202.5 * u.m),
        'lsc-aqwb-0m4a': EarthLocation(lat='30d10m3.56sS', lon='70d48m16.74sW', height=2202.5 * u.m),
        'cpt-doma-1m0a': EarthLocation(lat='32d22m50.0sS', lon='20d48m36.65sE', height=1807.0 * u.m),
        'cpt-domb-1m0a': EarthLocation(lat='32d22m50.0sS', lon='20d48m36.13sE', height=1807.0 * u.m),
        'cpt-domc-1m0a': EarthLocation(lat='32d22m50.38sS', lon='20d48m36.39sE', height=1807.0 * u.m),
        'cpt-aqwa-0m4a': EarthLocation(lat='32d22m50.25sS', lon='20d48m35.54sE', height=1804.0 * u.m),
        'tfn-doma-1m0a': EarthLocation(lat='28d18m1.56sN', lon='16d30m41.82sE', height=2406.0 * u.m),
        'tfn-domb-1m0a': EarthLocation(lat='28d18m1.8720sN', lon='16d30m41.4360sE', height=2406.0 * u.m),
        'tfn-aqwa-0m4a': EarthLocation(lat='28d18m1.11sN', lon='16d30m42.13sE', height=2390.0 * u.m),
        'tfn-aqwa-0m4a': EarthLocation(lat='28d18m1.11sN', lon='16d30m42.21sE', height=2390.0 * u.m),
    }

    # Center of the Earth
    if tel_code == 'geocenter':
        return EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)

    # Check to see if it is an LCO facility code first, because we have more precise
    # coordinates for each telescope than a generic site location
    if tel_code in lco_facilities.keys():
        return lco_facilities[tel_code]

    # If not found, use Astropy's built-in look-up table.  If this is fed an invalid
    # site ID code, it will automatically raise an UnknownSiteException, which is
    # the desired behaviour
    else:
        return EarthLocation.of_site(tel_code)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dateobs', help='Datetime in UTC of the exposure start, in ISOT format %Y-%m-%dT%H:%M:%S')
    parser.add_argument('RA', help='RA of the target or frame center in sexigesimal format, J2000.0')
    parser.add_argument('Dec', help='Dec of the target or frame center in sexigesimal format, J2000.0')
    parser.add_argument('tel_code', help="""Identifer code of the observatory site where the data were taken;
                      used with a look-up table to find the geographic location""")
    parser.add_argument('exp_time', help='Exposure time in seconds')
    args = parser.parse_args()

    hjd = calc_hjd(
        args.dateobs,
        args.RA,
        args.Dec,
        args.tel_code,
        float(args.exp_time)
    )
    print('Calculated HJD=' + str(hjd))
