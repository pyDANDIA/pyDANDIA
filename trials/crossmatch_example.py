from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from astropy.coordinates import matching
from astropy.coordinates import SkyCoord

# Set up an array of RAs and Decs in decimal degrees
# e.g
#In [9]: ras
#Out[9]: 
#array([ 268.17589811,  268.17275629,  268.18676186, ...,  268.44096179,
#        268.43968446,  268.27462012])
#
#In [10]: decs
#Out[10]: 
#array([-30.09908157, -30.25485921, -30.21523021, ..., -30.11503113,
#       -30.11325601, -30.32588947])

# Here is an example of how to query Vizier and find all stars 
# around a given position (I do it for FIELD 05)
# you can specify a search radius or a width (square). A square runs faster.

fieldra = 268.35435

fielddec =  -30.2578356389

evcoord = "17:53:19.30 -30:12:38.60"

c = coord.SkyCoord(evcoord, unit=(u.hourangle, u.deg))

vphas_cat = Vizier.find_catalogs('VPHAS+ DR2 survey').keys()[0]

Vizier.ROW_LIMIT=-1 # for getting the complete catalog

# Query VPHAS+ catalog on Vizier around the target region
tst = Vizier.query_region(coord.SkyCoord(ra=fieldra,dec=fielddec,
                                   unit=(u.deg, u.deg),
                                   frame='icrs'),
                                   #radius=0.5*u.deg,
                                   width=0.34*u.deg,
                                   catalog=vphas_cat)

# This is the returned VPHAS+ astropy catalog table
vphas_cat_table = Table(tst[0])

# Now I match this table with what I found on my image of ROME FIELD 05
# Match the catalogs
mystars = SkyCoord(ras,decs,unit=u.deg)
catstars = SkyCoord(vphas_cat_table['RAJ2000'],vphas_cat_table['DEJ2000'],unit=u.deg)

match_table = matching.search_around_sky(mystars, catstars, seplimit=1.0*u.arcsec)

# This will return the matching indexes for both arrays. 
# Note that you can have many to many matchings in very crowded fields.
# For example, my query matches the following indexes for the first ten matches of the two input catalogs:
#In [41]: match_table[0][0:10]
#Out[41]: array([0, 0, 0, 1, 1, 2, 2, 3, 3, 5])
#
#In [42]: match_table[1][0:10]
#Out[42]: 
#array([223155, 223156, 223157, 150946, 150947, 197371, 197372, 200103,
#       200104, 159970])

# You can read more about the returned match_table here:
# http://docs.astropy.org/en/stable/coordinates/matchsep.html#searching-around-coordinates
