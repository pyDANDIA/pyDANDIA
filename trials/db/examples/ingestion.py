# Not finished -- just here as a reminder for how this might work.
from astropy import table
from astropy import coordinates
import numpy as np
import os
from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd, '../../'))
systempath.append(path.join(cwd, '../'))

from phot_db import *

conn = get_connection()
ensure_tables(conn)
#sc would now be extractrion result
with open("/home/Tux/ytsapras/Programs/Workspace/pyDANDIA/pyDANDIA/tests/data/star_catalog.fits") as f:
    extracted_objects = table.Table.read(f)
#extracted_objects["RA_J2000_deg"] += np.random.uniform(
#    -1e-5, 1e-5, len(extracted_objects))
#extracted_objects["Dec_J2000_deg"] += np.random.uniform(
#    -1e-5, 1e-5, len(extracted_objects))

reference_catalog = query_to_astropy_table(
    conn, 
    "select * from stars") # or restrict to field in question

reference = coordinates.SkyCoord(
    reference_catalog["ra"], reference_catalog["dec"],unit='deg')
to_match = coordinates.SkyCoord(
    extracted_objects["RA_J2000_deg"], extracted_objects["Dec_J2000_deg"],unit='deg')
match = to_match.match_to_catalog_sky(reference)
print match

