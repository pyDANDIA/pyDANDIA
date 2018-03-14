# Not finished -- just here as a reminder for how this might work.
from astropy import table
from astropy import coordinates
import numpy as np

from pyDANDIA import db


if __name__=="__main__":
    #sc would now be extractrion result
    with open("gavo/sci/pyDANDIA/pyDANDIA/tests/data/star_catalog.fits") as f:
        extracted_objects = table.Table.read(f)
    extracted_objects["RA_J2000_deg"] += np.random.uniform(
        -1e-5, 1e-5, len(extracted_objects))
    extracted_objects["Dec_J2000_deg"] += np.random.uniform(
        -1e-5, 1e-5, len(extracted_objects))
    
    conn = db.get_connection()
    reference_catalog = db.query_to_astropy_table(
        conn, 
        "select * from stars") # or restrict to field in question

    reference = coordinates.SkyCoord(
        reference_catalog["ra"], reference_catalog["dec"])
    to_match = coordinates.SkyCoord(
        reference_catalog["RA_J2000_deg"], reference_catalog["Dec_J2000_deg"])
    match = to_match.match_to_catalog_sky(reference)
    print match

