import time
import os
from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd, '../../'))
systempath.append(path.join(cwd, '../'))

from phot_db import *
from astropy_interface import *

if __name__=="__main__":
    conn = get_connection(database_file_path)
    
    print "Testing a bunch of queries..."
    print query_to_astropy_table(conn,
        "SELECT jd, diff_flux"
        " FROM phot"
        " NATURAL JOIN exposures"
        " NATURAL JOIN stars"
        " WHERE star_id=%s" % str(random.randint(1,10)))
    time.sleep(1)
    
    print query_to_astropy_table(conn,
        "SELECT exposure_id, avg(diff_flux) as mean_diff"
        " FROM phot"
        " GROUP BY exposure_id")
    time.sleep(1)
    
    print query_to_astropy_table(conn,
        "SELECT star_id, avg(magnitude) as meanmag,"
        "	max(magnitude)-avg(magnitude) as maxdiff"
        " FROM phot"
        " GROUP BY star_id"
        " ORDER BY maxdiff DESC"
        " LIMIT 5")
    time.sleep(1)
    
    print query_to_astropy_table(conn,
        "SELECT star_id, magnitude, jd"
        " FROM phot"
        " NATURAL JOIN stars"
        " NATURAL JOIN exposures"
        " WHERE ra BETWEEN 260. AND 270."
        "	AND dec BETWEEN -30. AND -20."
        " LIMIT 10")
    time.sleep(1)
    
    print query_to_astropy_table(conn,
        "SELECT star_id, ra, dec, filter_id, avg(magnitude)"
        " FROM phot"
        " NATURAL JOIN stars"
        " NATURAL JOIN reference_images"
        " WHERE ra BETWEEN 260. AND 270."
        "	AND dec BETWEEN -30. AND -20."
        " GROUP BY star_id")
    time.sleep(1)
    
    print query_to_astropy_table(conn,
        "SELECT jd, diff_flux" 
        " FROM phot" 
        " NATURAL JOIN exposures")
    time.sleep(1)
    
    print query_to_astropy_table(conn,
        "SELECT star_id, jd, magnitude, magnitude_err"
        " FROM phot"
        " NATURAL JOIN stars"
        " NATURAL JOIN exposures"
        " WHERE star_id=3")
    time.sleep(1)
    
    print query_to_astropy_table(conn,
        "SELECT star_id, filter_id, magnitude"
        " FROM phot"
        " NATURAL JOIN stars"
        " NATURAL JOIN reference_images"
        " GROUP BY filter_id")
    
    cur = conn.cursor()
    cur.execute("SELECT jd, magnitude, magnitude_err, filter_id"
        " FROM phot"
        " NATURAL JOIN stars"
        " NATURAL JOIN exposures"
        " NATURAL JOIN reference_images"
        " WHERE star_id=1")
    
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()
