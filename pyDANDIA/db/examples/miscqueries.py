import sys

from pyDANDIA import db
from pyDANDIA.db import mockup

if __name__=="__main__":
    conn = db.get_connection()
    with conn:
        db.feed_exposure(conn, *mockup.get_test_data())

#    print db.query_to_astropy_table(conn,
#        "SELECT jd, diff_flux 
#        " FROM phot"
#        "   NATURAL JOIN exposures"
#        " WHERE star_id=?", (int(sys.argv[1]),))

#    print db.query_to_astropy_table(conn,
#        "SELECT exposure_id, avg(diff_flux) as mean_diff"
#        " FROM phot"
#        " GROUP BY exposure_id")

#    print db.query_to_astropy_table(conn,
#        "SELECT star_id, avg(magnitude) as meanmag,"
#        "   max(magnitude)-avg(magnitude) as maxdiff"
#        " FROM phot"
#        " GROUP BY star_id"
#        " ORDER BY maxdiff DESC"
#        " LIMIT 5")

#    print db.query_to_astropy_table(conn,
#        "SELECT star_id, filter_id, magnitude, jd"
#        " FROM phot"
#        " NATURAL JOIN stars"
#        " NATURAL JOIN exposures"
#        " WHERE ra BETWEEN 269.521 AND 269.522"
#        "   AND dec BETWEEN -27.997 AND -27.995")

    print db.query_to_astropy_table(conn,
        "SELECT star_id, ra, dec, filter_id, avg(magnitude)"
        " FROM phot"
        " NATURAL JOIN stars"
        " NATURAL JOIN exposures"
        " WHERE ra BETWEEN 268. AND 270."
        "   AND dec BETWEEN -29. AND -27."
        " GROUP BY star_id, filter_id")




