"""
Common code to access the photometry database.

Essentially, you call <get_connection> and pass whatever you get back
to functions like feed_exposure etc

Alternatively, call conn.execute directly.

This requires the location of the database file in an environment variable
PHOTDB_PATH; e.g.,

export PHOTDB_PATH=~/photdb
"""

import re
import sqlite3
from os import getcwd, path, remove, environ
import numpy as np

########## THESE NEED TO BE CHANGED TO THE ACTUAL VALUES TO USE #################
environ["PHOTDB_PATH"] = '/home/Tux/ytsapras/Programs/Workspace/pyDANDIA/pyDANDIA/db/phot_db'
database_file_path = path.expanduser(environ["PHOTDB_PATH"])

telescopes = {'Chile':[40.1,60.4,2235.6]}
instruments = {'camera1':[1,2,3,'blah']}
filters = {'i':'SDSS-i'}
#################################################################################

class TableDef(object):
    """a definition of a table in an SQLite DB.

    The schema is defined in attributes in c_nnn_whatever names, post
    creation commands in attributes using pc_nnn_whatever names.  All
    this is to make inheritance useful for these table defs.

    Attributes include:

    * name -- the table name
    * schema -- a list of column name, column type pairs
    * columns -- a list of the column names
    """
    def __init__(self, name):
        self.name = name
        self.schema = self._make_schema()
        self.columns = [n for n, t in self.schema]

    _to_name_RE = re.compile(r"^c_\d+_(.*)")

    def _make_schema(self):
        return [(mat.group(1), getattr(self, mat.group())) 
             for mat in (self._to_name_RE.match(name) 
                for name in sorted(dir(self)))
            if mat]

    def _iter_post_create_statements(self):
        for cmd_attr in (s for s in sorted(dir(self)) if s.startswith("pc_")):
            yield getattr(self, cmd_attr)

    def iter_build_statements(self):
        macros = self.__dict__.copy()
        macros["ddl"] = ", ".join("%s %s"%t for t in self.schema)
        yield "CREATE TABLE IF NOT EXISTS %(name)s (%(ddl)s)"%macros
        for statement in self._iter_post_create_statements():
            yield statement%macros

class ReferenceImages(TableDef):
    """A table with the (stacked) reference image.
    """
    c_000_refimg_id = 'INTEGER PRIMARY KEY'
    c_020_telescope_id = 'TEXT'
    c_030_instrument_id = 'TEXT'
    c_040_filter_id = 'TEXT'
    c_050_refimg_fwhm = 'REAL'
    c_060_refimg_fwhm_err = 'REAL'
    c_070_refimg_ellipticity = 'REAL'
    c_080_refimg_ellipticity_err = 'REAL'
    c_090_slope = 'REAL' #The slope of the photometric calibration: VPHAS mags vs instr mags
    c_095_slope_err = 'REAL'
    c_100_intercept = 'REAL' #The intercept of the photometric calibration: VPHAS mags vs instr mags
    c_105_intercept_err = 'REAL'
    c_120_refimg_name = 'TEXT'
    c_130_wcsfrcat = 'TEXT' #WCS fit information stored in the next lines (c_130 to c_152)
    c_131_wcsimcat = 'TEXT'
    c_132_wcsmatch = 'INTEGER'
    c_133_wcsnref = 'INTEGER'
    c_134_wcstol = 'REAL'
    c_135_wcsra = 'TEXT'
    c_136_wcsdec = 'TEXT'
    c_137_wequinox = 'INTEGER'
    c_138_wepoch = 'INTEGER'
    c_139_radecsys = 'FK5'
    c_140_cdelt1 = 'DOUBLE PRECISION'
    c_141_cdelt2 = 'DOUBLE PRECISION'
    c_142_crota1 = 'DOUBLE PRECISION'
    c_143_crota2 = 'DOUBLE PRECISION'
    c_144_secpix1 = 'REAL'
    c_145_secpix2 = 'REAL'
    c_146_wcssep = 'REAL'
    c_147_equinox = 'INTEGER'
    c_148_cd1_1 = 'DOUBLE PRECISION'
    c_149_cd1_2 = 'DOUBLE PRECISION'
    c_150_cd2_1 = 'DOUBLE PRECISION'
    c_151_cd2_2 = 'DOUBLE PRECISION'
    c_152_epoch = 'INTEGER'

class Exposures(TableDef):
    """The table storing individual sky exposures.
    """
    c_000_exposure_id = 'INTEGER PRIMARY KEY'
    c_005_reference_image = 'INTEGER REFERENCES reference_images(refimg_id)'
    c_010_jd = 'DOUBLE PRECISION'
    c_050_exposure_fwhm = 'REAL'
    c_060_exposure_fwhm_err = 'REAL'
    c_050_exposure_ellipticity = 'REAL'
    c_060_exposure_ellipticity_err = 'REAL'
    c_110_airmass = 'REAL'
    c_120_exposure_time = 'INTEGER'
    c_130_moon_phase = 'REAL'
    c_140_moon_separation = 'REAL'
    c_150_delta_x = 'REAL'
    c_160_delta_y = 'REAL'    
    c_170_exposure_name = 'TEXT'

class Stars(TableDef):
    """The object list.
    """
    c_000_star_id = 'INTEGER PRIMARY KEY'
    c_010_ra = 'DOUBLE PRECISION'
    c_020_dec = 'DOUBLE PRECISION'
    c_100_type = 'TEXT'

    pc_000_raindex = (
        'CREATE INDEX IF NOT EXISTS stars_ra ON stars (ra)')
    pc_010_decindex = (
        'CREATE INDEX IF NOT EXISTS stars_dec ON stars (dec)')

class PhotometryPoints(TableDef):
    """The table storing the primary information on the measurements taken.
    """
    c_000_phot_id = 'INTEGER PRIMARY KEY'
    c_010_exposure_id = 'INTEGER REFERENCES exposures(exposure_id)'
    c_020_star_id ='INTEGER REFERENCES stars(star_id)'
    c_022_reference_mag = 'REAL'
    c_025_reference_mag_err= 'REAL'
    c_022_reference_flux = 'DOUBLE PRECISION'
    c_025_reference_flux_err= 'DOUBLE PRECISION'
    c_030_diff_flux = 'DOUBLE PRECISION'
    c_040_diff_flux_err = 'DOUBLE PRECISION'
    c_050_magnitude = 'REAL'
    c_060_magnitude_err = 'REAL'
    c_070_phot_scale_factor = 'REAL'
    c_080_phot_scale_factor_err = 'REAL'
    c_090_local_background = 'DOUBLE PRECISION'
    c_100_local_background_err = 'DOUBLE PRECISION'
    c_130_residual_x = 'REAL'
    c_140_residual_y = 'REAL'

    pc_000_datesindex = (
        'CREATE INDEX IF NOT EXISTS phot_objs ON phot (star_id)')


# This is what the classes are actually called in the database schema
EXPOSURES_TD = Exposures("exposures")
REFERENCE_IMAGES_TD = ReferenceImages("reference_images")
STARS_TD = Stars("stars")
PHOTOMETRY_TD = PhotometryPoints("phot")


def ensure_table(conn, table_def):
    """makes sure the TableDef instance table_def exists on the database.
    """
    curs = conn.cursor()
    for stmt in table_def.iter_build_statements():
        curs.execute(stmt)
    curs.close()


def ensure_tables(conn, *table_defs):
    """creates tables if necessary.
    """
    for table_def in table_defs:
        ensure_table(conn, table_def)


def get_connection(dsn=database_file_path):
    conn = sqlite3.connect(dsn,
        detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_tables(conn, 
        EXPOSURES_TD, REFERENCE_IMAGES_TD, STARS_TD, PHOTOMETRY_TD)
    return conn


def feed_to_table(conn, table_name, names, values):
    """makes a row out of names and values and inserts it into table_name.

    This returns the value of last_insert_rowid(), whether or not that
    actually has a meaning.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO %s (%s) VALUES (%s)"%(
            table_name, ",".join(names), ",".join("?"*len(values))),
            values)
        return list(cursor.execute("SELECT last_insert_rowid()"))[0][0]
    finally:
        cursor.close()


def feed_to_table_many(conn, table_name, names, tuples):
    """dumps a sequence of tuples into table_name.

    names gives the sequence of column names per tuple element.
    !!! careful not to include the PRIMARY KEY column for the table in the 
        names or tuples !!!
    """
    conn.executemany("INSERT OR REPLACE INTO %s (%s) VALUES (%s)"%(
        table_name, ",".join(names), ",".join("?"*len(names))),
        tuples)

def feed_to_table_many_dict(conn, table_name, rows):
    """dumps a list of (structurally identical!) dictionaries to the database.
    """
    names = rows[0].keys()
    feed_to_table_many(conn, table_name,
        names,
        [[d[n] for n in names] for d in rows])


def feed_exposure(conn, exp_properties, photometry_points):
    """feed extract from a new image.

    exp_properties is a dictionary with keys named like the Exposures
    field.

    photometry_points is a list of dicts with keys from PhotometryPoints.
    Leave empty exposure field.
    """
    exposure_id = feed_to_table(conn, "exposures", 
        exp_properties.keys(), exp_properties.values())
    
    for row in photometry_points:
        row["exposure_id"] = exposure_id
    
    feed_to_table_many_dict(conn, "phot", photometry_points)


def _adaptFloat(f):
    return float(f)
sqlite3.register_adapter(np.float32, _adaptFloat)
sqlite3.register_adapter(np.float64, _adaptFloat)
