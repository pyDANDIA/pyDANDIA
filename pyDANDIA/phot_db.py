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
from astropy import table

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


class TargetFields(TableDef):
    """A table with the target field definition.
    """
    c_000_field_id = 'INTEGER PRIMARY KEY'
    c_020_ra_center = 'REAL'
    c_030_dec_center = 'REAL'

class Exposures(TableDef):
    """The table storing individual sky exposures.
    """
    c_000_exposure_id = 'INTEGER PRIMARY KEY'
    c_005_field_id = 'INTEGER REFERENCES target_fields(field_id)'
    c_010_jd = 'DOUBLE PRECISION'
    c_050_exposure_fwhm = 'REAL'
    c_060_exposure_fwhm_err = 'REAL'
    c_050_exposure_ellipticity = 'REAL'
    c_060_exposure_ellipticity_err = 'REAL'
    c_110_airmass = 'REAL'
    c_120_exposure_time = 'REAL'
    c_130_moon_phase = 'REAL'
    c_140_moon_separation = 'REAL'
    c_150_delta_x = 'REAL'
    c_160_delta_y = 'REAL'    
    c_170_exposure_name = 'TEXT'
    c_180_filter = 'TEXT'

class ReferenceImages(TableDef):
    """A table with the (stacked) reference image.
    """
    c_000_refimg_id = 'INTEGER PRIMARY KEY'
    c_005_field_id = 'INTEGER REFERENCES target_fields(field_id)'
    c_020_telescope_id = 'TEXT'
    c_030_instrument_id = 'TEXT'
    c_040_filter = 'TEXT'
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
    c_160_stage3_version = 'TEXT'
    c_170_current_best = 'INTEGER'

class Stars(TableDef):
    """The object list.f
    """
    c_000_star_id = 'INTEGER PRIMARY KEY'
    c_005_reference_images = 'INTEGER REFERENCES reference_images(refimg_id)'
    c_010_ra = 'DOUBLE PRECISION'
    c_020_dec = 'DOUBLE PRECISION'    
    c_100_type = 'TEXT'
    
    pc_000_raindex = (
        'CREATE INDEX IF NOT EXISTS stars_ra ON stars (ra)')
    pc_010_decindex = (
        'CREATE INDEX IF NOT EXISTS stars_dec ON stars (dec)')

class ReferencePhotometry(TableDef):
    """The table storing the primary information on the measurements taken.
    """
    c_000_ref_phot_id = 'INTEGER PRIMARY KEY'
    c_021_star_id = 'INTEGER REFERENCES stars(star_id)'
    c_022_reference_mag = 'REAL'
    c_023_reference_mag_err = 'REAL'
    c_024_reference_flux = 'DOUBLE PRECISION'
    c_025_reference_flux_err = 'DOUBLE PRECISION'
    c_041_cal_reference_mag = 'REAL'
    c_042_cal_reference_mag_err = 'REAL'
    c_052_reference_x = 'REAL'
    c_053_reference_y = 'REAL'
    
class PhotometryPoints(TableDef):
    """The table storing the primary information on the measurements taken.
    """
    c_000_phot_id = 'INTEGER PRIMARY KEY'
    c_010_exposure_id = 'INTEGER REFERENCES exposures(exposure_id)'
    c_025_ref_phot_id = 'INTEGER REFERENCES ref_phot(ref_phot_id)'
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
    # c_150_stage6_version = 'TEXT'
    # c_160_current_best = 'INTEGER'
    
    pc_000_datesindex = (
        'CREATE INDEX IF NOT EXISTS phot_objs ON phot (star_id)')


# This is what the classes are actually called in the database schema
EXPOSURES_TD = Exposures("exposures")
REFERENCE_IMAGES_TD = ReferenceImages("reference_images")
STARS_TD = Stars("stars")
PHOTOMETRY_TD = PhotometryPoints("phot")
REFERENCEPHOT_TD = ReferencePhotometry("ref_phot")
FIELD_TD = TargetFields("target_fields")

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
        EXPOSURES_TD, REFERENCE_IMAGES_TD, STARS_TD,
        PHOTOMETRY_TD, REFERENCEPHOT_TD, FIELD_TD)
    return conn

def update_table_entry(conn,table_name,key_name,search_key,entry_id,value):
    """Function to commit a single-value entry for a single keyword in a
    given table
    
    :param Connection conn: Open DB connection object
    :param string table_name: Name of the DB table to insert into
    :param string key_name: Entry keyword to be modified in the DB table
    :param string search_key: Search keyword to identify entry (normally the
                                table PK)
    :param int entry_id: PK index in table of the entry to be modified
    :param dtype value: Values of the keyword to be set
    """
    
    cursor = conn.cursor()
    try:
        
        command = 'UPDATE '+str(table_name)+ ' SET '+key_name+' = '+str(value)+\
                 ' WHERE '+search_key+' = '+str(entry_id)

        cursor.execute(command)
        
    finally:

        conn.commit()

        cursor.close()

def feed_to_table(conn, table_name, names, values):
    """makes a row out of names and values and inserts it into table_name.

    This returns the value of last_insert_rowid(), whether or not that
    actually has a meaning.
    
    :param Connection conn: Open DB connection object
    :param string table_name: Name of the DB table to insert into
    :param list names: List of entry keywords in the DB table
    :param list values: List of values corresponding to the keywords
    """
    cursor = conn.cursor()
    try:
        
        command = 'INSERT OR REPLACE INTO '+str(table_name)+ ' (' + \
                    ','.join(names)+') VALUES ('+\
                    ','.join("?"*len(values)) + ')'
        
        cursor.execute(command, values)
        
        return list(cursor.execute("SELECT last_insert_rowid()"))[0][0]
        
    finally:

        conn.commit()

        cursor.close()


def feed_to_table_many(conn, table_name, names, tuples):
    """dumps a sequence of tuples into table_name.

    names gives the sequence of column names per tuple element.
    !!! careful not to include the PRIMARY KEY column for the table in the 
        names or tuples !!!
    """

    command = 'INSERT OR REPLACE INTO ' + str(table_name) + ' (' +\
                                        ','.join(names) + ') ' +\
                                        ' VALUES ('+\
                                         ','.join("?"*len(names)) + ')'
    
    conn.executemany(command, tuples)
    
    conn.commit()


def update_stars_ref_image_id(conn, ref_image_name, star_ids):
    """Dumps a sequence of tuples into table_name, where foreign keys 
    require a table JOIN.

    names gives the sequence of column names per tuple element.
    !!! careful not to include the PRIMARY KEY column for the table in the 
        names or tuples !!!
    """

    command = 'UPDATE stars SET reference_images=(SELECT refimg_id FROM reference_images WHERE refimg_name="lsc1m005-fl15-20170418-0131-e91_cropped.fits") WHERE star_id=(?)'
    
    print(command)
    conn.executemany(command, star_ids)
    
    conn.commit()
    
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

def ingest_reference_in_db(conn, setup, reference_header, 
                           reference_image_directory, reference_image_name,
                           field_id, version):
    """Function to ingest a ReferenceImage to the photometric database

    Parameters added:
        c_020_telescope_id = 'TEXT'
        c_030_instrument_id = 'TEXT'
        c_040_filter_id = 'TEXT'
        c_045_field_id = 'TEXT'
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
        c_160_stage3_version = 'TEXT'
        c_170_current_best = 'INTEGER'
    """

    names = ('refimg_name', 'telescope_id', 'instrument_id', 
             'filter_id', 'field_id', 'refimg_fwhm', 'refimg_fwhm_err', 'refimg_ellipticity',
             'refimge_ellipticity_err', 'refimg_name', 'wcsfrcat', 'wcsimcat', 'wcsmatch', 'wcsnref', 'wcstol',
             'wcsra', 'wcsdec', 'wequinox', 'wepoch', 'radecsys', 'cdelt1', 'cdelt2', 'crota1', 'crota2', 'secpix1',
             'secpix2',
             'wcssep', 'equinox', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2', 'epoch',
             'stage3_version', 'current_best')
                            
    data = [ table.Column(name='refimg_name', data=[reference_image_name]),
                  table.Column(name='filter_id', data=[reference_header['FILTKEY']]),
                  table.Column(name='telescope_id', data=[reference_image_name.split('-')[0]]),
                  table.Column(name='instrument_id', data=[reference_image_name.split('-')[1]]),
                  table.Column(name='field_id', data=[field_id]),
                  table.Column(name='stage3_version', data=[version]) ]
            
    new_table = table.Table(data=data)
    
    ingest_astropy_table(conn, 'reference_images', new_table)
    conn.commit()
    
    # Workaround for known bug with sqlite3 ingestion of integer data, which
    # it mis-interprets as binary 'Blobs'.  
    query = 'SELECT refimg_name,refimg_id,current_best FROM reference_images'
    t = query_to_astropy_table(conn, query, args=())
    ref_id = t['refimg_id'].data[-1]

    update_table_entry(conn,'reference_images','current_best','refimg_id',ref_id,1)
    
    conn.commit()

def ingest_astropy_table(conn, db_table_name, table):
    """ingests an astropy table into db_table_name via conn.
    """

    feed_to_table_many(
        conn,
        db_table_name,
        table.colnames,
        [tuple(r) for r in table])

def query_to_astropy_table(conn, query, args=()):
    """tries to come up with a reasonable astropy table for a database
    query result.
    """
    cursor = conn.cursor()
    cursor.execute(query, args)
    keys = [cd[0] for cd in cursor.description]
    tuples = list(cursor)
    
    def getColumn(index):
        return [t[index] for t in tuples]
    
    data = [table.Column(name=k,
            data=getColumn(i))
        for i,k in enumerate(keys)]
    return table.Table(data=data)

def box_search_on_position(conn, ra_centre, dec_centre, dra, ddec):
    """Function to search the database for stars within (d(ra),d(dec)) of the 
    (ra_centre, dec_centre) given.
    
    :param connection conn: SQlite3 open connection object
    :param float ra_centre: Box central RA in decimal degrees
    :param float dec_centre: Box central Dec in decimal degrees
    :param float dra:       Box half-width in decimal degrees
    :param float ddec:      Box half-width in decimal degrees
    """
    
    ra_min = ra_centre - dra
    ra_max = ra_centre + dra
    dec_min = dec_centre - ddec
    dec_max = dec_centre + ddec
    
    query = 'SELECT star_id,ra,dec FROM stars WHERE ra BETWEEN '+\
            str(ra_min)+' AND '+str(ra_max)+\
            ' AND dec BETWEEN '+\
            str(dec_min)+' AND '+str(dec_max)
    
    t = query_to_astropy_table(conn, query, args=())
    
    return t
    
