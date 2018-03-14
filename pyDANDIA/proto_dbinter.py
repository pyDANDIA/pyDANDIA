# export PHOTDB_PATH=/home/msdemlei/photdb

import os
import random
import re
import sqlite3

telescopes = {'Australia':[10.5,20.3,1235.6],'Chile':[40.1,60.4,2235.6]}
instruments = {'camera1':[1,2,3,'blah'],'camera2':[4,5,6,'bleh']}
filters = {'r':'SDSS-r','g':'SDSS-g','i':'SDSS-i'}

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


class Exposures(TableDef):
	"""The table storing individual sky exposures.
	"""
	c_000_exposure_id = 'INTEGER PRIMARY KEY'
	c_010_jd = 'DOUBLE PRECISION'
	c_020_telescope_id = 'TEXT'
	c_030_instrument_id = 'TEXT'
	c_040_filter_id = 'TEXT'
	c_050_fwhm = 'REAL'
	c_060_fwhm_err = 'REAL'
	c_050_ellipticity = 'REAL'
	c_060_ellipticity_err = 'REAL'
	c_110_airmass = 'REAL'
	c_120_exposure_time = 'INTEGER'
	c_130_moon_phase = 'REAL'
	c_140_moon_separation = 'REAL'
	c_150_name = 'TEXT'


class ReferenceImages(TableDef):
	"""A table with the (stacked) reference image.
	"""
	c_000_refimg_id = 'INTEGER PRIMARY KEY'
	c_010_astrometric_reference = 'INTEGER REFERENCES exposures(id)'
	c_120_name = 'TEXT'


class Stars(TableDef):
	"""The object list.
	"""
	c_000_star_id = 'INTEGER PRIMARY KEY'
	c_010_ra = 'DOUBLE PRECISION'
	c_020_dec = 'DOUBLE PRECISION'
	c_030_x_pix = 'REAL'
	c_040_y_pix = 'REAL'
	c_050_reference_image = 'INTEGER REFERENCES ReferenceImages(refimg_id)'
	c_060_reference_flux = 'DOUBLE PRECISION'
	c_070_reference_flux_err= 'DOUBLE PRECISION'
	c_100_type = 'TEXT'


class PhotometryPoints(TableDef):
	"""The table storing the primary information on the measurements taken.
	"""
	c_000_phot_id = 'INTEGER PRIMARY KEY'
	c_010_exposure_id = 'INTEGER REFERENCES exposures(exposure_id)'
	c_020_star_id ='INTEGER REFERENCES stars(star_id)'
	c_030_diff_flux = 'REAL'
	c_040_diff_flux_err = 'REAL'
	c_050_magnitude = 'REAL'
	c_060_magnitude_err = 'REAL'
	c_070_phot_scale_factor = 'REAL'
	c_080_phot_scale_factor_error = 'REAL'
	c_090_local_background = 'REAL'
	c_100_local_background_error = 'REAL'
	c_110_delta_x = 'REAL'
	c_120_delta_y = 'REAL'
	c_130_residual_x = 'REAL'
	c_140_residual_y = 'REAL'

	pc_000_datesindex = (
		'CREATE INDEX IF NOT EXISTS phot_objs ON phot (star_id)')


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


def get_connection(dsn=os.environ["PHOTDB_PATH"]):
	conn = sqlite3.connect(dsn,
		detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
	conn.execute("PRAGMA foreign_keys=ON")
	ensure_tables(conn, 
		EXPOSURES_TD, REFERENCE_IMAGES_TD, STARS_TD, PHOTOMETRY_TD)
	return conn


def _feed_to_table(conn, table_name, names, values):
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


def _feed_to_table_many(conn, table_name, rows):
	"""dumps a list of (structurally identical!) dictionary to the database.
	"""
	names = rows[0].keys()
	conn.executemany("INSERT OR REPLACE INTO %s (%s) VALUES (%s)"%(
		table_name, ",".join(names), ",".join("?"*len(names))),
		[[d[n] for n in names] for d in rows])


def feed_exposure(conn, exp_properties, photometry_points):
	"""feed extract from a new image.

	exp_properties is a dictionary with keys named like the Exposures
	field.

	photometry_points is a list of dicts with keys from PhotometryPoints.
	Leave empty exposure field.
	"""
	exposure_id = _feed_to_table(conn, "exposures", 
		exp_properties.keys(), exp_properties.values())
	
	for row in photometry_points:
		row["exposure_id"] = exposure_id
	
	_feed_to_table_many(conn, "phot",
		photometry_points)


def create_stars_if_necessary(conn, n):
	"""stuffs n random stars into the stars table if there's less rows in 
	there.

	(warning: minimal star id is 1!)
	"""
	if list(conn.execute("SELECT COUNT(*) FROM stars"))[0][0]<n:
		for i in range(n):
			_feed_to_table(conn, "stars", [
					"ra", 
					"dec"
				], [
					random.normalvariate(266.4, 15), 
					random.normalvariate(-29, 15),
			])

def get_test_data():
	"""returns fake data for test and demo purposes.

	It's an exposure row and a sequence of photometry records.
	"""
	return {
			'jd': 2540000+random.random()*500,
			'telescope_id': ['fred', 'joe'][random.randint(0, 1)],
			'instrument_id': ['cam1','cam2'][random.randint(0, 1)],
			'filter_id': ['filt1','filt2'][random.randint(0, 1)],
			'airmass': random.random()+0.1,
			'name': '/foo/bar/baz%0d.fz'%random.randint(0, 100000000),
			'fwhm' : random.random()+0.1,
			'fwhm_err' : random.random()+0.1,
			'ellipticity' : random.random()+0.1,
			'ellipticity_err' : random.random()+0.1,
			'exposure_time' : random.randint(0, 1)
			'moon_phase' : random.random()+0.1,
			'moon_separation' : np.random.uniform(0,60)+0.1
		}, [{
				'star_id': i+1,
				'diff_flux': random.random()*10,
				'diff_flux_err': random.random()*0.1,
				'phot_scale_factor': 0.14+random.random()*0.1,
				'phot_scale_factor_error': 0.014+random.random()*0.01,
				'magnitude': random.uniform(14,19),
				'magnitude_error': 0.014+random.random()*0.01,
				'local_background': random.random(),
				'local_background_error': random.random()/100,
				'delta_x': 0.1*random.random(),
				'delta_y': 0.1*random.random(),
				'residual_x': 0.01*random.random(),
				'residual_y': 0.01*random.random(),
				} for i in range(random.randint(5, 15))]

if __name__=="__main__":
	conn = get_connection()
	with conn:
		create_stars_if_necessary(conn, 20)
		feed_exposure(conn, *get_test_data())
