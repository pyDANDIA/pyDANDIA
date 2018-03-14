"""
Read and write tables to astropy tables.

[This is missing a lot of metadata -- figure out how to cope]
"""

from . import common


def load_astropy_table(conn, db_table_name, table):
	"""ingests the astropy table table into db_table_name via conn.
	"""
	common.feed_to_table_many(
		conn,
		db_table_name,
		table.colnames,
		[tuple(r) for r in table])
