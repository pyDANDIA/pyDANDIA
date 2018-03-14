"""
Read and write tables to astropy tables.

[This is missing a lot of metadata -- figure out how to cope]
"""

from astropy import table

from . import common


def load_astropy_table(conn, db_table_name, table):
	"""ingests the astropy table table into db_table_name via conn.
	"""
	common.feed_to_table_many(
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

	data = [
		table.Column(name=k,
			data=getColumn(i))
		for i,k in enumerate(keys)]
	return table.Table(data=data)
