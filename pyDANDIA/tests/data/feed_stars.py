from pyDANDIA import db

from astropy import table


if __name__=="__main__":
	conn = db.get_connection()
	with open("star_catalog.fits") as f:
		star_cat_upstream = table.Table.read(f)

	fixed_table = table.Table(data=[
		table.Column(name="star_id",
			data=range(1, len(star_cat_upstream)+1),
			description="Star index"),
		table.Column(name="ra",
			data= star_cat_upstream["RA_J2000_deg"].data,
			unit="deg",
			description="TBD number of objects contributing here",
			meta={"ucd": "pos.eq.ra"}),
		table.Column(name="dec",
			data= star_cat_upstream["Dec_J2000_deg"].data,
			unit="deg"),
		table.Column(name="x_pix",
			data= star_cat_upstream["X_pixel"].data),
		table.Column(name="y_pix",
			data= star_cat_upstream["Y_pixel"].data),
		# todo: reference image
		table.Column(name="reference_flux",
			data= star_cat_upstream["Instr_mag"].data),
		table.Column(name="reference_flux_err",
			data= star_cat_upstream["Instr_mag_err"].data)])

	with conn:
		db.load_astropy_table(conn, "stars", fixed_table)

