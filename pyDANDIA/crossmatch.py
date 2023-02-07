from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.coordinates import SkyCoord
from astropy import units
from pyDANDIA import match_utils
import numpy as np
from os import path
import copy

class CrossMatchTable():
    """Class describing the cross-identification of stars between multiple
    datasets"""

    def __init__(self):
        self.field_index = Table()
        self.primary_ref_code = None
        self.primary_ref_dir = None
        self.primary_ref_filter = None
        self.gaia_dr = None
        self.datasets = Table()
        self.stars = Table()
        self.images = Table()
        self.stamps = Table()

    def create(self, params):
        xmatch = fits.HDUList()

        pref = params['primary_ref']
        self.primary_ref_code = pref
        self.primary_ref_dir = params['datasets'][pref][1]
        self.primary_ref_filter = None

        field_index_columns = [Column(name='field_id', data=[], dtype='int'),
                    Column(name='ra', data=[], dtype='float'),
                    Column(name='dec', data=[], dtype='float'),
                    Column(name='quadrant', data=[], dtype='int'),
                    Column(name='quadrant_id', data=[], dtype='int'),
                    Column(name='gaia_source_id', data=[], dtype='S19'),
                    Column(name=pref+'_index', data=[], dtype='int')]

        # Format: dataset_id: [reference status, red_dir path, filter]
        codes = []
        reddirs = []
        filters = []
        pref_index = []
        for dataset_id, dataset_info in params['datasets'].items():
            if dataset_id == pref:
                pref_index.append(1)
            else:
                pref_index.append(0)
            codes.append(dataset_id)
            reddirs.append(dataset_info[1])
            filters.append(dataset_info[2])
            if dataset_id != pref:
                field_index_columns.append(Column(name=dataset_id+'_index', data=[], dtype='int'))

        dataset_columns = [ Column(name='dataset_code', data=codes, dtype='str'),
                     Column(name='dataset_red_dir', data=reddirs, dtype='str'),
                     Column(name='dataset_filter', data=filters, dtype='S8'),
                     Column(name='primary_ref',data=pref_index, dtype='int'),
                     Column(name='norm_a0',data=np.zeros(len(codes)), dtype='float'),
                     Column(name='norm_a1',data=np.zeros(len(codes)), dtype='float'),
                     Column(name='norm_covar_0',data=np.zeros(len(codes)), dtype='float'),
                     Column(name='norm_covar_1',data=np.zeros(len(codes)), dtype='float'),
                     Column(name='norm_covar_2',data=np.zeros(len(codes)), dtype='float'),
                     Column(name='norm_covar_3',data=np.zeros(len(codes)), dtype='float'),
                     ]

        self.datasets = Table(dataset_columns)
        self.field_index = Table(field_index_columns)
        self.create_stars_table()
        self.create_images_table()
        self.create_stamps_table()

    def create_stars_table(self):

        # The stars table holds the reference image photometry for all stars
        # in each dataset.  Although not all fields have data from all sites,
        # to avoid having a variable number of table columns, the table is
        # scaled to the maximum possible set of datasets:
        filters = ['g', 'r', 'i']
        sitecodes = ['lsc_doma', 'lsc_domb', 'lsc_domc',
                     'cpt_doma', 'cpt_domb', 'cpt_domc',
                     'coj_doma', 'coj_domb', 'coj_domc']

        stars_columns = [  Column(name='field_id', data=[], dtype='int'),
                            Column(name='ra', data=[], dtype='float'),
                            Column(name='dec', data=[], dtype='float') ]

        for site in sitecodes:
            for f in filters:
                stars_columns.append( Column(name='cal_'+f+'_mag_'+site, data=[], dtype='float') )
                stars_columns.append( Column(name='cal_'+f+'_magerr_'+site, data=[], dtype='float') )
                stars_columns.append( Column(name='norm_'+f+'_mag_'+site, data=[], dtype='float') )
                stars_columns.append( Column(name='norm_'+f+'_magerr_'+site, data=[], dtype='float') )

        stars_columns.append(Column(name='gaia_source_id', data=[], dtype='S19'))
        stars_columns.append(Column(name='gaia_ra', data=[], dtype='float'))
        stars_columns.append(Column(name='gaia_ra_error', data=[], dtype='float'))
        stars_columns.append(Column(name='gaia_dec', data=[], dtype='float'))
        stars_columns.append(Column(name='gaia_dec_error', data=[], dtype='float'))
        stars_columns.append(Column(name='phot_g_mean_flux', data=[], dtype='float'))
        stars_columns.append(Column(name='phot_g_mean_flux_error', data=[], dtype='float'))
        stars_columns.append(Column(name='phot_bp_mean_flux', data=[], dtype='float'))
        stars_columns.append(Column(name='phot_bp_mean_flux_error', data=[], dtype='float'))
        stars_columns.append(Column(name='phot_rp_mean_flux', data=[], dtype='float'))
        stars_columns.append(Column(name='phot_rp_mean_flux_error', data=[], dtype='float'))
        stars_columns.append(Column(name='pm', data=[], dtype='float'))
        stars_columns.append(Column(name='pmra', data=[], dtype='float'))
        stars_columns.append(Column(name='pmra_error', data=[], dtype='float'))
        stars_columns.append(Column(name='pmdec', data=[], dtype='float'))
        stars_columns.append(Column(name='pmdec_error', data=[], dtype='float'))
        stars_columns.append(Column(name='parallax', data=[], dtype='float'))
        stars_columns.append(Column(name='parallax_error', data=[], dtype='float'))

        self.stars = Table(stars_columns)

    def create_images_table(self):
        # Index filename dataset_code filter hjd datetime exposure RA Dec moon_ang_separation moon_fraction airmass sigma_x sigma_y \
        # sky median_sky fwhm corr_xy nstars fraction_saturated_pix symmetry use_phot use_ref shift_x shift_y pscale pscale_error \
        # var_per_pix_diff n_unmasked skew_diff kurtosis_diff
        image_columns = [  Column(name='index', data=[], dtype='int'),
                            Column(name='filename', data=[], dtype='S80'),
                            Column(name='dataset_code', data=[], dtype='S80'),
                            Column(name='filter', data=[], dtype='S10'),
                            Column(name='hjd', data=[], dtype='float'),
                            Column(name='datetime', data=[], dtype='S25'),
                            Column(name='exposure', data=[], dtype='float'),
                            Column(name='RA', data=[], dtype='S15'),
                            Column(name='Dec', data=[], dtype='S15'),
                            Column(name='moon_ang_separation', data=[], dtype='float'),
                            Column(name='moon_fraction', data=[], dtype='float'),
                            Column(name='airmass', data=[], dtype='float'),
                            Column(name='sigma_x', data=[], dtype='float'),
                            Column(name='sigma_y', data=[], dtype='float'),
                            Column(name='sky', data=[], dtype='float'),
                            Column(name='median_sky', data=[], dtype='float'),
                            Column(name='fwhm', data=[], dtype='float'),
                            Column(name='corr_xy', data=[], dtype='float'),
                            Column(name='nstars', data=[], dtype='int'),
                            Column(name='frac_sat_pix', data=[], dtype='float'),
                            Column(name='symmetry', data=[], dtype='float'),
                            Column(name='use_phot', data=[], dtype='int'),
                            Column(name='use_ref', data=[], dtype='int'),
                            Column(name='shift_x', data=[], dtype='float'),
                            Column(name='shift_y', data=[], dtype='float'),
                            Column(name='pscale', data=[], dtype='float'),
                            Column(name='pscale_err', data=[], dtype='float'),
                            Column(name='var_per_pix_diff', data=[], dtype='float'),
                            Column(name='n_unmasked', data=[], dtype='float'),
                            Column(name='skew_diff', data=[], dtype='float'),
                            Column(name='kurtosis_diff', data=[], dtype='float'),
                            Column(name='warp_matrix_0', data=[], dtype='float'),
                            Column(name='warp_matrix_1', data=[], dtype='float'),
                            Column(name='warp_matrix_2', data=[], dtype='float'),
                            Column(name='warp_matrix_3', data=[], dtype='float'),
                            Column(name='warp_matrix_4', data=[], dtype='float'),
                            Column(name='warp_matrix_5', data=[], dtype='float'),
                            Column(name='warp_matrix_6', data=[], dtype='float'),
                            Column(name='warp_matrix_7', data=[], dtype='float'),
                            Column(name='warp_matrix_8', data=[], dtype='float'),
                            Column(name='warp_matrix_9', data=[], dtype='float'),
                            Column(name='warp_matrix_10', data=[], dtype='float'),
                            Column(name='warp_matrix_11', data=[], dtype='float'),
                            Column(name='warp_matrix_12', data=[], dtype='float'),
                            Column(name='warp_matrix_13', data=[], dtype='float'),
                            Column(name='warp_matrix_14', data=[], dtype='float'),
                            Column(name='warp_matrix_15', data=[], dtype='float'),
                            Column(name='qc_flag', data=[], dtype='int'),
                            ]
        self.images = Table(image_columns)

    def create_stamps_table(self):
        stamps_columns = [   Column(name='dataset_code', data=[], dtype='S80'),
                            Column(name='filename', data=[], dtype='S80'),
                            Column(name='stamp_id', data=[], dtype='int'),
                            Column(name='xmin', data=[], dtype='float'),
                            Column(name='xmax', data=[], dtype='float'),
                            Column(name='ymin', data=[], dtype='float'),
                            Column(name='ymax', data=[], dtype='float'),
                            Column(name='warp_matrix_0', data=[], dtype='float'),
                            Column(name='warp_matrix_1', data=[], dtype='float'),
                            Column(name='warp_matrix_2', data=[], dtype='float'),
                            Column(name='warp_matrix_3', data=[], dtype='float'),
                            Column(name='warp_matrix_4', data=[], dtype='float'),
                            Column(name='warp_matrix_5', data=[], dtype='float'),
                            Column(name='warp_matrix_6', data=[], dtype='float'),
                            Column(name='warp_matrix_7', data=[], dtype='float'),
                            Column(name='warp_matrix_8', data=[], dtype='float'),
                            ]
        self.stamps = Table(stamps_columns)

    def add_dataset_header(self, dataset_idx, dataset_code, dataset_info):

        if '/' in dataset_info[0][-1:]:
            red_dir = dataset_info[0][0:-1]
        else:
            red_dir = dataset_info[0]
        self.datasets.add_column(Column(name='dataset_code'+str(dataset_idx), data=[dataset_code], dtype='str'))
        self.datasets.add_column(Column(name='dataset_red_dir'+str(dataset_idx), data=[dataset_info[0]], dtype='str'))
        self.datasets.add_column(Column(name='dataset_filter'+str(dataset_idx), data=[dataset_info[1]], dtype='str'))

    def add_dataset(self, red_dir, filter_name):
        idx = self.count_datasets()
        self.add_dataset_header(idx, red_dir, filter_name)
        self.init_matched_stars_table()

        return idx

    def count_datasets(self):
        n_datasets = 0
        for col in self.datasets.colnames:
            if 'dataset' in col:
                n_datasets += 1
        return n_datasets

    def init_field_index(self, metadata, filter_id):

        self.datasets[0]['dataset_filter'] = filter_id

        ncol = len(self.field_index.colnames)

        for star in metadata.star_catalog[1]:
            self.field_index.add_row( [star['index'], star['ra'], star['dec'], 0, 0, star['gaia_source_id'], star['index']]+([0]*(ncol-7)) )

    def match_dataset_with_field_index(self, dataset_metadata, params, log,
                                    verbose=True):
        """Method to perform a crossmatch of the star_catalog from a single
        dataset's reduction against the field catalog.
        Note that the separation_threshold parameter is expected to be in
        decimal degrees

        During the matching process, new potential matches are compared
        against the list of existing known matches involving the same stars
        to check for duplications and to replace existing matches if a closer
        match is found.

        This process can result in a star that was previously matched to a
        catalogue object being removed from the matched_stars list, if a
        closer match is found.  For this reason, the match process is repeated,
        so that alternative matches can be sought for those stars.
        """

        log.info('Matching dataset against field index')

        matched_stars = match_utils.StarMatchIndex()
        orphans = match_utils.StarMatchIndex()

        # Cross-match all entries in the field catalog with those stars
        # detected in the current dataset:
        dataset_stars = SkyCoord( dataset_metadata.star_catalog[1]['ra'],
                                    dataset_metadata.star_catalog[1]['dec'],
                                    frame='icrs', unit=(units.deg, units.deg) )

        field_stars = SkyCoord(self.field_index['ra'], self.field_index['dec'],
                            frame='icrs', unit=(units.deg, units.deg) )

        (field_idx, separations2D, separations3D) = dataset_stars.match_to_catalog_3d(field_stars)

        # Add stars with matches within the separation_threshold to the
        # matched_stars index
        constraint = separations2D < params['separation_threshold']
        matching_dataset_index = np.arange(0,len(dataset_metadata.star_catalog[1]),1)[constraint]
        matching_field_index = field_idx[constraint]

        for j,jdataset in enumerate(matching_dataset_index):
            jfield = matching_field_index[j]
            field_star_row = self.field_index[jfield]
            dataset_star_row = dataset_metadata.star_catalog[1][jdataset]
            (star_added,matched_stars) = self.add_to_match_index(field_star_row, dataset_star_row,
                                    separations2D[jdataset], matched_stars, log,
                                    verbose=verbose)

        # Add all other stars to the orphan's list:
        unmatched_dataset_index = np.arange(0,len(dataset_metadata.star_catalog[1]),1)
        matched_star_indices = np.array(matched_stars.cat2_index) - 1
        unmatched_dataset_index = np.delete(unmatched_dataset_index,matched_star_indices)
        for jdataset in unmatched_dataset_index:
            dataset_star_row = dataset_metadata.star_catalog[1][jdataset]
            (star_added, orphans) = self.add_to_orphans(dataset_star_row, orphans,
                                                    log, verbose=verbose)

        return matched_stars, orphans

    def match_field_index_with_gaia_catalog(self, gaia_data, params, log):
        log.info('Matching field index against Gaia data')

        self.gaia_dr = params['gaia_dr']

        matched_stars = match_utils.StarMatchIndex()

        field_stars = SkyCoord(self.field_index['ra'], self.field_index['dec'],
                            frame='icrs', unit=(units.deg, units.deg) )
        gaia_stars = SkyCoord(gaia_data['ra'], gaia_data['dec'],
                            frame='icrs', unit=(units.deg, units.deg) )

        log.info('Matching '+str(len(field_stars))+' field stars against '+\
                str(len(gaia_stars))+' stars in the Gaia catalog')

        (field_idx, separations2D, separations3D) = gaia_stars.match_to_catalog_3d(field_stars)

        constraint = separations2D < params['separation_threshold']
        matching_gaia_index = np.arange(0,len(gaia_data),1)[constraint]
        matching_field_index = field_idx[constraint]
        separations = separations2D[constraint]
        log.info(str(len(matching_field_index))+' matches found between Gaia sources and detected field stars')

        for j,jgaia in enumerate(matching_gaia_index):
            jfield = matching_field_index[j]
            p = {'cat1_index': jfield,          # Index NOT target ID
                 'cat1_ra': self.field_index['ra'][jfield],
                 'cat1_dec': self.field_index['dec'][jfield],
                 'cat1_x': 0.0,
                 'cat1_y': 0.0,
                 'cat2_index': int(gaia_data['source_id'][jgaia]),  # Source ID
                 'cat2_ra': gaia_data['ra'][jgaia],
                 'cat2_dec': gaia_data['dec'][jgaia],
                 'cat2_x': 0.0,
                 'cat2_y': 0.0,
                 'separation': separations[j]}

            star_added = matched_stars.add_match(p, log=log, verbose=True)
            star = self.stars[jfield]
            if self.gaia_dr == 'Gaia-EDR3':
                self.stars[jfield] = [star['field_id'], star['ra'], star['dec']] + [0.0]*108 + \
                                    [int(gaia_data['source_id'][jgaia]),
                                    gaia_data['ra'][jgaia], gaia_data['ra_error'][jgaia],
                                    gaia_data['dec'][jgaia], gaia_data['dec_error'][jgaia],
                                    gaia_data['phot_g_mean_flux'][jgaia], gaia_data['phot_g_mean_flux_error'][jgaia],
                                    gaia_data['phot_bp_mean_flux'][jgaia], gaia_data['phot_bp_mean_flux_error'][jgaia],
                                    gaia_data['phot_rp_mean_flux'][jgaia], gaia_data['phot_rp_mean_flux_error'][jgaia],
                                    gaia_data['pm'][jgaia],
                                    gaia_data['pmra'][jgaia], gaia_data['pmra_error'][jgaia],
                                    gaia_data['pmdec'][jgaia], gaia_data['pmdec_error'][jgaia],
                                    gaia_data['parallax'][jgaia], gaia_data['parallax_error'][jgaia]]
            else:
                self.stars[jfield] = [star['field_id'], star['ra'], star['dec']] + [0.0]*108 + \
                                    [int(gaia_data['source_id'][jgaia]),
                                    gaia_data['ra'][jgaia], gaia_data['ra_error'][jgaia],
                                    gaia_data['dec'][jgaia], gaia_data['dec_error'][jgaia],
                                    gaia_data['phot_g_mean_flux'][jgaia], gaia_data['phot_g_mean_flux_error'][jgaia],
                                    gaia_data['phot_bp_mean_flux'][jgaia], gaia_data['phot_bp_mean_flux_error'][jgaia],
                                    gaia_data['phot_rp_mean_flux'][jgaia], gaia_data['phot_rp_mean_flux_error'][jgaia],
                                    0.0,
                                    0.0,0.0,
                                    0.0,0.0,
                                    0.0,0.0]

        for j in range(0,matched_stars.n_match,1):
            self.field_index['gaia_source_id'][matched_stars.cat1_index[j]] = str(matched_stars.cat2_index[j])

        log.info('Matched '+str(matched_stars.n_match)+' Gaia targets to stars in field index')


    def check_separation(self,dataset_star, field_star, separation_threshold,log):
        separation = dataset_star.separation(field_star)

        if separation <= separation_threshold:
            return True, separation
        else:
            return False, separation

    def add_to_match_index(self,field_star, star, separation, matched_stars, log,
                        verbose=False):
        """Expects field_star and star to be Table rows"""
        #print('Field: ', field_star['field_id'])
        #print('Dataset: ', star)
        #print('Separation: ', separation)

        p = {'cat1_index': field_star['field_id'],
             'cat1_ra': field_star['ra'],
             'cat1_dec': field_star['dec'],
             'cat1_x': 0.0,
             'cat1_y': 0.0,
             'cat2_index': star['index'],
             'cat2_ra': star['ra'],
             'cat2_dec': star['dec'],
             'cat2_x': 0.0,
             'cat2_y': 0.0,
             'separation': separation.value}

        star_added = matched_stars.add_match(p, log=log)

        return star_added, matched_stars

    def add_to_orphans(self,star, orphans, log, verbose=False):
        p = {'cat1_index': None,
             'cat1_ra': 0.0,
             'cat1_dec': 0.0,
             'cat1_x': 0.0,
             'cat1_y': 0.0,
             'cat2_index': star['index'],
             'cat2_ra': star['ra'],
             'cat2_dec': star['dec'],
             'cat2_x': 0.0,
             'cat2_y': 0.0,
             'separation': -1.0}

        # Append the orphan without checking for duplication, since no cat1
        # index is given
        star_added = orphans.add_match(p,log=log,replace_worse_matches=False)

        return star_added, orphans

    def update_field_index(self, dataset_code, matched_stars, orphans, dataset_metadata, log):

        # Update field index with matched stars
        log.info('Updating field index with matched stars:')
        for j in range(0,len(matched_stars.cat1_index),1):
            jfield = np.where(self.field_index['field_id'] == matched_stars.cat1_index[j])
            row = self.field_index[jfield]
            row[dataset_code+'_index'] = matched_stars.cat2_index[j]    # Cat 2 star ID NOT index
            self.field_index[jfield] = row
            #log.info(repr(row))

        # Append orphans to the end of the field index
        log.info('Updating field index with orphans:')
        log.info(repr(orphans.cat2_index))
        ncol = len(self.field_index.colnames)
        dataset_col = self.field_index.colnames.index(dataset_code+'_index')
        for j in range(0,len(orphans.cat2_index),1):
            jfield = len(self.field_index) + 1      # Becomes new star ID NOT index
            jdataset = orphans.cat2_index[j] - 1    # Converts cat 2 star ID to index
            gaia_id =  dataset_metadata.star_catalog[1][int(jdataset)]['gaia_source_id']
            row = [jfield, orphans.cat2_ra[j], orphans.cat2_dec[j], 0, 0, gaia_id] + [0]*(ncol-6)
            row[dataset_col] = orphans.cat2_index[j]    # Star ID in dataset
            self.field_index.add_row(row)
            #log.info(repr(row)+' dataset star '+str(j))

    def dataset_index(self, red_dir):
        """Method to search the header index of matched data directories and
        return it's index in the matched_stars table list, or -1 if not present"""

        if '/' in red_dir[-1:]:
            red_dir = red_dir[0:-1]

        idx = -1
        for col in self.datasets.colnames:
            #import pdb; pdb.set_trace()
            if 'dataset' in col and self.datasets[col][0] == red_dir:
                idx = int(col.replace('dataset',''))

        return idx

    def assign_stars_to_quadrants(self):
        """Function to assign stars to quadrants of the field pointing,
        based on sky position.  This will be used to identify stars in the
        full-field photometry tables, which would be too large to handle if
        the entire field was stored in a single file"""

        field_stars = SkyCoord(self.field_index['ra'], self.field_index['dec'],
                            frame='icrs', unit=(units.deg, units.deg) )

        ra_range = (field_stars.ra.min(), field_stars.ra.max())
        dra = (ra_range[1] - ra_range[0])/2.0
        dec_range = (field_stars.dec.min(), field_stars.dec.max())
        ddec = (dec_range[1] - dec_range[0])/2.0

        quadrants = {1: [ra_range[0], ra_range[0]+dra, dec_range[0], dec_range[0]+ddec],
                     2: [ra_range[0]+dra, ra_range[1], dec_range[0], dec_range[0]+ddec],
                     3: [ra_range[0]+dra, ra_range[1], dec_range[0]+ddec, dec_range[1]],
                     4: [ra_range[0], ra_range[0]+dra, dec_range[0]+ddec, dec_range[1]]}

        nstars_quadrants = [0,0,0,0]

        for j in range(0,len(self.field_index),1):
            star = self.field_index[j]

            for q, quad_data in quadrants.items():
                if field_stars[j].ra >= quad_data[0] and field_stars[j].ra <= quad_data[1] \
                    and field_stars[j].dec >= quad_data[2] and field_stars[j].dec <= quad_data[3]:
                    nstars_quadrants[q-1] += 1
                    star['quadrant'] = q
                    star['quadrant_id'] = nstars_quadrants[q-1]
                    self.field_index[j] = star

    def init_stars_table(self):

        nstars = len(self.field_index['field_id'])

        filters = ['g', 'r', 'i']
        sitecodes = ['lsc_doma', 'lsc_domb', 'lsc_domc',
                     'cpt_doma', 'cpt_domb', 'cpt_domc',
                     'coj_doma', 'coj_domb', 'coj_domc']

        stars_columns = [  Column(name='field_id', data=self.field_index['field_id'], dtype='int'),
                            Column(name='ra', data=self.field_index['ra'], dtype='float'),
                            Column(name='dec', data=self.field_index['dec'], dtype='float') ]

        for site in sitecodes:
            for f in filters:
                stars_columns.append( Column(name='cal_'+f+'_mag_'+site, data=np.zeros(nstars), dtype='float') )
                stars_columns.append( Column(name='cal_'+f+'_magerr_'+site, data=np.zeros(nstars), dtype='float') )
                stars_columns.append( Column(name='norm_'+f+'_mag_'+site, data=np.zeros(nstars), dtype='float') )
                stars_columns.append( Column(name='norm_'+f+'_magerr_'+site, data=np.zeros(nstars), dtype='float') )

        stars_columns.append(Column(name='gaia_source_id', data=np.array(['0.0']*nstars), dtype='S19'))
        stars_columns.append(Column(name='gaia_ra', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='gaia_ra_error', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='gaia_dec', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='gaia_dec_error', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='phot_g_mean_flux', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='phot_g_mean_flux_error', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='phot_bp_mean_flux', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='phot_bp_mean_flux_error', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='phot_rp_mean_flux', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='phot_rp_mean_flux_error', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='pm', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='pm_ra', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='pm_ra_error', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='pm_dec', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='pm_dec_error', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='parallax', data=np.zeros(nstars), dtype='float'))
        stars_columns.append(Column(name='parallax_error', data=np.zeros(nstars), dtype='float'))

        self.stars = Table(stars_columns)

    def record_dataset_stamps(self, dataset_code, dataset_metadata, log):

        images = dataset_metadata.images_stats[1]['IM_NAME'].tolist()
        list_of_stamps = dataset_metadata.stamps[1]['PIXEL_INDEX'].tolist()

        for image in images:
            for stamp in list_of_stamps:
                stamp_row = np.where(dataset_metadata.stamps[1]['PIXEL_INDEX'] == stamp)[0][0]
                xmin = int(dataset_metadata.stamps[1][stamp_row]['X_MIN'])
                xmax = int(dataset_metadata.stamps[1][stamp_row]['X_MAX'])
                ymin = int(dataset_metadata.stamps[1][stamp_row]['Y_MIN'])
                ymax = int(dataset_metadata.stamps[1][stamp_row]['Y_MAX'])

                self.stamps.add_row([dataset_code, image, stamp, xmin, xmax, ymin, ymax]+[0.0]*9)

        log.info('Recorded stamp dimensions for dataset '+dataset_code)

    def save(self, file_path):
        """Output crossmatch table to file"""
        hdr = fits.Header()
        hdr['NAME'] = 'Field crossmatch table'
        hdr['PRIREFID'] = self.primary_ref_code
        hdr['GAIA_DR'] = self.gaia_dr

        hdu_list = [fits.PrimaryHDU(header=hdr),
                    fits.BinTableHDU(self.field_index, name='FIELD_INDEX'),
                    fits.BinTableHDU(self.datasets, name='DATASETS'),
                    fits.BinTableHDU(self.stars, name='STARS'),
                    fits.BinTableHDU(self.images, name='IMAGES'),
                    fits.BinTableHDU(self.stamps, name='STAMPS')]
        hdu_list = fits.HDUList(hdu_list)
        hdu_list.writeto(file_path, overwrite=True)


    def load_matched_stars(self, binary_table):
        """Method to load the matched_stars list"""

        matched_stars = match_utils.StarMatchIndex()

        matched_stars.cat1_index = list(binary_table.data['field_star_id'])
        matched_stars.cat1_ra = list(binary_table.data['field_ra'])
        matched_stars.cat1_dec = list(binary_table.data['field_dec'])
        matched_stars.cat1_x = list(binary_table.data['field_x'])
        matched_stars.cat1_y = list(binary_table.data['field_y'])
        matched_stars.cat2_index = list(binary_table.data['dataset_star_id'])
        matched_stars.cat2_ra = list(binary_table.data['dataset_ra'])
        matched_stars.cat2_dec = list(binary_table.data['dataset_dec'])
        matched_stars.cat2_x = list(binary_table.data['dataset_x'])
        matched_stars.cat2_y = list(binary_table.data['dataset_y'])
        matched_stars.separation = list(binary_table.data['separation'])
        matched_stars.n_match = len(matched_stars.cat1_index)

        return matched_stars

    def load(self,file_path, log=None):
        """Method to load an existing crossmatch table"""

        def load_binary_table(hdu_list, hdu_index):
            table_data = []
            for col in hdu_list[hdu_index].columns.names:
                table_data.append( Column(hdu_list[hdu_index].data[col], name=col) )
            return Table( table_data )

        if not path.isfile(file_path):
            raise IOError('Cannot find cross-match table at '+file_path)

        hdu_list = fits.open(file_path, mmap=True)
        self.gaia_dr = hdu_list[0].header['GAIA_DR']
        self.field_index = load_binary_table(hdu_list, 1)
        self.datasets = load_binary_table(hdu_list, 2)
        self.stars = load_binary_table(hdu_list, 3)
        self.images = load_binary_table(hdu_list, 4)
        self.stamps = load_binary_table(hdu_list, 5)

        if log:
            log.info('Loaded crossmatch table from '+file_path)

    def locate_stars_in_field_index(self, field_ids):
        """Method to return the field_index indices of a set of stars based on
        an input list of field_index identifiers"""

        field_index = []
        for star_id in field_ids:
            idx = np.where(self.field_index['field_id'] == star_id)[0]
            if len(idx) == 1:
                field_index.append(idx[0])
            else:
                field_index.append(None)
        return field_index

    def cone_search(self, params, log=None, debug=False):
        """Method to perform a cone search on the field index for all objects
        within the search radius (in decimal degrees) of the (ra_center, dec_centre)
        given"""

        starlist = SkyCoord(self.field_index['ra'], self.field_index['dec'],
                            frame='icrs', unit=(units.deg,units.deg))

        target = SkyCoord(params['ra_centre'], params['dec_centre'],
                            frame='icrs', unit=(units.deg,units.deg))

        separations = target.separation(starlist)

        idx = np.where(separations.value <= params['radius'])[0]
        results = self.field_index[idx]
        results.add_column(separations[idx], name='separation')

        log.info('Identified '+str(len(results))+' candidates within '+str(params['radius'])+\
                    ' of ('+str(params['ra_centre'])+', '+str(params['dec_centre'])+')')
        log.info(' '.join(results.colnames))
        for star in results:
            log.info(' '.join(str(star[col]) for col in results.colnames))

        if debug and len(idx[0]) == 0:
            idx = np.where(separations.value == separations.value.min())
            if log!=None:
                log.info('Nearest closest star: ')
                log.info(self.field_index['field_id'][idx], self.field_index['ra'][idx], self.field_index['dec'][idx], separations[idx])

        return results

    def id_primary_datasets_per_filter(self):
        ref_dataset = {'g': None, 'r': None, 'i': None}

        primary_ref = np.where(self.datasets['primary_ref'] == 1)[0]
        f_primary_ref = self.datasets['filter'][primary_ref]
        ref_dataset[f_primary_ref] = self.datasets['dataset_code'][primary_ref]
        entries = self.datasets['dataset_code'][primary_ref].split('_')
        reference_source = entries[0]+'_'+entries[1]

        for f in ['g', 'r', 'i']:
            if ref_dataset[f] == None:
                ref_dataset[f] = reference_source+'_'+f+'p'

        self.reference_datasets = ref_dataset
