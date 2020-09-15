from os import path
from sys import argv
from pyDANDIA import crossmatch
from pyDANDIA import log

def build_crossmatch_table(params):

    log = logs.start_stage_log( params['log_dir'], 'crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    test_xmatch = crossmatch.CrossMatchTable()

    # Load an existing crossmatch table, or create one if none exists.
    # This initializes empty matched_stars tables for each dataset
    if path.isfile(params['file_path']):
        xmatch.load(params['file_path'])
        log.info('Loaded existing crossmatch table from '+params['file_path'])
    else:
        xmatch.create(params)
        log.info('Initialized new crossmatch table')

    # Load the star catalog from the primary dataset's metadata:
    primary_metadata = metadata.MetaData()
    primary_metadata.load_all_metadata(xmatch.header['PRIMARY'], 'pyDANDIA_metadata.fits')
    log.info('Loaded primary reference metadata from '+xmatch.header['PRIMARY'])

    for i, red_dir in enumerate(params['red_dir_list']):
        log.info('Performing cross-match with primary reference for dataset '+red_dir)

        # Fetch the index of the current dataset in the table, or create an
        # empty table for a new dataset being added to an existing table
        dataset_idx = xmatch.dataset_index(red_dir)
        if dataset_idx == -1:
            dataset_idx = xmatch.add_dataset(red_dir, params['red_dataset_filters'][i])
            log.info('Added '+path.basename(red_dir)+' to the crossmatch table')
        log.info('Index of dataset in crossmatch table: '+str(dataset_idx))

        dataset_metadata = metadata.MetaData()
        dataset_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')
        log.info('Loaded dataset metadata')

        matched_stars = match_dataset_with_primary_reference(primary_metadata, dataset_metadata,
                                                log,verbose=False)

        (transform_xy,transform_sky) = calc_transform_to_primary_ref(setup,matched_stars,log)

        # Output the matched_stars table to the dataset's own metadata:
        dataset_metadata.create_matched_stars_layer(matched_stars)
        dataset_metadata.create_transform_layer(transform_xy)
        dataset_metadata.save_a_layer_to_file(red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'matched_stars', log)
        dataset_metadata.save_a_layer_to_file(red_dir,
                                            'pyDANDIA_metadata.fits',
                                            'transformation', log)

        # Update the appropriate matched_stars table in the crossmatch table:
        xmatch.update_matched_stars_table(dataset_idx, matched_stars)

        log.info('Finished crossmatch for '+path.basename(red_dir))

    # Output the full crossmatch table:
    xmatch.save(params['file_path'])
    
    log.info('Crossmatch: complete')

    logs.close_log(log)

def match_dataset_with_primary_reference(primary_metadata, dataset_metadata,
                                        log,verbose=False):

    log.info('Matching all stars detected in the dataset reference frame with the primary reference catalog')

    matched_stars = match_utils.StarMatchIndex()

    idx = np.where(primary_metadata.star_catalog[1]['gaia_source_id'] != 'None')[0]
    jdx = np.where(reduction_metadata.star_catalog[1]['gaia_source_id'] != 'None')[0]

    log.info(str(len(idx))+' stars with Gaia identifications selected from the primary reference starlist')
    log.info(str(len(jdx))+' stars with Gaia identifications selected from dataset catalog')

    for star in primary_metadata.star_catalog[1][idx]:

        kdx = np.where(dataset_metadata.star_catalog[1]['gaia_source_id'][jdx] == star['gaia_source_id'])

        if len(kdx[0]) == 1:

            dataset_star = SkyCoord( dataset_metadata.star_catalog[1]['ra'][jdx[kdx[0]]],
                          dataset_metadata.star_catalog[1]['dec'][jdx[kdx[0]]],
                          frame='icrs', unit=(units.deg, units.deg) )

            field_star = SkyCoord( star['ra'], star['dec'],
                                    frame='icrs', unit=(units.deg, units.deg) )

            separation = dataset_star.separation(field_star)

            jj = jdx[kdx[0]][0]

            p = {'cat1_index': star['star_id'],
                 'cat1_ra': star['ra'],
                 'cat1_dec': star['dec'],
                 'cat1_x': star['x'][0],
                 'cat1_y': star['y'][0],
                 'cat2_index': jj+1,
                 'cat2_ra': dataset_metadata.star_catalog[1]['ra'][jj],
                 'cat2_dec': dataset_metadata.star_catalog[1]['dec'][jj],
                 'cat2_x': dataset_metadata.star_catalog[1]['x'][jj],
                 'cat2_y': dataset_metadata.star_catalog[1]['y'][jj],
                 'separation': separation[0]}

            matched_stars.add_match(p)

            if verbose:
                log.info(matched_stars.summarize_last(units='pixels'))

    return matched_stars

def get_args():
    params = {}

    if len(argv) < 5:
        params['primary_ref_dir'] = input('Please enter the path to the primary reference dataset: ')
        params['primary_ref_filter'] = input('Please enter the filter name used for the primary reference dataset: ')
        params['red_dir_list'] = [input('Please enter the path to the dataset to cross-match: ')]
        params['red_dataset_filters'] = [input('Please enter filter name used for the dataset: ')]
        params['file_path'] = input('Please enter the path to the crossmatch table: ')
    else:
        params['primary_ref_dir'] = argv[1]
        params['primary_ref_filter'] = argv[2]
        params['red_dir_list'] = [argv[2]]
        params['red_dataset_filters'] = [argv[4]]
        params['file_path'] = argv[5]

    params['log_dir'] = path.dirname(params['file_path'])

    return params


if __name__ == '__main__':
    params = get_args()
    build_crossmatch_table(params)