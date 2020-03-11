import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import match_utils
import numpy as np
import logs

def test_find_starlist_match_index():

    log = logs.start_stage_log( cwd, 'test_match_utils' )

    nstars = 200000

    matched_stars = match_utils.StarMatchIndex()
    for j in range(1,nstars+1,1):
        star = {'cat1_index': j,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
                'cat2_index': j,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
                'separation': 0.0}
        matched_stars.add_match(star)

    log.info('Test with identical indices')
    test_cat1_ids = np.arange(1,int(nstars/2),1, dtype='int')

    (star_ids,star_cat_ids) = matched_stars.find_starlist_match_ids('cat2_index', test_cat1_ids, log)

    assert type(star_cat_ids) == type(np.array([]))
    assert star_cat_ids.all() == test_cat1_ids.all()
    assert len(star_ids) == len(test_cat1_ids)

    log.info('Test with star added to search index')
    test_cat1_ids = test_cat1_ids.tolist()
    test_cat1_ids.append(nstars+10)
    test_cat1_ids = np.array(test_cat1_ids)

    (star_ids,star_cat_ids) = matched_stars.find_starlist_match_ids('cat2_index', test_cat1_ids, log)
    assert star_cat_ids[-1] == -1

    log.info('Test where starlist is longer than entries in array')
    test_cat1_ids = np.arange(1,nstars+200,1, dtype='int')

    (star_ids,star_cat_ids) = matched_stars.find_starlist_match_ids('cat2_index', test_cat1_ids, log)

    assert len(star_cat_ids) == len(test_cat1_ids)
    assert len(np.where(star_cat_ids > 0)[0]) < len(test_cat1_ids)

    log.info('Test for double-match scenario with index expansion')
    star = {'cat1_index': matched_stars.n_match+1,
            'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
            'cat2_index': matched_stars.n_match-2,
            'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
            'separation': 0.0}
    matched_stars.add_match(star)
    log.info('Added duplicate star ID='+str(star['cat1_index'])+' matched with '+str(star['cat2_index']))

    test_cat1_ids = np.arange(1,nstars+200,1, dtype='int')

    (star_ids,star_cat_ids) = matched_stars.find_starlist_match_ids('cat2_index', test_cat1_ids,
                                                        log, verbose=True, expand_star_ids=True)

    assert len(star_ids) == len(star_cat_ids)
    (unique_star_ids, unique_star_index) = np.unique(star_ids,return_index=True)
    non_unique_star_ids = np.delete(star_ids, unique_star_index)
    assert len(non_unique_star_ids) == 1
    assert non_unique_star_ids[0] == star['cat2_index']


    log.info('Test for double-match scenario without index expansion')
    log.info('Duplicated star  has ID='+str(star['cat1_index'])+' matched with '+str(star['cat2_index']))
    (star_ids,star_cat_ids) = matched_stars.find_starlist_match_ids('cat2_index', test_cat1_ids,
                                                        log, verbose=True)

    assert len(star_ids) == len(star_cat_ids)
    assert len(star_ids == len(test_cat1_ids))
    assert star_ids.all() == test_cat1_ids.all()
    (unique_star_ids, unique_star_index) = np.unique(star_ids,return_index=True)
    non_unique_star_ids = np.delete(star_ids, unique_star_index)
    assert len(non_unique_star_ids) == 0

    logs.close_log(log)

if __name__ == '__main__':

    test_find_starlist_match_index()
