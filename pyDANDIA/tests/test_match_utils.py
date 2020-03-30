import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import match_utils
import numpy as np
import logs

def test_add_match():

    nstars = 10

    matched_stars = match_utils.StarMatchIndex()
    for j in range(1,nstars+1,1):
        star = {'cat1_index': j,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
                'cat2_index': j,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
                'separation': 0.1}
        matched_stars.add_match(star)

    assert matched_stars.n_match == nstars

    new_star_id = 1
    new_separation = 0.01
    new_star = {'cat1_index': new_star_id,
            'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
            'cat2_index': new_star_id,
            'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
            'separation': new_separation}
    matched_stars.add_match(new_star)

    assert matched_stars.n_match == nstars
    idx = matched_stars.cat1_index.index(new_star_id)
    assert matched_stars.separation[idx] == new_separation

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

def build_test_matched_stars_index(nstars, log=None):

    if log!=None:
        log.info('Building test matched_stars index:')

    matched_stars = match_utils.StarMatchIndex()

    for j in range(1,nstars+1,1):
        star = {'cat1_index': j,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
                'cat2_index': j,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
                'separation': 0.5}
        matched_stars.add_match(star)

        if log!=None:
            log.info(matched_stars.summarize_last())

    return matched_stars

def test_filter_duplicates():

    log = logs.start_stage_log( cwd, 'test_match_utils' )

    nstars = 10
    matched_stars = build_test_matched_stars_index(nstars, log)

    test_star1 = {'cat1_index': 1,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': 25.0, 'cat1_y': 25.0,
                'cat2_index': nstars+10,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': 25.0, 'cat2_y': 25.0,
                'separation': 0.01}
    test_star2 = {'cat1_index': nstars+10,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': 35.0, 'cat1_y': 35.0,
                'cat2_index': 1,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': 35.0, 'cat2_y': 35.0,
                'separation': 0.01}
    test_star3 = {'cat1_index': 1,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': 45.0, 'cat1_y': 45.0,
                'cat2_index': nstars+20,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': 45.0, 'cat2_y': 45.0,
                'separation': 1.5}
    test_star4 = {'cat1_index': 140,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': 45.0, 'cat1_y': 45.0,
                'cat2_index': 1,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': 45.0, 'cat2_y': 45.0,
                'separation': 1.5}

    log.info('Testing addition of a star duplicated in cat1_index with smaller separation:')
    log.info(repr(test_star1))

    matched_stars.add_match(test_star1, log=log, verbose=True)

    assert matched_stars.n_match == nstars
    jdx = matched_stars.cat1_index.index(test_star1['cat1_index'])
    assert matched_stars.separation[jdx] == test_star1['separation']

    log.info('Re-building clean match index')
    matched_stars = build_test_matched_stars_index(nstars, log)

    log.info('Testing addition of a star duplicated in cat2_index with smaller separation')
    log.info(repr(test_star2))

    matched_stars.add_match(test_star2, log=log, verbose=True)

    assert matched_stars.n_match == nstars
    jdx = matched_stars.cat2_index.index(test_star2['cat2_index'])
    assert matched_stars.separation[jdx] == test_star2['separation']

    log.info('Re-building clean match index')
    matched_stars = build_test_matched_stars_index(nstars, log)

    log.info('Testing addition of a star duplicated in cat1_index with larger separation')
    log.info(repr(test_star3))

    matched_stars.add_match(test_star3, log=log, verbose=True)
    jdx = np.where(np.array(matched_stars.cat2_index) == test_star3['cat2_index'])[0]
    assert len(jdx) == 0

    log.info('Re-building clean match index')
    matched_stars = build_test_matched_stars_index(nstars, log)

    log.info('Testing addition of a star duplicated in cat2_index with larger separation')
    log.info(repr(test_star4))

    matched_stars.add_match(test_star4, log=log, verbose=True)
    jdx = np.where(np.array(matched_stars.cat1_index) == test_star4['cat1_index'])[0]
    assert len(jdx) == 0

    logs.close_log(log)

if __name__ == '__main__':

    #test_find_starlist_match_index()
    #test_add_match()
    test_filter_duplicates()
