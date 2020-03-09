import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import match_utils
import numpy as np

def test_find_starlist_match_index():

    nstars = 200000

    matched_stars = match_utils.StarMatchIndex()
    for j in range(1,nstars+1,1):
        star = {'cat1_index': j,
                'cat1_ra': 260.0, 'cat1_dec': -19.0, 'cat1_x': j, 'cat1_y': j,
                'cat2_index': j,
                'cat2_ra': 260.0, 'cat2_dec': -19.0, 'cat2_x': j, 'cat2_y': j,
                'separation': 0.0}
        matched_stars.add_match(star)

    test_cat1_ids = np.arange(1,int(nstars/2),1, dtype='int')

    star_cat_ids = matched_stars.find_starlist_match_ids('cat2_index', test_cat1_ids)

    assert type(star_cat_ids) == type(np.array([]))
    assert star_cat_ids.all() == test_cat1_ids.all()

    test_cat1_ids = test_cat1_ids.tolist()
    test_cat1_ids.append(nstars+10)

    star_cat_ids = matched_stars.find_starlist_match_ids('cat2_index', test_cat1_ids)
    assert star_cat_ids[-1] == -1

if __name__ == '__main__':

    test_find_starlist_match_index()
