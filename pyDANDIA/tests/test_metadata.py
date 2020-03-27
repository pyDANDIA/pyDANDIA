import numpy as np
import mock
import pytest
import os
from os import getcwd, path
from sys import path as systempath
import collections

cwd = getcwd()
systempath.append(path.join(cwd, '../'))

import metadata
from pyDANDIA import match_utils
from skimage.transform import AffineTransform

def test_create_metadata_file():
    metad = metadata.MetaData()

    metad.create_metadata_file('./', 'dummy_metadata.fits')

    assert os.path.isfile('./dummy_metadata.fits') == True


def test_load_metadata_from_file():
    metad = metadata.MetaData()
    metad.load_all_metadata('./', 'dummy_metadata.fits')

    assert metad.stamps == [None, None]
    assert metad.reduction_parameters == [None, None]
    assert metad.headers_summary == [None, None]

    assert metad.data_inventory[1].keys() == ['IMAGES', 'STAGE_0', 'STAGE_1', 'STAGE_2', 'STAGE_3',
                                              'STAGE_4', 'STAGE_5', 'STAGE_6', 'STAGE_7']
    assert len(metad.data_inventory[1]) == 0

    assert metad.data_architecture[1].keys() == ['METADATA_NAME', 'OUTPUT_DIRECTORY']

    assert metad.data_architecture[1]['METADATA_NAME'] == 'dummy_metadata.fits'
    assert metad.data_architecture[1]['OUTPUT_DIRECTORY'] == './'

    assert len(metad.data_architecture[1]) == 1


def test_create_a_new_layer():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0', '1'], ['59',
                                           '41']])

    new_layer = metad.dummy_layer

    assert new_layer[1].keys() == ['OHOHOH', 'IHIHIH']

    assert new_layer[1]['OHOHOH'].dtype == 'S150'
    assert new_layer[1]['IHIHIH'].dtype == 'S10'

    assert new_layer[1]['OHOHOH'].unit == 'km/s'
    assert new_layer[1]['IHIHIH'].unit == 'h/(2pi)'

    assert new_layer[1]['OHOHOH'][0] == '0'
    assert new_layer[1]['OHOHOH'][1] == '1'
    assert new_layer[1]['IHIHIH'][0] == '59'
    assert new_layer[1]['IHIHIH'][1] == '41'


def test_load_a_layer_from_file():
    metad = metadata.MetaData()
    metad.load_a_layer_from_file('./', 'dummy_metadata.fits', 'data_inventory')

    assert metad.data_inventory[1].keys() == ['IMAGES', 'STAGE_0', 'STAGE_1', 'STAGE_2', 'STAGE_3',
                                              'STAGE_4', 'STAGE_5', 'STAGE_6', 'STAGE_7']

    assert len(metad.data_inventory[1]) == 0


def test_save_updated_metadata():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0', '1'], ['59',
                                           '41']])

    metad.save_updated_metadata('./', 'dummy_metadata.fits')

    metad2 = metadata.MetaData()
    metad2.load_all_metadata('./', 'dummy_metadata.fits')

    assert metad2.dummy_layer[1].keys() == ['OHOHOH', 'IHIHIH']


def test_save_a_layer_to_file():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0', '1'], ['59',
                                           '41']])
    metad.save_a_layer_to_file('./', 'dummy_metadata.fits', 'dummy_layer')

    metad2 = metadata.MetaData()
    metad2.load_all_metadata('./', 'dummy_metadata.fits')

    assert metad2.dummy_layer[1].keys() == ['OHOHOH', 'IHIHIH']


def test_transform_2D_table_to_dictionary():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0'], ['59']])

    dico = metad.transform_2D_table_to_dictionary('dummy_layer')

    assert len(dico._fields) == 2
    assert getattr(dico,'OHOHOH') == '0'
    assert getattr(dico,'IHIHIH') == '59'


def test_update_2D_table_with_dictionary():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0'], ['59']])

    dictionary = collections.namedtuple('dummy_dictionary', ['OHOHOH', 'IHIHIH'])
    setattr(dictionary, 'OHOHOH', 'monalisa')
    setattr(dictionary, 'IHIHIH', 'batistuta')

    metad.update_2D_table_with_dictionary('dummy_layer', dictionary)

    assert metad.dummy_layer[1]['OHOHOH'] == 'monalisa'
    assert metad.dummy_layer[1]['IHIHIH'] == 'batistuta'


def test_add_row_to_layer():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0'], ['59']])


    new_row = ['purple','orange']
    metad.add_row_to_layer('dummy_layer',new_row)

    assert metad.dummy_layer[1]['OHOHOH'][0] == '0'
    assert metad.dummy_layer[1]['OHOHOH'][1] == 'purple'

    assert metad.dummy_layer[1]['IHIHIH'][0] == '59'
    assert metad.dummy_layer[1]['IHIHIH'][1] == 'orange'

def test_add_column_to_layer():

    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0'], ['59']])


    new_column_name = 'LOL'
    new_column_data = [42]
    new_column_format = 'float64'
    new_column_unit = 'N/c'

    metad.add_column_to_layer('dummy_layer', new_column_name, new_column_data, new_column_format, new_column_unit)

    assert metad.dummy_layer[1]['LOL'] == 42
    assert metad.dummy_layer[1]['LOL'].dtype == 'float64'
    assert metad.dummy_layer[1]['LOL'].unit == 'N/c'


def test_update_row_to_layer():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH', 'IHIHIH'], ['S150', 'S10'], ['km/s', 'h/(2pi)']],
                             [['0'], ['59']])

    metad.update_row_to_layer('dummy_layer', 0, ['89','-98' ])


    assert metad.dummy_layer[1]['OHOHOH'][0] == '89'
    assert metad.dummy_layer[1]['IHIHIH'][0] == '-98'

def test_update_column_to_layer():
    metad = metadata.MetaData()

    metad.create_a_new_layer('dummy_layer', [['OHOHOH'], ['S150'], ['km/s']],
                             [['0','59']])

    metad.update_row_to_layer('dummy_layer', 'OHOHOH', ['89', '-98'])
    assert metad.dummy_layer[1]['OHOHOH'][0] == '89'
    assert metad.dummy_layer[1]['OHOHOH'][1] == '-98'

def build_test_match_index():

    matched_stars = match_utils.StarMatchIndex()

    star_ids = list(range(0,10,1))

    matched_stars.cat1_index = star_ids
    matched_stars.cat1_ra = [290.0]*len(star_ids)
    matched_stars.cat1_dec = [-19.0]*len(star_ids)
    matched_stars.cat1_x = [1800.0]*len(star_ids)
    matched_stars.cat1_y = [2000.0]*len(star_ids)
    matched_stars.cat2_index = list(np.array(star_ids) + 1)
    matched_stars.cat2_ra = [290.0]*len(star_ids)
    matched_stars.cat2_dec = [-19.0]*len(star_ids)
    matched_stars.cat2_x = [1800.0]*len(star_ids)
    matched_stars.cat2_y = [2000.0]*len(star_ids)
    matched_stars.separation = [ 0.01 ]*len(star_ids)
    matched_stars.n_match = len(star_ids)

    return matched_stars

def test_create_matched_stars_layer():

    metad = metadata.MetaData()

    matched_stars = build_test_match_index()

    metad.create_matched_stars_layer(matched_stars)

    metad.create_metadata_file('.', 'dummy_metadata.fits')

    metad.save_a_layer_to_file('./', 'dummy_metadata.fits',
                                          'matched_stars', log=None)

    assert (metad.matched_stars[1]['field_star_id'] == np.array(matched_stars.cat1_index)).all()

def test_load_matched_stars():

    test_create_matched_stars_layer()
    test_matched = build_test_match_index()

    metad = metadata.MetaData()
    metad.load_all_metadata('.', 'dummy_metadata.fits')

    matched_stars = metad.load_matched_stars()

    assert type(matched_stars) == type(test_matched)
    assert len(matched_stars.cat1_index) > 0
    for col in ['cat1_index', 'cat1_ra', 'cat1_dec', 'cat1_x', 'cat2_y',
                'cat2_index', 'cat2_ra', 'cat2_dec', 'cat2_x', 'cat2_y', 'separation']:
        print(col, getattr(matched_stars,col), getattr(test_matched,col))
        assert ( np.array(getattr(matched_stars,col)) == np.array(getattr(test_matched,col)) ).all()

def build_tranform():

    matrix = np.array( [ [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [5.0, 6.0, 7.0] ] )
    transform = AffineTransform(matrix = matrix)

    return transform

def test_create_transform_layer():

    metad = metadata.MetaData()

    transform = build_tranform()

    metad.create_transform_layer(transform)

    metad.create_metadata_file('.', 'dummy_metadata.fits')

    metad.save_a_layer_to_file('./', 'dummy_metadata.fits',
                                          'transformation', log=None)

    assert (metad.transformation[1]['matrix_column0'] == np.array(transform.params[:,0])).all()
    assert (metad.transformation[1]['matrix_column1'] == np.array(transform.params[:,1])).all()
    assert (metad.transformation[1]['matrix_column2'] == np.array(transform.params[:,2])).all()

def test_load_field_dataset_transform():

    test_create_transform_layer()
    test_transform = build_tranform()

    metad = metadata.MetaData()
    metad.load_all_metadata('.', 'dummy_metadata.fits')

    transform = metad.load_field_dataset_transform()

    assert type(transform) == type(test_transform)
    assert (transform.params == test_transform.params).all()

os.remove('./dummy_metadata.fits')


if __name__ == '__main__':
    #test_create_matched_stars_layer()
    #test_load_matched_stars()
    #test_create_transform_layer()
    test_load_field_dataset_transform()
