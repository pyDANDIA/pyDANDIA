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

os.remove('./dummy_metadata.fits')