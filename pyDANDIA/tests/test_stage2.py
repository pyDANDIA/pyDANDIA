import numpy as np
import mock
import pytest
import os
from os import getcwd, path
from sys import path as systempath
import collections
from astropy.io import fits

cwd = getcwd()
systempath.append(path.join(cwd, '../'))

import stage2
import metadata

def test_moon_brightness_header():
    setup = mock.MagicMock()

    header_dummy = {'MOONFRAC':0.5,'MOONDIST':100}
    moon_state = stage2.moon_brightness_header(header_dummy)
    assert moon_state == 'gray'

def test_electrons_per_second_sinistro():
    setup = mock.MagicMock()
    mag = 15
    magzero_electrons = 1
    elpers = stage2.electrons_per_second_sinistro(mag, magzero_electrons)

    assert elpers == 1e-6

