import numpy as np
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
from pyDANDIA import photometry_classes
from pyDANDIA import logs
from astropy import table
import json

def test_output_json():

    star = photometry_classes.Star()

    for key in star.parameter_list:
        setattr(star,key,1.0)

    file_path = 'data/test_phot_classes.json'

    star.output_json(file_path)

    par_dict = json.loads(open(file_path,'r').read())

    for key, value in par_dict.items():
        assert value == getattr(star,key)

if __name__ == '__main__':

    test_output_json()
