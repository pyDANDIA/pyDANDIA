import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sys.path.append('../pyDANDIA/')
import metadata



meta=metadata.MetaData()
meta.create_metadata_file('./','test_metadata.fits')
meta.load_a_layer_from_file('./','test_metadata.fits','data_architecture')
mydico = meta.transform_2D_table_to_dictionary('data_architecture')

mydico = metadata.update_a_dictionnary(mydico,'hello',51)
meta.update_2D_table_with_dictionary('data_architecture', mydico)
meta.add_column_to_layer('data_architecture', 'atchoum', np.arange(0,len(mydico._fields)), 'float')
meta.save_a_layer_to_file('./','test_metadata.fits','data_architecture')

