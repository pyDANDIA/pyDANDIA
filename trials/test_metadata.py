import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sys.path.append('../pyDANDIA/')
import metadata



meta=metadata.MetaData()

meta.create_metadata_file('./','test_metadata.fits')


mydico = meta.transform_2D_table_to_dictionary('data_architecture')
mydico = metadata.update_a_dictionnary(mydico,'hello',51)
meta.update_2D_table_with_dictionary('data_architecture', mydico)
meta.add_column_to_layer('data_architecture', 'atchoum', [0], 'float')
meta.save_a_layer_to_file('./','test_metadata.fits','data_architecture')


meta.add_row_to_layer('reduction_parameters', len(meta.reduction_parameters[1].keys())*[[0]])
mydico = meta.transform_2D_table_to_dictionary('reduction_parameters')
mydico = metadata.update_a_dictionnary(mydico,'hello', 51)
meta.update_2D_table_with_dictionary('reduction_parameters', mydico)

meta.save_a_layer_to_file('./','test_metadata.fits','reduction_parameters')

import pdb; pdb.set_trace()

