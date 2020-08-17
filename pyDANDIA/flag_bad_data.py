from os import path
from sys import argv
from pyDANDIA import metadata
import numpy as np

def set_image_reduction_status(params):

    params['metadata'] = path.join(params['red_dir'],'pyDANDIA_metadata.fits')
    if not path.isfile(params['metadata']):
        raise IOError('Cannot find metadata file at '+params['metadata'])

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(params['red_dir'], 'pyDANDIA_metadata.fits')

    stage_name = 'STAGE_'+str(params['stage_number'])

    image_list = reduction_metadata.reduction_status[1]['IMAGES'].data

    iimage = np.where(image_list == params['image_name'])[0][0]

    for j in range(params['stage_number'],8,1):
        stage_name = 'STAGE_'+str(j)
        reduction_metadata.update_a_cell_to_layer('reduction_status', iimage, stage_name, '-1')

    reduction_metadata.save_updated_metadata(params['red_dir'],'pyDANDIA_metadata.fits')

if __name__ == '__main__':

    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['image_name'] = input('Please enter the name of the image: ')
        params['stage_number'] = int(input('Please enter the number of the stage (this and subsequent stages will be flagged): '))

    else:
        params['red_dir'] = argv[1]
        params['image_name'] = argv[2]
        params['stage_number'] = int(argv[3])

    set_image_reduction_status(params)
