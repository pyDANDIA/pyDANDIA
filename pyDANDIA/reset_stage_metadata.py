import os
import sys
from pyDANDIA import metadata

def reset_red_status_for_stage(red_dir,stage_number):
    """Function to reset the reduction status table entries for a given stage.
    Note that the status for all images will be set to zero (process) except
    for entries that were flagged as -1 (do not process) by the PREVIOUS
    stage"""

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    stage_name = 'STAGE_'+str(stage_number)

    image_list = reduction_metadata.reduction_status[1]['IMAGES'].data
    image_status = reduction_metadata.reduction_status[1]['STAGE_'+str(stage_number-1)].data

    for i,image in enumerate(image_list):

        if '-1' not in image_status[i]:
            reduction_metadata.update_a_cell_to_layer('reduction_status', i, stage_name, '0')

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        red_dir = input('Please enter the path to the reduction directory: ')
        stage_number = input('Please enter the stage number to reset: ')
    else:
        red_dir = sys.argv[1]
        stage_number = sys.argv[2]

    stage_number = int(stage_number)

    reset_red_status_for_stage(red_dir,stage_number)
