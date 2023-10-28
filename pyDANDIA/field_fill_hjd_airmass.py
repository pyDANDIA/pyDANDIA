from os import path
import argparse
from pyDANDIA import crossmatch
from pyDANDIA import time_utils
from astropy.io import fits

def fill_images_table(args):
    """
    The Images table of the field crossmatch file can have some gaps in the columns for HJS and airmass.
    This is the result of processing only images selected for quality through the pipeline.
    However, it leaves inconvenient gaps in the final table, which this function fills in

    Args:
        args: parser containing the path to the crossmatch file and the top-level data directory for the field

    Returns:
        Outputs an updated crossmatch file
    """

    # Load the crossmatch table
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(args.xmatch_file)

    # Search for gaps in the Image table columns for HJD and airmass
    for i in range(0,len(xmatch.images['filename']),1):
        if xmatch.images['hjd'][i] == 0.0:
            xmatch.images['hjd'][i] = calc_hjd(xmatch.images['datetime'][i],
                     xmatch.images['RA'][i],
                     xmatch.images['Dec'][i],
                     xmatch.images['exposure'][i])

        if xmatch.images['airmass'][i] == 0.0:
            hdr = fits.gethead(path.join(args.red_dir, xmatch.images['dataset_code'][i],
                                         'data', xmatch.images['filename'][i]))
            airmass = hdr['AIRMASS']
            xmatch.images['airmass'][i] = airmass

    # Output the updated crossmatch table
    xmatch.save(args.xmatch_file)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("red_dir", help="Path to top-level field data directory", type=str)
    parser.add_argument("xmatch_file", help="Path to crossmatch table", type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    fill_images_table(args)