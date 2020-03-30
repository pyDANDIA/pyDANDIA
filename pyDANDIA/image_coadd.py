import os
import sys
from astropy.io import fits
import numpy as np
from pyDANDIA import  pipeline_setup
from pyDANDIA import  logs

VERSION = 'pyDANDIA_image_coadd_v1.0'

def run_coadd(setup):
    """Driver function for pyDANDIA image co-addition tool
    """

    log = logs.start_stage_log( setup.red_dir, 'image_coadd', version=VERSION )

    input_file = os.path.join( setup.red_dir, 'resampled', 'coadd_images.txt')

    if not os.path.isfile(input_file):
        raise IOError('Cannot find input image list to build coadded image from.  Looking for '+input_file)

    image_list = open(input_file).readlines()

    # Note the duplication of the image name here is intentional, to accommodate
    # the sub-directory for resampled stamp images.
    base_image = fits.getdata(os.path.join(setup.red_dir, 'resampled', image_list[0], image_list[0]))
    weight = np.zeros(base_image.shape)

    for image_name in image_list[1:]:
        image = fits.getdata(os.path.join(setup.red_dir, 'resampled', image_name, image_name))

        base_image += image
        weight += 1.0 / np.abs(image)**0.5

    coadded_image = base_image / weight

    hdu = fits.PrimaryHDU(coadded_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(setup.red_dir,
                                 'resampled', 'coadded_image.fits'),
                                 overwrite=True)

    status = 'OK'
    report = 'Completed successfully'

    log.info('Stage 3: '+report)
    logs.close_log(log)

    return status, report
