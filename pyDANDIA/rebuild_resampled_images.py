from os import path, makedirs
from sys import argv
import glob
import numpy as np
from skimage import transform as tf
from astropy.io import fits
from pyDANDIA import logs
from pyDANDIA import image_handling
from pyDANDIA import stage4

def resample_images_from_warp_matrix(red_dir):
    """Function to use the warp matrix parameters from a fullframe image produced
    in a previous reduction to resample an existing image and produced an
    aligned image"""

    log = logs.start_stage_log(red_dir, 'resampled_images')

    resampled_dir = path.join(red_dir, 'resampled')
    data_dir = path.join(red_dir, 'data')

    resimage_list = glob.glob(path.join(resampled_dir, '*.fits'))

    for entry in resimage_list:
        matrix_file = path.join(entry, 'warp_matrice_image.npy')
        image_path = path.join(data_dir, path.basename(entry))

        (image_header, data_image, mask_image) = load_image_data(image_path,log)

        if path.isfile(matrix_file) and type(data_image) == type(np.zeros(1)):
            model = load_resample_model(matrix_file,image_path,log)

            resampled_image = stage4.warp_image(data_image,model)

            resampled_mask = tf.warp(mask_image, inverse_map=model,
                            output_shape=data_image.shape, order=1, mode='constant',
                            cval=1, clip=True, preserve_range=True)

            output_resampled_image(red_dir, path.basename(image_path),
                                    image_header, resampled_image, resampled_mask, log)

    logs.close_log(log)

def load_image_data(image_path,log):

    if path.isfile(image_path):
        image_structure = image_handling.determine_image_struture(image_path, log=log)

        data_image_hdu = fits.open(image_path, memmap=True)
        image_header = data_image_hdu[0].header
        data_image = np.copy(data_image_hdu[image_structure['sci']].data)

        if image_structure['pyDANDIA_pixel_mask'] != None:
            mask_image = np.array(data_image_hdu[image_structure['pyDANDIA_pixel_mask']].data, dtype=float)
        else:
            mask_image = np.zeros(data_image.shape)
    else:
        data_image = None
        mask_image = None

    return image_header, data_image, mask_image

def load_resample_model(matrix_file,image_path,log):
    warp_matrix = np.load(matrix_file)

    if warp_matrix.shape != (3,3):
        model = tf.PolynomialTransform(warp_matrix)
    else:
        model = tf.AffineTransform()
        model.params = warp_matrix

    log.info('Loaded warp matrix parameters for '+path.basename(image_path)+': '+repr(model.params))

    return model

def output_resampled_image(red_dir, image_name, image_header, shifted_data, shifted_mask, log):

    out_dir = path.join(red_dir, 'resampled', 'fullframe_images')
    if not path.isdir(out_dir):
        makedirs(out_dir)

    image_path = path.join(out_dir,image_name)
    if '.fits' not in image_path:
        image_path = image_path+'.fits'

    hdul = fits.HDUList([fits.PrimaryHDU(shifted_data, header=image_header),fits.ImageHDU(shifted_mask)])
    hdul.writeto(image_path, overwrite=True)

    log.info('Output resampled fullframe image '+path.basename(image_path))

if __name__ == '__main__':

    if len(argv) > 1:
        red_dir = argv[1]
    else:
        red_dir = input('Please enter the path to the reduction directory: ')
    resample_images_from_warp_matrix(red_dir)
