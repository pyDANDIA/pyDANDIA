from pyDANDIA import aperture_photometry
from pyDANDIA import psf
import numpy as np
from astropy.io import fits
from os import path

TEST_DIR = './data'

def generate_test_image_and_catalog(naxis1=500, naxis2=500, nstars=100, output_image=False):

    # Generate 2D image data with a uniform sky background, ensuring that all background
    # values are positive
    median_bkgd = 100.0
    sat_value = 150000.0
    data = np.random.normal(loc=median_bkgd, scale=50.0, size=(naxis2, naxis1))
    idx = np.where(data <= 0.0)
    data[idx] = abs(data[idx])

    # Generate a set of star pixel positions (x, y) and fluxes
    star_catalog = np.zeros((nstars, 3))
    star_catalog[:,0] = np.random.uniform(1, naxis2-1, nstars)
    star_catalog[:,1] = np.random.uniform(1, naxis1-1, nstars)
    star_catalog[:,2] = median_bkgd + np.random.uniform(10.0, sat_value, nstars)


    # Add stellar PSFs to the generated image for all stars in the catalog
    psf_width_x = 40    # pixels
    x_cen = psf_width_x / 2.0
    psf_height_y = 40   # pixels
    y_cen = psf_height_y / 2.0
    for j in range(0,nstars,1):

        # Generate a PSF model for this star
        psf_model = psf.Moffat2D()
        psf_params = [star_catalog[j,2], y_cen, x_cen, 0.4, 1.5]
        psf_model.update_psf_parameters(psf_params)

        # Calculating the stamp dimensions of the PSF, generate pixel data for the PSF
        corners = psf.calc_stamp_corners(star_catalog[j,0], star_catalog[j,1], psf_width_x, psf_height_y,
                                     data.shape[1], data.shape[0],
                                     over_edge=True)
        dxc = corners[1] - corners[0]
        dyc = corners[3] - corners[2]
        Y_data, X_data = np.indices([int(dyc), int(dxc)])
        psf_image = psf_model.psf_model(Y_data, X_data, psf_model.get_parameters())

        # Add the PSF to the image
        data[corners[2]:corners[3], corners[0]:corners[1]] += psf_image

    # Output image as a diagnostic
    if output_image:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(path.join(TEST_DIR, 'sim_image.fits'), overwrite=True)

    return data, star_catalog

def test_ap_phot_image():

    # Produce a simulated image to test with
    (image, star_catalog) = generate_test_image_and_catalog(output_image=True)

    # Perform aperture photometry on this image:
    phot_table = aperture_photometry.ap_phot_image(image, star_catalog[:,[0,1]], radius=3)

    assert(len(phot_table) == len(star_catalog))


if __name__ == '__main__':
    test_ap_phot_image()
