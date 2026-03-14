import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import aperture_photometry, Background2D, DAOStarFinder

def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data

def perform_photometry(image, threshold=5.0):
    # Find stars in the image
    daofind = DAOStarFinder(fwhm=5, threshold=threshold)
    stars = daofind(image)
    
    # Compute background
    bkgd = Background2D(image, (50, 50), (100, 100))
    bkgd_subtracted = image - bkgd.background
    
    # Perform photometry
    phot_table = aperture_photometry(bkgd_subtracted, stars, apertures=5, method='pSF')
    
    return phot_table

def plot_ref_bkgd_hist(phot_table, image):
    # Plot histogram of reference background
    plt.figure(figsize=(10, 5))
    plt.hist(phot_table['aperture_sum'] - phot_table['background'], bins=50, color='blue', alpha=0.7)
    plt.title('Reference Background Histogram')
    plt.xlabel('Background Value')
    plt.ylabel('Frequency')
    plt.show()

def test_photometry():
    # Load test image
    test_image = load_fits_image('test_image.fits')
    
    # Perform photometry
    phot_table = perform_photometry(test_image)
    
    # Plot reference background histogram
    plot_ref_bkgd_hist(phot_table, test_image)
    
    # Check if the number of stars with good photometry is as expected
    if len(phot_table) < 10:
        print("Test failed: Not enough stars with good photometry.")
    else:
        print("Test passed: Adequate number of stars with good photometry.")

# Run test
test_photometry()