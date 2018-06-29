def generate_outlier_mask(reference_image, new_images, reduction_metadata):
    master_mask=np.zeros(np.shape(reference_image))
    for new_image in new_images:
        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'],-reduction_metadata.images_stats[1][row_index]['SHIFT_Y'] 
        #smooth reference to match data
        smoothing = 0
        if reduction_metadata.images_stats[1][row_index]['FWHM_X']>ref_fwhm_x:
            sigma_x = reduction_metadata.images_stats[1][row_index]['FWHM_X']/(2.*(2.*np.log(2.))**0.5)
            smoothing = (sigma_x**2-ref_sigma_x**2)**0.5       
        if reduction_metadata.images_stats[1][row_index]['FWHM_Y']>ref_fwhm_y:
            sigma_y = reduction_metadata.images_stats[1][row_index]['FWHM_Y']/(2.*(2.*np.log(2.))**0.5)
            smoothing_y = (ref_sigma_y**2-sigma_y**2)**0.5
            if smoothing_y>smoothing:
                smoothing = smoothing_y
        if smoothing > 0.1 and (1./0.95 > reduction_metadata.images_stats[1][row_index]['FWHM_X']/reduction_metadata.images_stats[1][row_index]['FWHM_Y'] > 0.95):
            model=reference_image = gaussian_filter(reference_image, sigma=smoothing)
            data_image, data_image_unmasked = open_data_image(setup, data_image_directory, new_image, bright_reference_mask, kernel_size, max_adu, xshift = x_shift, yshift = y_shift, sigma_smooth = 0, central_crop = maxshift)
            positive_mask = (model> 0)
            pscale = np.polyfit(model[positive_mask].ravel(),data_image[positive_mask].ravel(),1)
            test_difference = model*pscale[0]-data_image+pscale[1]             
            outlier = (np.abs(test_difference) > 8.* np.std(test_difference))
            master_mask[outlier]=master_mask[outlier] + 1
    outlier_mask = master_mask > 0
    mask_propagate = np.zeros(np.shape(reference_image))
    mask_propagate[outlier_mask] = 1.
    kernel_mask = mask_kernel(kernel_size)
    masked = convolve2d(mask_propagate, kernel_mask, mode='same')
    difference_image_tst = fits.PrimaryHDU(masked)
    difference_image_tst.writeto(os.path.join(diffim_directory_path,'mastermask.fits'),overwrite = True)
    bright_reference_mask = masked >0 
    return bright_reference_mask   
