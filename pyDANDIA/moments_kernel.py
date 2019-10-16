import numpy as np

def kernel_moment(kernel_array):
    '''
    Determine kernel moments 
    :param array_like kernel_data: the convolution kernel
    :return: expected x and y values and sqrt(variance)
    :rtype: array_like
    '''
    norm = np.sum(kernel) 
    mu_x = np.sum(np.arange(0,len(kernel)) * (np.sum(kernel,axis = 0)) / norm)  
    sigma_x = (np.sum((mu_x - np.arange(0,len(kernel)))**2 * (np.sum(kernel,axis = 0)) / norm))**0.5    
    mu_y = np.sum(np.arange(0,len(kernel))*(np.sum(kernel.transpose(),axis = 0)) / norm)  
    sigma_y = (np.sum((mu_y - np.arange(0,len(kernel)))**2 * (np.sum(kernel.transpose(),axis = 0)) / norm))**0.5    
    sigma_rms = (0.5*(sigma_x**2 + sigma_y**2))**0.5
    return mu_x, mu_y, sigma_y, sigma_y, sigma_rms
