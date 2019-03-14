###################################################################################
#                   INTERPOLATION OF BESSELL AND BRETT TABLES

##########################
# IMPORTED MODULES
from __future__ import division
from os import path
from sys import argv, exit
import numpy as np
from scipy import interpolate

###########################
# MAIN FUNCTION
def star_colour_conversion(input_colour, colour_error, star_class, input_index, req_colour_index):
    """Function to convert between stellar colour indices, by interpolating between 
    measurements provided in Tables 2 and 3 of 
    Bessell, M.S. and Brett, J.M., 1988, PASP, 100, 1134
    """
    
    # Loading the Table III colour conversion table for Giant stars
    (colour_table, colour_index, table_error) = load_colour_data(star_class)
    
    idx = colour_index.index(input_index)
    jdx = colour_index.index(req_colour_index)
    
    # Interpolate between the two colour indices as functions 
    # of spectral type.
    in_colour_data = colour_table[:,idx]
    out_colour_data = colour_table[:,jdx]

    # Sanity check: Interpolation is only valid if input colour is
    # within the range of the data to be interpreted.  If not, halt:
    if input_colour > in_colour_data.max() or input_colour < in_colour_data.min():
        print('Stellar colour is outside the available data range (' + \
        str(in_colour_data.min()) + ' - ' + str(in_colour_data.max()) + 'mag)')
        exit()
    
    spline = interpolate.InterpolatedUnivariateSpline(in_colour_data,out_colour_data)
    
    req_star_colour = spline(input_colour)
    
    # Since this is a direct conversion, the precision of the output
    # colour is determined by the precision of the data used for the 
    # interpolation.  Bessell & Brett quote this as 0.04mag
    #req_colour_error = np.sqrt( colour_error*colour_error + table_error*table_error )
    req_colour_error = table_error
    
    print(input_index + ' = ' + str(input_colour) + '+/-' + str(colour_error) + \
            ' -> ' + req_colour_index + ' = ' + \
	    str(req_star_colour) + '+/-' + str(req_colour_error) + 'mag')
    
    return req_star_colour, req_colour_error
    
############################
# DATA TABLES
def load_colour_data(star_class):
    '''Function to load the star colour conversion data tables for a particular
    spectral class of star.  
    So far, only class III (giant) stars are supported.
    '''
    
    index_list = [ 'V-I', 'V-K', 'J-H', 'H-K', 'J-K', 'K-L', 'K-Lprime', 'K-M' ]
    
    # Note: 0.00001 added to (V-I) entry for G8 star in order to ensure that 
    # the numerical sequence is increasing, a requirement for a spline fit.
    # The original value was 0.94, the same as that for a G5 star.  
    if star_class == 'III':
        data_list = [ \
        [ 0.81, 1.75, 0.37, 0.065, 0.45, 0.04, 0.05, 0.0 ], # G0 \
	[ 0.91, 2.05, 0.47, 0.08, 0.55, 0.05, 0.06, -0.01 ], # G4 \ 
	[ 0.94, 2.15, 0.50, 0.085, 0.58, 0.06, 0.07, -0.02 ], # G5 \
	[ 0.94001, 2.16, 0.50, 0.085, 0.58, 0.06, 0.07, -0.02 ], # G8 \
	[ 1.00, 2.31, 0.54, 0.095, 0.63, 0.07, 0.08, -0.03 ], # K0 \
	[ 1.08, 2.50, 0.58, 0.10, 0.68, 0.08, 0.09, -0.04 ], # K1 \ 
	[ 1.17, 2.70, 0.63, 0.115, 0.74, 0.09, 0.10, -0.05 ], # K2 \
	[ 1.36, 3.00, 0.68, 0.14, 0.82, 0.10, 0.12, -0.06 ], # K3 \ 
	[ 1.50, 3.26, 0.73, 0.15, 0.88, 0.11, 0.14, -0.07 ], # K4 \ 
	[ 1.63, 3.60, 0.79, 0.165, 0.95, 0.12, 0.16, -0.08 ], # K5 \
	[ 1.78, 3.85, 0.83, 0.19, 1.01, 0.12, 0.17, -0.09 ], # M0 \ 
	[ 1.90, 4.05, 0.85, 0.205, 1.05, 0.13, 0.17, -0.10 ], # M1 \
	[ 2.05, 4.30, 0.87, 0.215, 1.08, 0.15, 0.19, -0.12 ], # M2 \
	[ 2.25, 4.64, 0.90, 0.235, 1.13, 0.17, 0.20, -0.13 ], # M3 \
	[ 2.55, 5.10, 0.93, 0.245, 1.17, 0.18, 0.21, -0.14 ], # M4 \
	[ 3.05, 5.96, 0.95, 0.285, 1.23, 0.20, 0.22, -0.15 ], # M5 \
	]
    
    else:
        data_list = [ \
	[ -0.15, -0.35, -0.05, -0.035, -0.09, -0.03, -0.04, -0.05 ], # B8 \
	[  0.00,  0.00,  0.00,  0.00,   0.00,  0.00,  0.00,  0.00 ], # A0 \
	[  0.06,  0.14, 0.02,  0.005,   0.02,  0.01,  0.01,  0.01 ], # A2 \
	[  0.27,  0.38, 0.06,  0.015,   0.08,  0.02,  0.02,  0.03 ], # A5 \
	[  0.24,  0.50, 0.09,  0.025,   0.11,  0.03,  0.03,  0.03 ], # A7 \
	[  0.33,  0.70, 0.13,  0.03,    0.16,  0.03,  0.03,  0.03 ], # F0 \
	[  0.40,  0.82, 0.165, 0.035,   0.19,  0.03,  0.03,  0.03 ], # F2 \
	[  0.53,  1.10, 0.23,  0.04,    0.27,  0.04,  0.04,  0.02 ], # F5 \
	[  0.62,  1.32, 0.285, 0.045,   0.34,  0.04,  0.04,  0.02 ], # F7 \
	[  0.66,  1.41, 0.305, 0.05,    0.36,  0.05,  0.05,  0.01 ], # G0 \
	[  0.68,  1.46, 0.32,  0.052,   0.37,  0.05,  0.05,  0.01 ], # G2 \
	[  0.71,  1.53, 0.33,  0.055,   0.385, 0.05,  0.05,  0.01 ], # G4 \
	[  0.75,  1.64, 0.37,  0.06,    0.43,  0.05,  0.05,  0.00 ], # G6 \
	[  0.88,  1.96, 0.45,  0.075,   0.53,  0.06,  0.06, -0.01 ], # K0 \
	[  0.98,  2.22, 0.50,  0.09,    0.59,  0.07,  0.07, -0.02 ], # K2 \
	[  1.15,  2.63, 0.58,  0.105,   0.68,  0.09,  0.10, -0.04 ], # K4 \
	[  1.22,  2.85, 0.61,  0.11,    0.72,  0.10,  0.11, 99.99 ], # K5 \
	[  1.45,  3.16, 0.66,  0.13,    0.79,  0.11,  0.13, 99.99 ], # K7 \
	[  1.80,  3.65, 0.695, 0.165,   0.86,  0.14,  0.17, 99.99 ], # M0 \
	[  1.96,  3.87, 0.68,  0.20,    0.87,  0.15,  0.21, 99.99 ], # M1 \
	[  2.14,  4.11, 0.665, 0.21,    0.87,  0.16,  0.23, 99.99 ], # M2 \
	[  2.47,  4.65, 0.62,  0.25,    0.87,  0.20,  0.32, 99.99 ], # M3 \
	[  2.86,  5.26, 0.60,  0.275,   0.88,  0.23,  0.37, 99.99 ], # M4 \
	[  3.39,  6.12, 0.62,  0.32,    0.94,  0.29,  0.42, 99.99 ], # M5 \
	[  4.18,  7.30, 0.66,  0.37,    1.03,  0.36,  0.48, 99.99 ], # M6 \
	]
    
    uncertainty = 0.04
    
    return np.array( data_list ), index_list, uncertainty


#########################################################
if __name__ == '__main__':
    
    if len(argv) == 6:
        input_colour = float( argv[1] )
        colour_error = float( argv[2] )
        star_class = argv[3]
        input_index = argv[4]
        req_colour_index = argv[5]
    else:
        input_colour = float( input('Please input the star colour: ') )
        colour_error = float( input('Please input the error on the star colour: ') )
        star_class = input('Please input the spectral class [III, IV]: ')
        input_index = input('Input colour index [e.g. V-I]: ')
        req_colour_index = input('Input required colour index: ')
    	
    star_colour_conversion(input_colour, colour_error, star_class, input_index, req_colour_index)
    
