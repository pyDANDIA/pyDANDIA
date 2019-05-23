# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:04:36 2018

@author: Original code and reference from ytsapras, 
integrated by rstreet
"""

# Synthetic SDSS/2MASS photometry of Pickles (1998, Cat. J/PASP/110/863) for 
# solar metallicity standards (I've attached it in txt format):
#http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/134/2398

def get_spectral_class_data():
    """Function to read data on star colours and spectral type from 
    Vizier table in file spectral_class.txt
    """
    
    file_lines = open('spectral_class.txt','r').readlines()
    
    spectral_type = []
    luminosity_class = []
    gr_colour = []
    ri_colour = []
    
    for line in file_lines[39:]:
        spectral_type.append(line.split()[0])
        luminosity_class.append(line.split()[1])
        gr_colour.append(float(line.split()[2]))
        ri_colour.append(float(line.split()[3]))
    
    return spectral_type, luminosity_class, gr_colour, ri_colour
    