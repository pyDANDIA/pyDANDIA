# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:50:05 2018

@author: rstreet
"""

from os import path
import numpy as np
from astropy.io import fits
import glob

class BadPixelMask:
    """Class describing bad pixel masks for imaging data"""
    
    def __init__(self):
        
        self.camera = None
        self.dateobs = None
        self.file_path = None
        self.data = np.zeros([1])
    
    def read_mask(self,file_path):
        """Function to read a camera mask from a pre-existing file
        
        Input:
            :param str file_path: Path to the file to be read
        """
        
        if path.isfile(file_path):
            
            self.file_path = file_path
            
            with fits.open(file_path) as hdul:
                
                self.camera = hdul[0].header['INSTRUME']
                self.dateobs = str(hdul[0].header['DATE-OBS']).split('T')[0].replace('-','')
                
                self.data = hdr = hdul[0].data
        
        else:
            
            raise IOError('Cannot find bad pixel mask at '+file_path)

    def load_latest_mask(self,camera,setup):
        """Function to locate the most recent mask available for a given
        camera and read it in.
        
        Inputs:
            :param str camera: ID code of the instrument
            :param object setup: Pipeline setup instance for this reduction
        """
        
        self.camera = camera
        
        bpm_list = glob.glob(path.join(setup.pipeline_config_dir,
                                       'bpm_'+camera+'_*.fits'))
        
        if len(bpm_list) > 0:
            
            date_list = []
            
            for f in bpm_list:
                
                date_list.append(str(path.basename(f)).replace('.fits','').split('_')[2])
            
            idx = (np.array(date_list)).argsort()
    
            
            file_path = bpm_list[idx[-1]]
            
            self.read_mask(file_path)
    
    def add_image_mask(self,image_mask):
        """Function to combine the instrumental bad pixel mask (generated 
        offline for each instrument based on its flat field data and including
        permanent features such as dead columns, insensitive pixels etc) with 
        the BANZAI-produced mask for an individual image, which includes 
        saturated pixels.
        
        Input:
            :param array image_mask: Numpy data array for the image BPM
        """
        
        idx = np.where(image_mask != 0.0)
    
        self.data[idx] = image_mask[idx]
        