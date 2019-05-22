# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:44:01 2019

@author: rstreet
"""

import numpy as np
import os
from pyLIMA import telescopes

class Dataset():
    
    def __init__(self, params=None):
        self.name = None
        self.filter = None
        self.data_file = None
        self.tel = None
        self.gamma = None
        self.k = None
        self.k_err = None
        self.emin = None
        self.emin_err = None
        self.a0 = None
        self.a0_err = None
        self.a1 = None
        self.a1_err = None
        
        if params != None:
            self.name = params['name']
            self.filter = params['filter']
            self.data_file = params['data_file']
    
    def summary(self):
        if self.gamma != None:
            return self.name+' '+self.filter+' '+self.data_file+' '+str(self.gamma)
        else:
            return self.name+' '+self.filter+' '+self.data_file

    def read_dataset_to_telescope(self,model_type,rescaling=[]):
        
        if 'p.t' in self.data_file:
            lightcurve = np.loadtxt(self.data_file,dtype=str)
            lightcurve = np.c_[lightcurve[:,1],lightcurve[:,6],lightcurve[:,7]].astype(float)
        
        if 'cal.t' in self.data_file:
            lightcurve = np.loadtxt(self.data_file,dtype=str)
            lightcurve = np.c_[lightcurve[:,1],lightcurve[:,8],lightcurve[:,9]].astype(float)
                
        if 'DK-1.54' in self.data_file:
            lightcurve = np.loadtxt(self.data_file,dtype=str)
            lightcurve = np.c_[lightcurve[:,1],lightcurve[:,6],lightcurve[:,7]].astype(float)
        
        if len(rescaling) > 0:
            lightcurve = self.apply_error_rescaling(rescaling,lightcurve)
        
        self.tel = telescopes.Telescope(name=self.name, camera_filter=self.filter, 
                                 light_curve_magnitude=lightcurve,
                                 light_curve_magnitude_dictionnary={'time': 0, 'mag': 1, 'err_mag': 2})
        if 'FS' in model_type:
            self.tel.gamma = self.gamma
            print(self.tel.name, self.tel.gamma)
    
    def fetch_gamma(self):

        gamma_coeffs = { 'gp': 0.8371,
                         'rp': 0.6445,
                         'ip': 0.503,
                         'Z': 0.4134 }
    
        gamma = None
    
        for f in gamma_coeffs.keys():
    
            if '_'+f in self.data_file:

                gamma = gamma_coeffs[f]
                
                print(data_file+' -> filter = '+f+' gamma = '+str(gamma))
    
        if gamma == None:
            print('ERROR: No gamma value available for lightcurve '+data_file)
            exit()
    
        return gamma


    def apply_error_rescaling(self,coefficients,lightcurve):
        
        if len(coefficients) > 0:
            
            lightcurve[:,2] =  (coefficients[0]**2 + coefficients[1]**2*lightcurve[:,2]**2)**0.5
            
            print(' -> Applied error rescaling coefficient for dataset '+self.name+\
                    ' a0='+str(coefficients[0])+', a1='+str(coefficients[1]))
            
        return lightcurve
