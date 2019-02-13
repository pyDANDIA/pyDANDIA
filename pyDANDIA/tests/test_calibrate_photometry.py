# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:46:57 2018

@author: rstreet
"""
from os import getcwd, path
from sys import exit
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import calibrate_photometry
import numpy as np
import matplotlib.pyplot as plt

def test_calc_transform():
    """Function to test the photometric transform function"""
    
    a = [ 16.0, 0.15 ]
    x = np.linspace(1.0,100.0,100)
    y = a[0] + (x * a[1]) + np.random.normal(0.0, scale=0.5)
    
    p = [ -1.0, -10.0 ]
    
    fit = calibrate_photometry.calc_transform(p, x, y)
    
    assert fit.all() == np.array(a).all()
    
    fig = plt.figure(1)
    
    xplot = np.linspace(x.min(),x.max(),10)
    yplot = fit[0] + xplot * fit[1]
    
    plt.plot(x,y,'m.')
    
    plt.plot(xplot, yplot,'k-')
    
    plt.xlabel('X')

    plt.ylabel('Y')
        
    plt.savefig('test_transform_function.png')

    plt.close(1)

   
    
if __name__ == '__main__':
    
    test_calc_transform()