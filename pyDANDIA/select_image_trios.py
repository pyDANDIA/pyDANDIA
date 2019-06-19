# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 13:27:42 2018

@author: rstreet
"""

from os import path
from sys import argv
from pyDANDIA import metadata
from astropy.table import Table
import astropy.units as u
import numpy as np

class TriColourDataset:
    """Class describing the attributes and methods of a combined dataset
    consisting of trios of images taken in 3 filters"""
    
    def __init__(self):
        
        self.gdir = None
        self.rdir = None
        self.idir = None
        self.outdir = None
        self.image_trios = []
        self.fwhm_thresh = 2.0
        self.sky_thresh = 5000.0
        
    def read_meta_data(self):
        """Function to read in the meta data for each dataset"""
        
        for d in [ 'gdir', 'rdir', 'idir' ]:
            
            m = metadata.MetaData()
            
            m.load_a_layer_from_file( getattr(self,d), 
                                      'pyDANDIA_metadata.fits', 
                                      'images_stats' )
                        
            t = Table()
            t['im_name'] = m.images_stats[1]['IM_NAME']
            t['fwhm_x'] = m.images_stats[1]['SIGMA_X']
            t['fwhm_y'] = m.images_stats[1]['SIGMA_Y']
            t['sky'] = m.images_stats[1]['SKY']
            
            setattr(self, d+'_stats', t)
    
    def make_image_table(self):
        """Function to combine the image statistics from all datasets into 
        a single table, sorted into alphanumerical order"""
                
        names = list(self.gdir_stats['im_name']) + \
                        list(self.rdir_stats['im_name']) + \
                        list(self.idir_stats['im_name'])
        
        f = ['g']*len(list(self.gdir_stats['im_name'])) + \
            ['r']*len(list(self.rdir_stats['im_name'])) + \
            ['i']*len(list(self.idir_stats['im_name']))
        
        fwhmx = list(self.gdir_stats['fwhm_x']) + \
                        list(self.rdir_stats['fwhm_x']) + \
                        list(self.idir_stats['fwhm_x'])
                        
        fwhmy = list(self.gdir_stats['fwhm_y']) + \
                        list(self.rdir_stats['fwhm_y']) + \
                        list(self.idir_stats['fwhm_y'])
        sky = list(self.gdir_stats['sky']) + \
                        list(self.rdir_stats['sky']) + \
                        list(self.idir_stats['sky'])
        
        qc = ['0']*len(names)
        
        image_table = np.array(list(zip(names,f,fwhmx,fwhmy,sky,qc)))
        
        idx = np.argsort(image_table[:,0])
        
        for i in range(0,5,1):
            
            image_table[:,i] = image_table[idx,i]
        
        self.image_table = image_table
    
    def quality_control(self):
        """Function to apply quality control selection to the images"""
        
        fwhmx = np.zeros(len(self.image_table))
        fwhmy = np.zeros(len(self.image_table))
        sky = np.zeros(len(self.image_table))
        
        for i in range(0,len(self.image_table),1):
                
            if np.isnan(float(self.image_table[i,2])) == False and \
                np.isnan(float(self.image_table[i,3])) == False and \
                np.isnan(float(self.image_table[i,4])) == False:
                
                fwhmx[i] = float(self.image_table[i,2])
                fwhmy[i] = float(self.image_table[i,3])
                sky[i] = float(self.image_table[i,4])
                
            else:
                
                fwhmx[i] = 99.9
                fwhmy[i] = 99.9
                sky[i] = 9999.9
                        
        jdx = np.where(fwhmx <= self.fwhm_thresh)
        kdx = np.where(fwhmy <= self.fwhm_thresh)
        ldx = np.where(sky <= self.sky_thresh)
        
        idx = list((set(jdx[0]).intersection(set(kdx[0]))).intersection(set(ldx[0])))
        
        self.image_table[idx,5] = '1'
        
    def identify_image_trios(self,txt_output=False):
        """Function to review the sorted image table to identify trios of
        images in different passbands that were taken sequentially in the 
        same observation subrequest."""
        
        def append_image_list(name,f,gimages,rimages,iimages):
            
            if 'g' in f:
                gimages.append(name)
            elif 'r' in f:
                rimages.append(name)
            elif 'i' in f:
                iimages.append(name)
            
            return gimages,rimages,iimages
        
        if txt_output:
            output = open(path.join(self.outdir, 'image_trios.txt'),'w')
            output.write('# Image name   Filter  SIGMA_X   SIGMA_Y   SKY   QC\n')
        
        t = Table()
        
        gimages = []
        rimages = []
        iimages = []
        
        for i in range(0,len(self.image_table[:,0])-2,1):
            
            name1 = self.image_table[i,0]
            name2 = self.image_table[i+1,0]
            name3 = self.image_table[i+2,0]
            date1 = int(str(self.image_table[i,0]).split('-')[2])
            date2 = int(str(self.image_table[i+1,0]).split('-')[2])
            date3 = int(str(self.image_table[i+2,0]).split('-')[2])
            num1 = int(str(self.image_table[i,0]).split('-')[3])
            num2 = int(str(self.image_table[i+1,0]).split('-')[3])
            num3 = int(str(self.image_table[i+2,0]).split('-')[3])
            f1 = self.image_table[i,1]
            f2 = self.image_table[i+1,1]
            f3 = self.image_table[i+2,1]
            qc1 = self.image_table[i,5]
            qc2 = self.image_table[i+1,5]
            qc3 = self.image_table[i+2,5]
            
            if date1 == date2 and date2 == date3 and \
                num2 == (num1 + 1) and num3 == (num2 + 1) and \
                f1 != f2 and f1 != f3 and f2 != f3 and \
                qc1 == '1' and qc2 == '1' and qc3 == '1':
                
                self.image_trios.append( (i,i+1,i+2) )
                
                (gimages,rimages,iimages) = append_image_list(name1,f1,gimages,rimages,iimages)
                (gimages,rimages,iimages) = append_image_list(name2,f2,gimages,rimages,iimages)
                (gimages,rimages,iimages) = append_image_list(name3,f3,gimages,rimages,iimages)
                
                if txt_output:
                    for j in range(i,i+3,1):
                        
                        text = ''
    
                        for item in self.image_table[j,:]:
                            
                            text += ' '+str(item)
                            
                        output.write(text+'\n')
                        
                    output.write('\n')
        
        if txt_output:
            output.close()
        
        t['g_images'] = gimages
        t['r_images'] = rimages
        t['i_images'] = iimages
        
        self.image_trios_table = t
        
def select_image_trios():
    """Function to select trios of images in different passbands that were
    taken as a sequence"""
    
    dataset = get_args()
    
    dataset.read_meta_data()

    dataset.make_image_table()

    dataset.quality_control()
    
    dataset.identify_image_trios(txt_output=True)
    
def get_args():
    """Function to obtain the necessary parameters"""
    
    dataset = TriColourDataset()
    
    if len(argv) != 5:
        
        dataset.gdir = input('Please enter the path to the reduction directory for SDSS-g data: ')
        dataset.rdir = input('Please enter the path to the reduction directory for SDSS-r data: ')
        dataset.idir = input('Please enter the path to the reduction directory for SDSS-i data: ')
        dataset.outdir = input('Please enter the path to the output directory: ')
        
    else:
        
        dataset.gdir = argv[1]
        dataset.rdir = argv[2]
        dataset.idir = argv[3]
        dataset.outdir = argv[4]
    
    return dataset

if __name__ == '__main__':
    
    select_image_trios()
    