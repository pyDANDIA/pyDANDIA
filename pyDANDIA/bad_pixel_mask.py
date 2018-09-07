# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:50:05 2018

@author: rstreet, ebachelet
"""

from os import path
import numpy as np
from astropy.io import fits
import glob
import copy 

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


class PixelCluster():
    """Class describing the properties of clusters of pixels in an image"""
    
    def __init__(self,index=None, xc=None, yc=None):
        self.index = index
        self.xc = xc
        self.yc = yc
        self.pixels = []
        self.neighbours = []
        if xc != None and yc != None:
            self.pixels = [ [xc,yc] ]
    
    def summary(self):
        
        return str(self.index)+' (xc,yc)= '+str(self.xc)+', '+str(self.yc)+' '+\
                str(len(self.pixels))+' contiguous pixels'
                
    def check_pixel_in_cluster(self,xp,yp):
        """Method to verify whether a given pixel is already in this 
        cluster"""
        
        for p in self.pixels:
            
            if p[0] == xp and p[1] == yp:
                
                return True
        
        return False
        
    def id_neighbouring_pixels(self,image_shape):
        """Method to identify the x,y pixel positions of all pixels
        neighbouring all pixels within this cluster.
        The resulting list excludes pixels that are already within this
        cluster, or off the edge of the image.
        """
        
        self.neighbours = []
        
        for p in self.pixels:
            
            xmin = max(1, p[0]-1)
            xmax = min(p[0]+2,image_shape[1])
            ymin = max(1, p[1]-1)
            ymax = min(p[1]+2,image_shape[0])
            
            for x in range(xmin,xmax,1):
                for y in range(ymin,ymax,1):
                    
                    if self.check_pixel_in_cluster(x,y) == False:
                        
                        self.neighbours.append( (x,y) )
    
    def merge_cluster(self,kluster):
        """Method to merge this cluster with a second kluster
        
        The pixels from the merging cluster's pixel list are added to this 
        cluster's pixel list.  
        
        The centroid of this pixel cluster is recalculated.
        """
        
        merged_pixels_list = self.pixels
        
        for p in kluster.pixels:
            
            if [p[0], p[1]] not in merged_pixels_list:
                merged_pixels_list.append( [p[0], p[1]] )
            
        self.pixels = merged_pixels_list
        
        pixels = np.array(self.pixels)
        
        self.xc = np.median(pixels[:,0])
        self.yc = np.median(pixels[:,1])
        print('Re-calculated cluster center at: '+str(self.xc)+', '+str(self.yc))

    
def construct_the_saturated_pixel_mask(open_image, saturation_level=65535):
    """
    Function to identify the saturated pixels in a given image.
    
    Construct the saturated pixel mask : 0 = good pixel
                                         1 = bad pixel

    :param astropy.image open_image: the opened image
    :param int saturation_level: the level above considered as saturated


    :return: the saturated pixel mask
    :rtype: array_like
    """
    
    try:

        mask = open_image.data >= saturation_level
        saturated_pixel_mask = mask.astype(int)

    except:

        saturated_pixel_mask = np.zeros(open_image[0].data.shape, int)

    return saturated_pixel_mask

def find_clusters_saturated_pixels(saturated_pixel_mask,image_shape):
    """Function to find clusters in the pixels of a saturated pixel mask.
    
    This applies a varient of the Density-Based Spatial Cluster of Applications
    with Noise (DBSCAN) approach developed by Ester, M, et al. (1996), Proc.
    2nd Intl. Conf. Knowledge Discovery and Data Mining (KDD-96), E.Simmouds,
    J.Han & U.M.Fayyad eds., AAAI Press, p.226-231.  
    
    :param array saturated_pixel_mask: Binary mask of saturated pixels
    :param tuple image_shape: Shape of the full image, represented as a np.array
    """
    
    npix = saturated_pixel_mask.shape[0]*saturated_pixel_mask.shape[1]
    
    # Initially, every saturated pixel is considered to be in a 
    # unique cluster of 1 pixel.
    clusters = []
    
    # The cluster_map is a pixel map indicating which cluster each 
    # pixel currently belongs to.  If the pixel is not allocated to a cluster,
    # the cluster_map entry = -1, otherwise the entry indicates the index
    # of the cluster in the clusters list. 
    cluster_map = np.zeros(image_shape,dtype=int)
    cluster_map.fill(-1)
    
    j = -1
    
    idx = np.where(saturated_pixel_mask == 1)
    
    for i in range(0,len(idx[0]),1):
        
        x = idx[0][i]
        y = idx[1][i]
        
        j += 1
        
        clusters.append( PixelCluster(index=j, xc=x, yc=y) )
        
        cluster_map[y,x] = j
        print j,x, y, clusters[-1].index
    
    # Iteratively merge clusters if they have neighbouring saturated pixels
    n_iter = 0
    max_iter = 10
    
    iterate = True
    
    while iterate:
        
        n_iter += 1
        mergers = []
        remaining_clusters = {}
        
        print('Iteration '+str(n_iter))
        
        for ic,c in enumerate(clusters):
            
            if ic not in mergers:
                print('Cluster '+str(c.index)+' at '+str(c.xc)+', '+str(c.yc))
                
                c.id_neighbouring_pixels(image_shape)
                
                print(' -> Found '+str(len(c.neighbours))+' neighbouring pixels')
                print(c.neighbours)
                
                for n in c.neighbours:
                    
                    print(' -> Neighbouring pixel '+repr(n))
                    
                    for ik,k in enumerate(clusters):
                        
                        if ic != ik and ik not in mergers and \
                            k.check_pixel_in_cluster(n[1],n[0]):
                            
                            print('Merging '+str(ik)+' with '+str(ic))
                            
                            mergers.append(ik)
                            
                            c.merge_cluster(k)
                            
                            if c.index not in remaining_clusters.keys():
                                remaining_clusters[c.index] = c
                            
                            print('Remaining clusters: '+repr(remaining_clusters.keys()))
                            print('Mergers: '+repr(mergers))
            
        print('N mergers = '+str(len(mergers)))
        
        clusters = copy.copy(remaining_clusters.values())
        
        print('N remaining clusters = '+str(len(clusters)))
        print(str(remaining_clusters.keys()))
        
        print('Iteration '+str(n_iter))
            
        if len(mergers) == 0 or n_iter >= max_iter or len(clusters) == 1:
            
            iterate = False
            
        print('\nContinue? '+repr(iterate)+'\n')
        
    return clusters
    