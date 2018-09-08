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
import logs
import pixelmasks
import matplotlib.pyplot as plt

class BadPixelMask:
    """Class describing bad pixel masks for imaging data"""
    
    def __init__(self):
        
        self.camera = None
        self.dateobs = None
        self.file_path = None
        self.instrument_mask = np.zeros([1])
        self.banzai_mask = np.zeros([1])
        self.saturated_pixels = np.zeros([1])
        self.low_pixels = np.zeros([1])
        self.variable_stars = np.zeros([1])
        self.master_mask = np.zeros([1])
        
    def create_empty_masks(self,image_dims):
        
        self.instrument_mask = np.zeros(image_dims,int)
        self.banzai_mask = np.zeros(image_dims,int)
        self.saturated_pixels = np.zeros(image_dims,int)
        self.low_pixels = np.zeros(image_dims,int)
        self.master_mask = np.zeros(image_dims,int)
        
    def read_mask(self,file_path,log=None):
        """Function to read a camera mask from a pre-existing file
        
        Input:
            :param str file_path: Path to the file to be read
        """
        
        if path.isfile(file_path):
            
            self.file_path = file_path
            
            with fits.open(file_path) as hdul:
                
                self.camera = hdul[0].header['INSTRUME']
                self.dateobs = str(hdul[0].header['DATE-OBS']).split('T')[0].replace('-','')
                
                self.instrument_mask = hdul[0].data.astype(int)
                
            if log != None:
                log.info('Read bad pixel mask from '+file_path)
                
        else:
            
            if log != None:
                log.info('ERROR: Cannot read bad pixel mask from '+file_path)
                
            raise IOError('Cannot find bad pixel mask at '+file_path)

    def load_latest_instrument_mask(self,camera,setup,log=None):
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
            
            self.read_mask(file_path,log=log)
            
    def load_banzai_mask(self,image_mask,integers_to_flag=[1,3], log=None):
        """Function to combine the existing BPM data with the BANZAI-produced 
        mask for an individual image, which includes saturated pixels.
        
        BANZAI's definition of bad pixels are labeled 1 or 3 in its BPM. 
        These values are used as defaults but can be overridden.
        
        All pixels marked as bad are set to 1 within the pipeline.
        
        Input:
            :param array image_mask: Numpy data array for the image BPM
            :param list integers_to_flag: [Optional] List of integer pixel values
                                          from the input BPM which should be
                                          included in the output BPM.
        """
        
        if self.banzai_mask.shape != image_mask.shape:

            if log != None:

                log.warning('The bad_pixel_mask dimensions does not match image dimensions! ')
        else:
            
            for flag in integers_to_flag:
                
                index_saturation = np.where((image_mask == flag))
                
                self.banzai_mask[index_saturation] = 1
                
            if log != None:
                    
                log.info('Loaded the BANZAI bad pixel mask for this image')
                
    def mask_image_saturated_pixels(self, open_image, 
                                    saturation_level=65535, log=None):
        """
        Method to identify the saturated pixels in a given image and combine
        them with the BPM data.
        
        Construct the saturated pixel mask : 0 = good pixel
                                             1 = bad pixel
    
        :param astropy.image open_image: the opened image
        :param int saturation_level: the level above considered as saturated
        """
        
        try:
            
            mask = open_image.data >= saturation_level
            
            self.saturated_pixels = mask.astype(int)
            
            if log!=None:
                log.info('Masked saturated pixels for this image')
            
        except:
    
            pass
        
    def mask_image_low_level_pixels(self, open_image, low_level=0, log=None):
        """
        Construct the low level pixel mask : 0 = good pixel
                                             1 = bad pixel
    
        :param astropy.image open_image: the opened image
        :param int low_level: the level below is considered as bad value
        """
    
        try:
            mask = open_image.data <= low_level

            self.low_pixels = mask.astype(int)

            if log != None:
                
                log.info('Masked low pixels for this image')
                
        except:
    
            pass

    def mask_variable_stars(open_image, variable_star_pixels=10):
        '''
        Construct the variable stars pixel mask : 0 = good pixel
                                                  1 = bad pixel
    
        NEED WORK :)
    
        :param astropy.image open_image: the opened image
        :param int  variable_star_pixels: the pixel radius you want to flagged around the variable object
    
        :return: the variable star pixel mask
        :rtype: array_like
        '''
    
        try:
    
            pass
    
        except:
    
            pass
    
        pass
    
    def mask_ccd_blooming(self,setup,reduction_metadata,image,log):
        """Method to identify and mask regions of an image where bright stars 
        have become so saturated that charge has bled along the columns, and
        a bright halo is created with a steep sky gradient."""
        
        psf_radius = reduction_metadata.reduction_parameters[1]['PSF_SIZE'][0]/2.0
        min_cluster_pix = int(np.pi * psf_radius * psf_radius)
            
        sat_columns = find_columns_with_blooming(self.saturated_pixels)
        
        sat_regions = []
        
        for columns in sat_columns:
            
            sat_image_region = self.saturated_pixels[:,columns[1]:columns[2]]
            
            sat_rows = find_rows_with_blooming(sat_image_region, 
                                                  diagnostic_plots=True)
            
            print('X center '+str(columns[0])+' range: '+str(columns[1])+' to '+str(columns[2]))
            for rows in sat_rows:
                print('Y center '+str(rows[0])+' range: '+str(rows[1])+' to '+str(rows[2]))
            
            sat_regions.append( [ columns[1], columns[2], rows[1], rows[2] ] )
            
        for i,r in enumerate(sat_regions):
            
            sat_image_region = image[r[2]:r[3],r[0]:r[1]]
            
            hdu = fits.PrimaryHDU(sat_image_region)
            hdu.writeto('image_section'+str(i)+'.fits', overwrite=True)
            
#            cluster = find_clusters_saturated_pixels(setup,sat_image_region,
#                                                     sat_image_region.shape,log)
            
                
class PixelCluster():
    """Class describing the properties of clusters of pixels in an image"""
    
    def __init__(self,index=None, xc=None, yc=None):
        
        self.index = index
        self.xc = xc
        self.yc = yc
        self.pixels = []
        self.neighbours = []
        self.range = []
        self.xyratio = None
        
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
    
    def calc_min_separation(self,kluster):
        """Method to calculate the minimum Euclidean separation of all pixels 
        in this cluster from the pixels in kluster"""
        
        min_sep = 1e5
        
        kpositions = []
        for p in kluster.pixels:
            kpositions.append([p[0],p[1]])
        kpositions = np.array(kpositions)
        
        for p in self.pixels:
            
            sep = np.sqrt((kpositions[:,0]-p[0])**2 + (kpositions[:,1]-p[1])**2)
            
            if sep.min() < min_sep:
                min_sep = sep.min()
        
        return min_sep
        
    def merge_cluster(self,kluster, log=None):
        """Method to merge this cluster with a second kluster
        
        The pixels from the merging cluster's pixel list are added to this 
        cluster's pixel list.  
        
        The centroid of this pixel cluster is recalculated.
        """
        
        merged_pixels_list = self.pixels
        
        for p in kluster.pixels:
            
            if [p[0], p[1]] not in merged_pixels_list:
                merged_pixels_list.append( [p[0], p[1]] )
        
        if log!=None:
            log.info('Added '+str(len(kluster.pixels))+\
                    ' pixels from cluster '+str(kluster.index)+' to '+
                    str(len(self.pixels))+' from cluster '+str(self.index)+\
                    ' to make a combined list of '+str(len(merged_pixels_list))+'pixels')
                    
        self.pixels = merged_pixels_list
        
        pixels = np.array(self.pixels)
        
        self.xc = np.median(pixels[:,0])
        self.yc = np.median(pixels[:,1])
        
        if log!=None:
            log.info('Re-calculated cluster center at: '+\
                    str(self.xc)+', '+str(self.yc))

    def check_for_identical_cluster(self, kluster, log=None):
        """Method to check whether the pixel lists attributed to two 
        clusters are identical"""
        
        pixels1 = self.pixels
        pixels2 = kluster.pixels
        
        if len(pixels1) != len(pixels2):
            return False
            
        pixels1.sort()
        pixels2.sort()
        
        for ip,p in enumerate(pixels1):
            
            if p != pixels2[ip]:
                
                return False
                
        return True
    
    def calc_pixel_range(self):
        """Method to calculate the minimum and maximum range in x and y
        of all pixels included in this cluster"""
        
        if len(self.pixels) > 1:
            
            xmin = 1e5
            xmax = -1
            ymin = 1e5
            ymax = -1
            
            for p in self.pixels:
                
                xmin = min(p[0],xmin)
                xmax = max(p[0],xmax)
                ymin = min(p[1],ymin)
                ymax = max(p[1],ymax)
            
            self.xyratio = float(xmax-xmin)/float(ymax-ymin)
            
        else:
            
            xmin = 0.0
            xmax = 0.0
            ymin = 0.0
            ymax = 0.0
            self.xyratio = 1.0
        
        self.range = [xmin, xmax, ymin, ymax]
        
def construct_the_pixel_mask(setup, reduction_metadata,
                             open_image, banzai_bpm, integers_to_flag, log,
                             saturation_level=65535, low_level=0,
                             instrument_bpm=None):
    '''
    Construct the global pixel mask  using a bitmask approach.

    :param astropy.image open_image: the opened image
    :param list integers_to_flag: the list of integers corresponding to a bad pixel
    :param float saturation_level: the value above which a pixel is consider saturated
    :param float low_level: the value below which a pixel is consider as bad value


    :return: the low level pixel mask
    :rtype: array_like
    '''
    
    log.info('Constructing the image bad pixel mask')

    try:
        bpm = BadPixelMask()
        
        image_dims = [reduction_metadata.reduction_parameters[1]['IMAGEY2'][0],
                      reduction_metadata.reduction_parameters[1]['IMAGEX2'][0]]
            
        bpm.create_empty_masks(image_dims)
        
        if instrument_bpm == None:
            
            bpm.load_latest_instrument_mask(reduction_metadata.reduction_parameters[1]['INSTRID'][0],setup,log=log)
            
        else:
            
            bpm.instrument_mask = instrument_bpm.instrument_mask
            
            log.info('Included instrumental bad pixel mask data')
            
        if type(banzai_bpm) == type(fits.hdu.image.ImageHDU()):
            
            bpm.load_banzai_mask(banzai_bpm, log=log)
        
        # variables_pixel_mask = construct_the_variables_star_mask(open_image, variable_star_pixels=10)
    
        saturation_level = reduction_metadata.reduction_parameters[1]['MAXVAL']
        
        bpm.mask_image_saturated_pixels(open_image, saturation_level, log=log)
    
        bpm.mask_image_low_level_pixels(open_image, low_level, log=log)
    
        list_of_masks = [bpm.instrument_mask, 
                         bpm.banzai_mask, 
                         bpm.saturated_pixels, 
                         bpm.low_pixels]
        
        bpm.master_mask = pixelmasks.construct_a_master_mask(bpm.master_mask, 
                                                             list_of_masks)
    
        if log != None:
            log.info('Successfully built a BPM')
    
        return bpm

    except:

        #master_mask = np.zeros(open_image.data.shape, int)

        bpm = BadPixelMask()
        
        bpm.create_empty_masks(open_image.data.shape)

        if log != None:
            log.info('Error building the BPM; using zeroed array')

    return bpm


def find_clusters_saturated_pixels(setup,saturated_pixel_mask,image_shape,log):
    """Function to find clusters in the pixels of a saturated pixel mask.
    
    This applies a varient of the Density-Based Spatial Cluster of Applications
    with Noise (DBSCAN) approach developed by Ester, M, et al. (1996), Proc.
    2nd Intl. Conf. Knowledge Discovery and Data Mining (KDD-96), E.Simmouds,
    J.Han & U.M.Fayyad eds., AAAI Press, p.226-231.  
    
    :param array saturated_pixel_mask: Binary mask of saturated pixels
    :param tuple image_shape: Shape of the full image, represented as a np.array
    """

    logs.ifverbose(log,setup,'\n')
    logs.ifverbose(log,setup,'\nAnalysing saturated pixels to look for clusters around bright objects')
        
    # Initially, every saturated pixel is considered to be in a 
    # unique cluster of 1 pixel.
    clusters = []
    
    j = -1
    
    idx = np.where(saturated_pixel_mask == 1)
    
    logs.ifverbose(log,setup,'Image has a total of '+str(len(idx[0]))+' saturated pixels')
    
    for i in range(0,len(idx[0]),1):
        
        x = idx[1][i]
        y = idx[0][i]
        
        j += 1
        
        clusters.append( PixelCluster(index=j, xc=x, yc=y) )
        
    # Iteratively merge clusters if they have neighbouring saturated pixels
    n_iter = 0
    max_iter = 5
    
    iterate = True
    
    while iterate:
        
        n_iter += 1
        mergers = []
        remaining_clusters = {}
        
        logs.ifverbose(log,setup,'Iteration '+str(n_iter))
        
        for ic,c in enumerate(clusters):
            
            if ic not in mergers:
                
                logs.ifverbose(log,setup,'\nCluster '+str(c.index)+' at '+str(c.xc)+', '+str(c.yc))
                
                c.id_neighbouring_pixels(image_shape)
                
                logs.ifverbose(log,setup,' -> Found '+str(len(c.neighbours))+' neighbouring pixels')
                logs.ifverbose(log,setup,c.neighbours)
                
                n_mask_neighbours = 0
                
                for n in c.neighbours:
                                            
                    for ik,k in enumerate(clusters):
                        
                        if ic != ik and ik not in mergers and \
                            k.check_pixel_in_cluster(n[0],n[1]):
                            
                            n_mask_neighbours += 1
                            
                            logs.ifverbose(log,setup,' -> Neighbouring pixel '+\
                                            repr(n)+ ' in cluster '+str(ik)+' '+\
                                            repr(k.check_pixel_in_cluster(n[0],n[1])))
                            
                            logs.ifverbose(log,setup,'Merging '+str(ik)+' with '+str(ic))
                            
                            mergers.append(ik)
                            
                            c.merge_cluster(k,log=log)
                            
                            if c.index not in remaining_clusters.keys():
                                remaining_clusters[c.index] = c
                            
                            logs.ifverbose(log,setup,'Remaining clusters: '+repr(remaining_clusters.keys()))
                            logs.ifverbose(log,setup,'Mergers: '+repr(mergers))
            
                if n_mask_neighbours == 0:
                     
                     remaining_clusters[c.index] = c
                     
                     logs.ifverbose(log,setup,'Remaining clusters: '+repr(remaining_clusters.keys()))
                     logs.ifverbose(log,setup,'Mergers: '+repr(mergers))
                     
        logs.ifverbose(log,setup,'N mergers = '+str(len(mergers)))
        
        logs.ifverbose(log,setup,'\nChecking for duplicated clusters:')
        
        clusters = []
        
        for c in remaining_clusters.values():
            
            new = True
            
            for k in clusters:
                
                if k.check_for_identical_cluster(c):
                    
                    new = False

                    logs.ifverbose(log,setup,'Clusters '+str(k.index)+\
                    ' and '+str(c.index)+\
                    ' have the same pixel list, eliminating the duplicate')

            if new:

                clusters.append(c)
        
        logs.ifverbose(log,setup,'N remaining clusters = '+str(len(clusters)))
        
        logs.ifverbose(log,setup,'Iteration '+str(n_iter))
            
        if len(mergers) == 0 or n_iter >= max_iter or len(clusters) == 1:
            
            iterate = False
            
        logs.ifverbose(log,setup,'\nContinue? '+repr(iterate)+'\n')
        
    return clusters

def find_columns_with_blooming(saturated_pixel_mask, diagnostic_plots=False):
    """Function to identify which pixel columns in an image are affected by
    blooming.  
    
    The saturated pixel data is collapsed in the y-direction to create a single 
    pixel row ('spectrum') with 0 for background pixels and >0 for saturated.  
    Blooming affects several (typically ~10) consequtive columns, so affected 
    regions of the image can be identified from peaks in the 1D 'spectrum'.
    """
    
    columns = saturated_pixel_mask.sum(axis=0)
    
    idx = np.where(columns > 200)
    
    clusters = find_clusters_in_vector(idx[0])
    
    idx = np.where(columns == 0)[0]
    
    regions = []
    
    for c in clusters:
        
        sep = abs(idx - c.xc)
        sep.sort()
        
        sat_region_width = (sep[0]+sep[1])
        
        xmin = int(max(1,c.xc-sat_region_width))
        xmax = int(min(c.xc+sat_region_width,len(columns)))
        
        regions.append( [c.xc, xmin, xmax] )
        
    if diagnostic_plots:
        fig = plt.figure()
        
        plt.plot(np.arange(0,len(columns),1), columns, 'm-', label='Columns')
        plt.xlabel('X pixel column')
        plt.ylabel('Accumulated saturation flag')
        
        plt.savefig('saturated_columns_spectrum.png')
    
    return regions
    
def find_rows_with_blooming(saturated_pixel_mask, diagnostic_plots=False):
    """Function to identify which pixel rows in an image are affected by
    blooming.  
    
    The saturated pixel data is collapsed in the x-direction to create a single 
    pixel row ('spectrum') with 0 for background pixels and >0 for saturated.  
    Blooming affects several (typically ~10) consequtive columns, so affected 
    regions of the image can be identified from wide plateax in the 1D 'spectrum'.
    """
    
    rows = saturated_pixel_mask.sum(axis=1)
    
    idx = np.where(rows > 10)
    
    clusters = find_clusters_in_vector(idx[0])
    
    idx = np.where(rows == 0)[0]
    
    regions = []
    
    for c in clusters:
        
        sep = abs(idx - c.xc)
        sep.sort()
        
        sat_region_width = (sep[0]+sep[1])
        
        xmin = int(max(1,c.xc-sat_region_width))
        xmax = int(min(c.xc+sat_region_width,len(rows)))
        
        regions.append( [c.xc, xmin, xmax] )
        
    if diagnostic_plots:
        fig = plt.figure()
        
        plt.plot(np.arange(0,len(rows),1), rows, 'b--', label='Rows')
        plt.title('Rows')
        plt.xlabel('Y pixel row')
        
        plt.savefig('saturated_row_spectrum.png')
    
    return regions
    

def find_clusters_in_vector(vector,verbose=False):
    """Function to identify clusters of contiguous values in an integer vector"""
    
    clusters = []
    
    for i in range(0,len(vector),1):
        clusters.append( PixelCluster(index=i, xc=vector[i], yc=0) )    

    n_iter = 0
    max_iter = 3
    
    iterate = True
    
    while iterate:
        
        n_iter += 1
    
        mergers = []
        remaining_clusters = {}
        
        if verbose: print('Iteration: '+str(n_iter))
        
        for ic,c in enumerate(clusters):
            
            if verbose: print('Cluster '+str(c.index)+' '+repr(c.pixels))
            
            if c.index not in mergers:
                
                n_c_mergers = 0
            
                for ik,k in enumerate(clusters):
                    
                    min_sep = c.calc_min_separation(k)
                    
                    if verbose: 
                        print(' -> kluster '+str(k.index)+\
                                ' separation='+str(min_sep))
                    
                    if ic != ik and abs(min_sep) <= 1:
                        
                        if verbose: 
                            print(' --> Merged kluster '+str(k.index)+\
                                    ' with '+str(c.index))
                        
                        n_c_mergers += 1
                        
                        c.merge_cluster(k)
                        
                        mergers.append(ik)
                        
                        if c.index not in remaining_clusters.keys():
                            remaining_clusters[c.index] = c
            
                if n_c_mergers == 0:
                    remaining_clusters[c.index] = c

            if verbose: 
                print('Mergers: '+repr(mergers))
                print('Remaining clusters: '+repr(remaining_clusters))
            
        clusters = []
        for c in remaining_clusters.values():
            clusters.append(c)
        
        if len(mergers) == 0 or n_iter >= max_iter or len(clusters) == 1:
            iterate = False
        
    return clusters
    