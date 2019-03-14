# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:28:17 2018

@author: rstreet
"""

import jester_phot_transforms
import bilir_phot_transforms
import numpy as np
import stellar_radius_relations
from astropy import constants

class Star:
    """Class describing the photometric parameters of a single stellar object"""
    
    def __init__(self):
        
        self.g = None
        self.sig_g = None
        self.r = None
        self.sig_r = None
        self.i = None
        self.sig_i = None
        
        self.gr = None
        self.sig_gr = None
        self.gi = None
        self.sig_gi = None
        self.ri = None
        self.sig_ri = None
        
        self.B = None
        self.sig_B = None
        self.V = None
        self.sig_V = None
        self.R = None
        self.sig_R = None
        self.I = None
        self.sig_I = None
        self.BV = None
        self.sig_BV = None
        self.RI = None
        self.sig_RI = None
        self.VI = None
        self.sig_VI = None
        self.VR = None
        self.sig_VR = None
        
        self.lightcurves = {'g': None, 'r': None, 'i': None}
        
    def compute_colours(self,use_inst=True,use_cal=False):
        """Method to calculate the star's colours and colour uncertainties, 
        given measurements in 3 passbands.  
        use_inst=True will apply this calculation to the input magnitudes
        use_cal=True will apply it to the extinction-corrected magnitudes
        """
        
        if use_inst:
            
            (self.gr,self.sig_gr) = self.calc_colour(self.g,self.sig_g,self.r,self.sig_r)
            (self.gi,self.sig_gi) = self.calc_colour(self.g,self.sig_g,self.i,self.sig_i)
            (self.ri,self.sig_ri) = self.calc_colour(self.r,self.sig_r,self.i,self.sig_i)
        
        if use_cal:
            
            (self.gr_0,self.sig_gr_0) = self.calc_colour(self.g_0,self.sig_g_0,self.r_0,self.sig_r_0)
            (self.gi_0,self.sig_gi_0) = self.calc_colour(self.g_0,self.sig_g_0,self.i_0,self.sig_i_0)
            (self.ri_0,self.sig_ri_0) = self.calc_colour(self.r_0,self.sig_r_0,self.i_0,self.sig_i_0)
                                    
    def calc_colour(self,col1, sig_col1, col2, sig_col2):
        
        col = col1 - col2
        sig = np.sqrt( (sig_col1*sig_col1)  + \
                        (sig_col2*sig_col2) )
        return col, sig
    
    def set_delta_mag(self, params):
        """When comparing the magnitudes measured from DanDIA in the lightcurve
        with those from pyDANDIA in the calibration data, the total correction 
        to the magnitude consists of:
        
        corr_mag = lc_mag + delta_m
        
        where delta_m = lc_mag[ref_frame] - cal_ref_mag[star_catalog] + dmag[RC offset]
        
        and:
        lc_mag[ref_frame] = magnitude of star in lightcurve entry for reference frame
        cal_ref_mag[star_catalog] = magnitude of star from metadatas star_catalog
        
        Note: The reference frame used to produce both the lightcurve and the
        metadata must be the same.  
        
        These coefficients are measured by the code calibrate_lightcurve.py, 
        and set from the input parameters here. 
        """
        
        for f in ['g', 'r', 'i']:
            
            setattr(self, 'delta_m_'+f, params['delta_m_'+f])
            setattr(self, 'sig_delta_m_'+f, params['sig_delta_m_'+f])
            
    def summary(self,show_mags=True, show_cal=False, show_colours=False, 
                johnsons=False, show_instrumental=False):
        
        output = ''
        
        if show_mags:
            output = 'g_meas = '+str(round(self.g,3))+' +/- '+str(round(self.sig_g,3))+\
                ' r_meas = '+str(round(self.r,3))+' +/- '+str(round(self.sig_r,3))+\
                ' i_meas = '+str(round(self.i,3))+' +/- '+str(round(self.sig_i,3))
        
        elif show_instrumental:
            output = 'g_inst = '+str(round(self.g_inst,3))+' +/- '+str(round(self.sig_g_inst,3))+\
                ' r_inst = '+str(round(self.r_inst,3))+' +/- '+str(round(self.sig_r_inst,3))+\
                ' i_inst = '+str(round(self.i_inst,3))+' +/- '+str(round(self.sig_i_inst,3))
                
        elif show_cal and show_mags == False and show_colours == False and johnsons == False:
            output = 'g_0 = '+str(round(self.g_0,3))+' +/- '+str(round(self.sig_g_0,3))+\
                ' r_0 = '+str(round(self.r_0,3))+' +/- '+str(round(self.sig_r_0,3))+\
                ' i_0 = '+str(round(self.i_0,3))+' +/- '+str(round(self.sig_i_0,3))
            
        elif show_colours and show_cal == False and show_mags == False and johnsons == False:
            output = '(g-r)_meas = '+str(round(self.gr,3))+' +/- '+str(round(self.sig_gr,3))+\
                ' (g-i)_meas = '+str(round(self.gi,3))+' +/- '+str(round(self.sig_gi,3))+\
                ' (r-i)_meas = '+str(round(self.ri,3))+' +/- '+str(round(self.sig_ri,3))
                
        elif show_colours and show_cal and show_mags == False and johnsons == False:
            output = '(g-r)_0 = '+str(round(self.gr_0,3))+' +/- '+str(round(self.sig_gr_0,3))+\
                ' (g-i)_0 = '+str(round(self.gi_0,3))+' +/- '+str(round(self.sig_gi_0,3))+\
                ' (r-i)_0 = '+str(round(self.ri_0,3))+' +/- '+str(round(self.sig_ri_0,3))
        
        elif johnsons and show_mags == False and show_cal == False:
            
            if self.VR != None and self.RI != None:
                output += '(V-R)_meas = '+str(self.VR)+' +/- '+str(self.sig_VR)+'mag '+\
                         '(Rc-Ic)_meas = '+str(self.RI)+' +/- '+str(self.sig_RI)+'mag'
    
            if self.V != None and self.BV != None:
            
                output += 'V_meas = '+str(self.V)+' +/- '+str(self.sig_V)+'mag '+\
                         '(B-V)_meas = '+str(self.BV)+' +/- '+str(self.sig_BV)+'mag'
    
            if self.V != None and self.VR != None:
                    
                output += 'V_meas = '+str(self.V)+' +/- '+str(self.sig_V)+'mag'+\
                         ' I_meas = '+str(self.I)+' +/- '+str(self.sig_I)+'mag'+\
                         ' (V-I)_meas = '+str(self.VI)+' +/- '+str(self.sig_VI)+'mag'
            
        elif johnsons and show_mags == False and show_cal:
            
            try:
                output += '(V-R)_0 = '+str(self.VR_0)+' +/- '+str(self.sig_VR_0)+'mag '+\
                         '(Rc-Ic)_0 = '+str(self.RI_0)+' +/- '+str(self.sig_RI_0)+'mag'
            except AttributeError:
                pass
                        
            try:
                output += 'V_0 = '+str(self.V_0)+' +/- '+str(self.sig_V_0)+'mag '+\
                         '(B-V)_0 = '+str(self.BV_0)+' +/- '+str(self.sig_BV_0)+'mag'
            except AttributeError:
                pass
            
            try:    
                output += 'V_0 = '+str(self.V_0)+' +/- '+str(self.sig_V_0)+'mag'+\
                         ' I_0 = '+str(self.I_0)+' +/- '+str(self.sig_I_0)+'mag'+\
                         ' (V-I)_0 = '+str(self.VI_0)+' +/- '+str(self.sig_VI_0)+'mag'
            except AttributeError:
                pass
            
        return output
    
    def transform_to_JohnsonCousins(self):
        
        if self.ri != None:
            
            target_phot = jester_phot_transforms.transform_SDSS_to_JohnsonCousins(ri=self.ri, 
                                                                                  sigri=self.sig_ri)
            self.VR = target_phot['V-R']
            self.sig_VR = target_phot['sigVR']
            self.RI = target_phot['Rc-Ic']
            self.sig_RI = target_phot['sigRI']
        
        if self.g != None and self.gr != None:
            
            target_phot = jester_phot_transforms.transform_SDSS_to_JohnsonCousins(g=self.g,
                                                                                  sigg=self.sig_g,
                                                                                  gr=self.gr, 
                                                                                  siggr=self.sig_gr)
            
            self.V = target_phot['V']
            self.sig_V = target_phot['sigV']
            self.BV = target_phot['B-V']
            self.sig_BV = target_phot['sigBV']
            
        
        self.R = self.V - self.VR
        self.sig_R = np.sqrt( (self.sig_V*self.sig_V) + (self.sig_VR*self.sig_VR) ) 
        self.I = self.R - self.RI
        self.sig_I = np.sqrt( (self.sig_R*self.sig_R) + (self.sig_RI*self.sig_RI) ) 
        self.VI = self.V - self.I
        self.sig_VI = np.sqrt( (self.sig_V*self.sig_V) + (self.sig_I*self.sig_I) ) 

    def transform_2MASS_to_SDSS(self):
        
        (self.gr_0, self.sig_gr_0, self.ri_0, self.sig_ri_0) = bilir_phot_transforms.transform_2MASS_to_SDSS(JH=self.JH_0, HK=self.HK_0, MH=None)
    
    def calibrate_phot_properties(self, RC, log=None, verbose=False):
        """Function to calculate the de-reddened and extinction-corrected 
        photometric properties of the Star
        """
        
        self.g_0 = self.g - RC.A_g
        self.sig_g_0 = np.sqrt( (self.sig_g*self.sig_g) + (RC.sig_A_g*RC.sig_A_g) )
        self.r_0 = self.r - RC.A_r
        self.sig_r_0 = np.sqrt( (self.sig_r*self.sig_r) + (RC.sig_A_r*RC.sig_A_r) )
        self.i_0 = self.i - RC.A_i
        self.sig_i_0 = np.sqrt( (self.sig_i*self.sig_i) + (RC.sig_A_i*RC.sig_A_i) )
        self.gr_0 = self.gr - RC.Egr
        self.sig_gr_0 = np.sqrt( (self.sig_gr*self.sig_gr) + (RC.sig_Egr*RC.sig_Egr) )
        self.gi_0 = self.gi - RC.Egi
        self.sig_gi_0 = np.sqrt( (self.sig_gi*self.sig_gi) + (RC.sig_Egi*RC.sig_Egi) )
        self.ri_0 = self.ri - RC.Eri
        self.sig_ri_0 = np.sqrt( (self.sig_ri*self.sig_ri) + (RC.sig_Eri*RC.sig_Eri) )

        self.gr_0 = self.g_0 - self.r_0
        self.sig_gr_0 = np.sqrt( (self.sig_g*self.sig_g) + (self.sig_r*self.sig_r) )
        self.gi_0 = self.g_0 - self.i_0
        self.sig_gi_0 = np.sqrt( (self.sig_g*self.sig_g) + (self.sig_i*self.sig_i) )
        self.ri_0 = self.r_0 - self.i_0
        self.sig_ri_0 = np.sqrt( (self.sig_r*self.sig_r) + (self.sig_i*self.sig_i) )

        
        self.I_0 = self.I - RC.A_I
        self.sig_I_0 = np.sqrt( (self.sig_I*self.sig_I) + (RC.sig_A_I*RC.sig_A_I) )
        self.V_0 = self.V - RC.A_V
        self.sig_V_0 = np.sqrt( (self.sig_V*self.sig_V) + (RC.sig_A_V*RC.sig_A_V) )
        self.VI_0 = self.VI - RC.EVI
        self.sig_VI_0 = np.sqrt( (self.sig_VI*self.sig_VI) + (RC.sig_EVI*RC.sig_EVI) )
        
        if verbose:
            output = '\nExtinction-corrected magnitudes and de-reddened colours:\n'
            output += 'g_S,0 = '+str(self.g_0)+' +/- '+str(self.sig_g_0)+'\n'
            output += 'r_S,0 = '+str(self.r_0)+' +/- '+str(self.sig_r_0)+'\n'
            output += 'i_S,0 = '+str(self.i_0)+' +/- '+str(self.sig_i_0)+'\n'
            output += '(g-r)_S,0 = '+str(self.gr_0)+' +/- '+str(self.sig_gr_0)+'\n'
            output += '(g-i)_S,0 = '+str(self.gi_0)+' +/- '+str(self.sig_gi_0)+'\n'
            output += '(r-i)_S,0 = '+str(self.ri_0)+' +/- '+str(self.sig_ri_0)
            
            if log != None:
                log.info(output)
            else:
                print(output)
        
        in_use = False
        if in_use:
            phot = jester_phot_transforms.transform_SDSS_to_JohnsonCousins(ri=self.ri_0, 
                                                                           sigri=self.sig_ri_0)
            
            self.VR_0 = phot['V-R']
            self.sig_VR_0 = phot['sigVR']
            self.RI_0 = phot['Rc-Ic']
            self.sig_RI_0 = phot['sigRI']
        
            if verbose:
                output = '\n(V-R)_S,0 = '+str(self.VR_0)+' +/- '+str(self.sig_VR_0)+'mag\n'
                output += '(R-I)_S,0 = '+str(self.RI_0)+' +/- '+str(self.sig_RI_0)+'mag\n'
            
            phot = jester_phot_transforms.transform_SDSS_to_JohnsonCousins(g=self.g_0,
                                                                           sigg=self.sig_g_0,
                                                                           gr=self.gr_0, 
                                                                           siggr=self.sig_gr_0)
            self.V_0 = phot['V']
            self.sig_V_0 = phot['sigV']
            self.BV_0 = phot['B-V']
            self.sig_BV_0 = phot['sigBV']
        
            
            if verbose:
                output += '\n(B-V)_S,0 = '+str(self.BV_0)+' +/- '+str(self.sig_BV_0)+'mag\n'
                output += 'V_S,0 = '+str(self.V_0)+' +/- '+str(self.sig_V_0)+'mag'
            
                if log != None:
                    log.info(output)
                else:
                    print(output)
    
    def calc_stellar_ang_radius(self, log):
        """Function to calculate the angular radius of the star"""
    
        def calc_theta(log_theta_LD,sig_log_theta_LD):
            
            theta_LD = 10**(log_theta_LD) * 1000.0
            sig_theta_LD = (sig_log_theta_LD/abs(log_theta_LD)) * theta_LD
            
            (ang_radius,sig_ang_radius) = calc_ang_radius(theta_LD,sig_theta_LD)
            
            return theta_LD, sig_theta_LD, ang_radius, sig_ang_radius
        
        def calc_ang_radius(theta_LD,sig_theta_LD):
            
            ang_radius = theta_LD / 2.0
            sig_ang_radius = (sig_theta_LD / theta_LD) * ang_radius
            
            return ang_radius, sig_ang_radius
            
        radii = []
        
        log.info('\n')
        
        (log_theta_LD, sig_log_theta_LD, flag) = stellar_radius_relations.calc_star_ang_radius_Adams2018(self.V_0,self.sig_V_0,self.VI_0,self.sig_VI_0,'V-I',Lclass='dwarfs',log=log)
        
        (theta_LD,sig_theta_LD, ang_radius, sig_ang_radius) = calc_theta(log_theta_LD,sig_log_theta_LD)
        
        log.info('Based on Adams et al.(2018) relations for Johnsons passbands:')
        log.info(' -> Using the (V-I) colour index = '+str(round(self.VI_0,3)))
        log.info(' -> Assuming the star is a dwarf:')
        log.info(' -> Log_10(theta_LD) = '+str(log_theta_LD)+' +/- '+str(sig_log_theta_LD))
        log.info(' -> Theta_LD = '+str(round(theta_LD,3))+' +/- '+str(round(sig_theta_LD,3))+' microarcsec')
        log.info(' -> Angular radius = '+str(round(ang_radius,3))+' +/- '+str(round(sig_ang_radius,3))+' microarcsec')
        log.info(' -> Valid estimate? '+repr(flag))
        
        if flag:
            radii.append(ang_radius)
        
        log.info('\n')
        
        (log_theta_LD, sig_log_theta_LD, flag) = stellar_radius_relations.calc_star_ang_radius_Adams2018(self.V_0,self.sig_V_0,self.VI_0,self.sig_VI_0,'V-I',Lclass='giants',log=log)
        
        (theta_LD,sig_theta_LD, ang_radius, sig_ang_radius) = calc_theta(log_theta_LD,sig_log_theta_LD)
        
        log.info(' -> Assuming the star is a giant:')
        log.info(' -> Log_10(theta_LD) = '+str(log_theta_LD)+' +/- '+str(sig_log_theta_LD))
        log.info(' -> Theta_LD = '+str(round(theta_LD,3))+' +/- '+str(round(sig_theta_LD,3))+' microarcsec')
        log.info(' -> Angular radius = '+str(round(ang_radius,3))+' +/- '+str(round(sig_ang_radius,3))+' microarcsec')
        log.info(' -> Valid estimate? '+repr(flag))
        
        if flag:
            radii.append(ang_radius)
        
        log.info('\n')
        
        (log_theta_LD, sig_log_theta_LD, flag_gr) = stellar_radius_relations.calc_star_ang_radius_Boyajian2014(self.gr_0,self.sig_gr_0,self.g_0,self.sig_g_0,'g-r',0.0,log=log)
        
        (theta_LD_gr, sig_theta_LD_gr, ang_radius_gr, sig_ang_radius_gr) = calc_theta(log_theta_LD,sig_log_theta_LD)
        
        log.info('Based on Boyajian et al. 2014 relations for SDSS/Johnsons passbands:')
        log.info('Applies to main-sequence stars only.')
        log.info(' -> Using the (g-r) colour index = '+str(round(self.gr_0,3)))
        log.info(' -> Log_10(theta_LD) = '+str(log_theta_LD)+' +/- '+str(sig_log_theta_LD))
        log.info(' -> Theta_LD = '+str(round(theta_LD_gr,3))+' +/- '+str(round(sig_theta_LD_gr,3))+' microarcsec')
        log.info(' -> Angular radius = '+str(round(ang_radius_gr,3))+' +/- '+str(round(sig_ang_radius_gr,3))+' microarcsec')
        log.info(' -> Valid estimate? '+repr(flag_gr))
        
        if flag:
            radii.append(ang_radius_gr)
        
        log.info('\n')
        
        (log_theta_LD, sig_log_theta_LD, flag_gi) = stellar_radius_relations.calc_star_ang_radius_Boyajian2014(self.gi_0,self.sig_gi_0,self.g_0,self.sig_g_0,'g-i',0.0,log=log)
        
        (theta_LD_gi,sig_theta_LD_gi, ang_radius_gi, sig_ang_radius_gi) = calc_theta(log_theta_LD,sig_log_theta_LD)
        
        log.info(' -> Using the (g-i) colour index = '+str(round(self.gi_0,3)))
        log.info(' -> Log_10(theta_LD) = '+str(log_theta_LD)+' +/- '+str(sig_log_theta_LD))
        log.info(' -> Theta_LD = '+str(round(theta_LD_gi,3))+' +/- '+str(round(sig_theta_LD_gi,3))+' microarcsec')
        log.info(' -> Angular radius = '+str(round(ang_radius_gi,3))+' +/- '+str(round(sig_ang_radius_gi,3))+' microarcsec')
        log.info(' -> Valid estimate? '+repr(flag_gi))
        
        if flag:
            radii.append(ang_radius_gr)
        
        use_index = None
        
        if flag_gr and flag_gi:
            
            self.theta = (theta_LD_gr + theta_LD_gi)/2.0
            self.sig_theta = np.sqrt( sig_theta_LD_gr*sig_theta_LD_gr +\
                                        sig_theta_LD_gi*sig_theta_LD_gi)
                                        
            (self.ang_radius,self.sig_ang_radius) = calc_ang_radius(self.theta,self.sig_theta)
            
            use_index = 'average of (g-r) and (g-i)'
            
        elif flag_gr and flag_gi == False:
            
            self.theta = theta_LD_gr
            self.sig_theta = sig_theta_LD_gr
            self.ang_radius = ang_radius_gr
            self.sig_ang_radius = sig_ang_radius_gr
            
            use_index = '(g-r)'

        elif flag_gr == False and flag_gi:
            
            self.theta = theta_LD_gi
            self.sig_theta = sig_theta_LD_gi
            self.ang_radius = ang_radius_gi
            self.sig_ang_radius = sig_ang_radius_gi

            use_index = '(g-i)'
            
        log.info('\n')
        
        (log_theta_LD, sig_log_theta_LD, flag) = stellar_radius_relations.calc_star_ang_radius_Boyajian2014(self.VI_0,self.sig_VI_0,self.V_0,self.sig_V_0,'V-I',0.0,log=log)
        
        (theta_LD,sig_theta_LD, ang_radius, sig_ang_radius) = calc_theta(log_theta_LD,sig_log_theta_LD)
        
        log.info(' -> Using the (V-I) colour index = '+str(round(self.VI_0,3)))
        log.info(' -> Log_10(theta_LD) = '+str(log_theta_LD)+' +/- '+str(sig_log_theta_LD))
        log.info(' -> Theta_LD = '+str(round(theta_LD,3))+' +/- '+str(round(sig_theta_LD,3))+' microarcsec')
        log.info(' -> Angular radius = '+str(round(ang_radius,3))+' +/- '+str(round(sig_ang_radius,3))+' microarcsec')
        log.info(' -> Valid estimate? '+repr(flag))
        
        if flag:
            radii.append(ang_radius)
            
        if len(radii) > 0:
            radii = np.array(radii)
            
            if radii.std() > 1.0:
                log.info('WARNING: High degree of scatter '+str(round(radii.std(),4))+\
                ' in angular radii estimates from different colour indices - possibly unreliable')
            
        else:
            log.info('\n')
            log.info('WARNING: No reliable angular radii estimates were possible')
        
        log.info('\n')
        if use_index == None:
            log.info('WARNING: No reliable angular radii estimates were possible')
        else:
            log.info('Adopting theta_S value derived from index: '+use_index)
            log.info('Source angular diameter = '+str(round(self.theta,3))+' +/- '+str(round(self.sig_theta,3))+' microarcsec')
            log.info('Source angular radius = '+str(round(self.ang_radius,3))+' +/- '+str(round(self.sig_ang_radius,3))+' microarcsec')
            
        log.info('\n')

    def calc_physical_radius(self, log):
        """Function to infer the physical radius of the source star from the 
        Torres mass-radius relation based on Teff, logg, and Fe/H
        
        Assumes a solar metallicity of Zsol = 0.0152.    
        """
        
        (self.radius, self.sig_radius) = stellar_radius_relations.star_mass_radius_relation(self.teff,self.logg,0.0152,log=log)
    
    def calc_distance(self,log):
        """Function to calculate the distance to the source star, given the
        angular and physical radius estimates"""
        
        def calc_D(R_S, sig_RS, theta_S, sig_theta_S):
            
            D = (R_S/np.tan(theta_S)) / constants.pc.value
            sig_D = np.sqrt((sig_RS/R_S)**2 + ((1.0/sig_theta_S)/(1.0/theta_S)**2))*D
            
            return D, sig_D
            
        theta_S = ((self.ang_radius / 1e6)/3600.0)*(np.pi/180.0)  # radians
        sig_theta_S = ((self.sig_ang_radius / 1e6)/3600.0)*(np.pi/180.0)
        
        R_S = self.radius * constants.R_sun.value  # units of m
        sig_RS = self.sig_radius * constants.R_sun.value
        
        (self.D,self.sig_D) = calc_D(R_S, sig_RS, theta_S, sig_theta_S)
        
        try:
            R_S = self.starR_small_giant * constants.R_sun.value 
            sig_RS = self.sig_starR_small_giant * constants.R_sun.value
            
            (self.D_small_giant,self.sig_D_small_giant) = calc_D(R_S, sig_RS, theta_S, sig_theta_S)
            
            R_S = self.starR_large_giant * constants.R_sun.value 
            sig_RS = self.sig_starR_large_giant * constants.R_sun.value
            
            (self.D_large_giant,self.sig_D_large_giant) = calc_D(R_S, sig_RS, theta_S, sig_theta_S)
            
        except AttributeError:
            pass
        
    def estimate_luminosity_class(self,log=None):
        """Function to estimate the luminosity class of an instance, based on
        the divisions based on the threshold relating log(g) and t_eff 
        derived from 
        Ciardi, D. et al. (2011), AJ, 141, 108.
        """
        if self.teff >= 6000.0:
            
            self.logg_thresh = 3.5
            
        elif self.teff <= 4250:
            
            self.logg_thresh = 4.0
            
        elif 4250.0 < self.teff and self.teff < 6000:
            
            self.logg_thresh = 5.2 - (2.8e-4*self.teff)
        
        if log!=None:
            log.info('\n')
            log.info('Calculated minimum log(g) for a dwarf star of teff='+\
                        str(round(self.teff,1))+': '+
                        str(round(self.logg_thresh,1)))
        
        if self.logg > self.logg_thresh:
            
            self.Lclass = 'dwarf'
            
        else:
            
            self.Lclass = 'giant'
        
        if log!=None:
            
            log.info(' -> Star is likely to be a '+str(self.Lclass))
    