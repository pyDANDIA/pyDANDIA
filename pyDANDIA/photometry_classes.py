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
from astropy.table import Column
import json
from os import path

class Star:
    """Class describing the photometric parameters of a single stellar object"""

    def __init__(self, file_path=None):

        self.parameter_list = [ 'g', 'sig_g', 'r', 'sig_r', 'i', 'sig_i',
                                'g_0', 'sig_g_0', 'r_0', 'sig_r_0', 'i_0', 'sig_i_0',
                                'gr', 'sig_gr', 'gi', 'sig_gi', 'ri', 'sig_ri',
                                'gr_0', 'sig_gr_0', 'gi_0', 'sig_gi_0', 'ri_0', 'sig_ri_0',
                                'B', 'sig_B', 'V', 'sig_V', 'R', 'sig_R', 'I', 'sig_I',
                                'B_0', 'sig_B_0', 'V_0', 'sig_V_0', 'R_0', 'sig_R_0', 'I_0', 'sig_I_0',
                                'BV', 'sig_BV', 'RI', 'sig_RI', 'VI', 'sig_VI', 'VR', 'sig_VR',
                                'BV_0', 'sig_BV_0', 'RI_0', 'sig_RI_0', 'VI_0', 'sig_VI_0', 'VR_0', 'sig_VR_0',
                                'A_g', 'sig_A_g', 'A_r', 'sig_A_r', 'A_i', 'sig_A_i',
                                'Egr', 'sig_Egr', 'Egi', 'sig_Egi', 'Eri', 'sig_Eri',
                                'A_I', 'sig_A_I', 'A_V', 'sig_A_V', 'EVI', 'sig_EVI',
                                'D']

        for key in self.parameter_list:
            setattr(self, key, None)

        self.lightcurves = {'g': None, 'r': None, 'i': None}

        if file_path != None and path.isfile(file_path):
            f = open(file_path,'r')
            par_dict = json.loads(f.read())
            f.close()

            for key, value in par_dict.items():
                if 'null' not in str(value) and value != None:
                    setattr(self,key,float(value))
                else:
                    setattr(self,key,None)

    def convert_fluxes_pylima(self, filter_name):

        (mag, err) = flux_to_mag_pylima(getattr(self,'fs_'+filter_name),
                                        getattr(self,'sig_fs_'+filter_name))

        setattr(self, filter_name, mag)
        setattr(self, 'sig_'+filter_name, err)

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

        if col1 != None and col2 != None:
            col = col1 - col2
        else:
            col = None

        if sig_col1 != None and sig_col2 != None:
            sig = np.sqrt( (sig_col1*sig_col1)  + \
                        (sig_col2*sig_col2) )
        else:
            sig = None

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

    def outval(self,a,dp):
        if getattr(self,a) == None:
            return 'None'
        else:
            val = getattr(self,a)
            if type(val) == type(Column()):
                return str(round(val[0],dp))
            else:
                return str(round(val,dp))

    def summary(self,show_mags=True, show_cal=False, show_colours=False,
                johnsons=False, show_instrumental=False):

        output = ''

        if show_mags:
            output = 'g_meas = '+self.outval('g',3)+' +/- '+self.outval('sig_g',3)+\
                ' r_meas = '+self.outval('r',3)+' +/- '+self.outval('sig_r',3)+\
                ' i_meas = '+self.outval('i',3)+' +/- '+self.outval('sig_i',3)

        elif show_instrumental:
            output = 'g_inst = '+self.outval('g_inst',3)+' +/- '+self.outval('sig_g_inst',3)+\
                ' r_inst = '+self.outval('r_inst',3)+' +/- '+self.outval('sig_r_inst',3)+\
                ' i_inst = '+self.outval('i_inst',3)+' +/- '+self.outval('sig_i_inst',3)

        elif show_cal and show_mags == False and show_colours == False and johnsons == False:
            output = 'g_0 = '+self.outval('g_0',3)+' +/- '+self.outval('sig_g_0',3)+\
                ' r_0 = '+self.outval('r_0',3)+' +/- '+self.outval('sig_r_0',3)+\
                ' i_0 = '+self.outval('i_0',3)+' +/- '+self.outval('sig_i_0',3)

        elif show_colours and show_cal == False and show_mags == False and johnsons == False:
            output = '(g-r)_meas = '+self.outval('gr',3)+' +/- '+self.outval('sig_gr',3)+\
                ' (g-i)_meas = '+self.outval('gi',3)+' +/- '+self.outval('sig_gi',3)+\
                ' (r-i)_meas = '+self.outval('ri',3)+' +/- '+self.outval('sig_ri',3)

        elif show_colours and show_cal and show_mags == False and johnsons == False:
            output = '(g-r)_0 = '+self.outval('gr_0',3)+' +/- '+self.outval('sig_gr_0',3)+\
                ' (g-i)_0 = '+self.outval('gi_0',3)+' +/- '+self.outval('sig_gi_0',3)+\
                ' (r-i)_0 = '+self.outval('ri_0',3)+' +/- '+self.outval('sig_ri_0',3)

        elif johnsons and show_mags == False and show_cal == False:

            if self.VR != None and self.RI != None:
                output += ' (V-R)_meas = '+self.outval('VR',3)+' +/- '+self.outval('sig_VR',3)+'mag '+\
                         '(Rc-Ic)_meas = '+self.outval('RI',3)+' +/- '+self.outval('sig_RI',3)+'mag'

            if self.V != None and self.BV != None:

                output += 'V_meas = '+self.outval('V',3)+' +/- '+self.outval('sig_V',3)+'mag '+\
                         '(B-V)_meas = '+self.outval('BV',3)+' +/- '+self.outval('sig_BV',3)+'mag'

            if self.V != None and self.VR != None:

                output += ' V_meas = '+self.outval('V',3)+' +/- '+self.outval('sig_V',3)+'mag'+\
                         ' I_meas = '+self.outval('I',3)+' +/- '+self.outval('sig_I',3)+'mag'+\
                         ' (V-I)_meas = '+self.outval('VI',3)+' +/- '+self.outval('sig_VI',3)+'mag'

        elif johnsons and show_mags == False and show_cal:

            try:
                output += '(V-R)_0 = '+self.outval('VR_0',3)+' +/- '+self.outval('sig_VR_0',3)+'mag '+\
                         '(Rc-Ic)_0 = '+self.outval('RI_0',3)+' +/- '+self.outval('sig_RI_0',3)+'mag'
            except AttributeError:
                pass

            try:
                output += ' V_0 = '+self.outval('V_0',3)+' +/- '+self.outval('sig_V_0',3)+'mag '+\
                         '(B-V)_0 = '+self.outval('BV_0',3)+' +/- '+self.outval('sig_BV_0',3)+'mag'
            except AttributeError:
                pass

            try:
                output += ' I_0 = '+self.outval('I_0',3)+' +/- '+self.outval('sig_I_0',3)+'mag'+\
                         ' (V-I)_0 = '+self.outval('VI_0',3)+' +/- '+self.outval('sig_VI_0',3)+'mag'
            except AttributeError:
                pass

        return output

    def output_json(self,file_path):

        par_dict = {}
        for key in self.parameter_list:
            par_dict[key] = getattr(self,key)

        f = open(file_path,'w')
        f.write(json.dumps(par_dict, indent=4))
        f.close()

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
            self.B = self.BV + self.V
            self.sig_B = np.sqrt( (self.sig_V*self.sig_V) + (self.sig_BV*self.sig_BV) )

        if self.V != None and self.VR != None:
            self.R = self.V - self.VR

        if self.sig_V != None and self.sig_VR != None:
            self.sig_R = np.sqrt( (self.sig_V*self.sig_V) + (self.sig_VR*self.sig_VR) )

        if self.R != None and self.RI != None:
            self.I = self.R - self.RI

        if self.sig_R != None and self.sig_RI != None:
            self.sig_I = np.sqrt( (self.sig_R*self.sig_R) + (self.sig_RI*self.sig_RI) )

        if self.V != None and self.I != None:
            self.VI = self.V - self.I

        if self.sig_V != None and self.sig_I != None:
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

        if flag and use_index == None:
            self.theta = theta_LD
            self.sig_theta = sig_theta_LD
            self.ang_radius = ang_radius
            self.sig_ang_radius = sig_ang_radius

            use_index = '(V-I)'

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

        if 'teff' in dir(self) and self.teff != None and self.logg != None:
            (self.radius, self.sig_radius) = stellar_radius_relations.star_mass_radius_relation(self.teff,self.logg,0.0152,log=log)
        else:
            log.info('Could not infer a physical radius for this star with no Teff and.or log(g) values')

    def calc_distance(self,log):
        """Function to calculate the distance to the source star, given the
        angular and physical radius estimates"""

        def calc_D(R_S, sig_RS, theta_S, sig_theta_S):

            D = (R_S/np.tan(theta_S)) / constants.pc.value
            sig_D = np.sqrt((sig_RS/R_S)**2 + ((1.0/sig_theta_S)/(1.0/theta_S)**2))*D

            return D, sig_D

        theta_S = ((self.ang_radius / 1e6)/3600.0)*(np.pi/180.0)  # radians
        sig_theta_S = ((self.sig_ang_radius / 1e6)/3600.0)*(np.pi/180.0)

        if 'radius' in dir(self) and self.radius != None:
            R_S = self.radius * constants.R_sun.value  # units of m
            sig_RS = self.sig_radius * constants.R_sun.value

            (self.D,self.sig_D) = calc_D(R_S, sig_RS, theta_S, sig_theta_S)
        else:
            log.info('WARNING: Cannot calculate distance without a stellar physical radius estimate')

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

def flux_to_mag_pylima(flux, flux_err):
    """Function to convert the flux and flux uncertainty measured by
    modeling in pyLIMA to magnitudes

    Uses default pyLIMA zeropoint = 27.4 mag
    """

    def flux2mag(ZP, flux):

        return ZP - 2.5 * np.log10(flux)

    ZP = 27.40

    if flux < 0.0 or flux_err < 0.0:

        mag = 0.0
        mag_err = 0.0

    else:

        mag = flux2mag(ZP, flux)

        mag_err = (2.5/np.log(10.0))*flux_err/flux

    return mag, mag_err

def mag_to_flux_pylima(mag, mag_err):
    """Function to convert magnitudes into flux units.
    Magnitude zeropoint is that used by pyLIMA.
    """

    ZP = 27.40

    flux = 10**( (mag - ZP) / -2.5 )

    ferr = mag_err/(2.5*np.log10(np.e)) * flux

    return flux, ferr

def output_red_clump_data_latex(params,RC,log):
    """Function to output a LaTeX format table with the data for the Red Clump"""

    file_path = path.join(params['output_dir'],'red_clump_data_table.tex')

    t = open(file_path, 'w')

    t.write('\\begin{table}[h!]\n')
    t.write('\\centering\n')
    t.write('\\caption{Photometric properties of the Red Clump, with absolute magnitudes ($M_{\\lambda}$) taken from \cite{Ruiz-Dern2018}, and the measured properties from ROME data.} \label{tab:RCproperties}\n')
    t.write('\\begin{tabular}{ll}\n')
    t.write('\\hline\n')
    t.write('\\hline\n')
    t.write('$M_{g,RC,0}$ & '+convert_ndp(RC.M_g_0,3)+' $\pm$ '+convert_ndp(RC.sig_Mg_0,3)+'\,mag\\\\\n')
    t.write('$M_{r,RC,0}$ & '+convert_ndp(RC.M_r_0,3)+' $\pm$ '+convert_ndp(RC.sig_Mr_0,3)+'\,mag\\\\\n')
    t.write('$M_{i,RC,0}$ & '+convert_ndp(RC.M_i_0,3)+' $\pm$ '+convert_ndp(RC.sig_Mi_0,3)+'\,mag\\\\\n')
    t.write('$(g-r)_{RC,0}$ & '+convert_ndp(RC.gr_0,3)+' $\pm$ '+convert_ndp(RC.sig_gr_0,3)+'\,mag\\\\\n')
    t.write('$(g-i)_{RC,0}$ & '+convert_ndp(RC.gi_0,3)+' $\pm$ '+convert_ndp(RC.sig_gi_0,3)+'\,mag\\\\\n')
    t.write('$(r-i)_{RC,0}$ & '+convert_ndp(RC.ri_0,3)+' $\pm$ '+convert_ndp(RC.sig_ri_0,3)+'\,mag\\\\\n')
    t.write('$m_{g,RC,0}$ & '+convert_ndp(RC.m_g_0,3)+' $\pm$ '+convert_ndp(RC.sig_mg_0,3)+'\,mag\\\\\n')
    t.write('$m_{r,RC,0}$ & '+convert_ndp(RC.m_r_0,3)+' $\pm$ '+convert_ndp(RC.sig_mr_0,3)+'\,mag\\\\\n')
    t.write('$m_{i,RC,0}$ & '+convert_ndp(RC.m_i_0,3)+' $\pm$ '+convert_ndp(RC.sig_mi_0,3)+'\,mag\\\\\n')
    t.write('$m_{g,RC,\\rm{centroid}}$  & '+convert_ndp(RC.g,2)+' $\pm$ '+convert_ndp(RC.sig_g,2)+'\,mag\\\\\n')
    t.write('$m_{r,RC,\\rm{centroid}}$  & '+convert_ndp(RC.r,2)+' $\pm$ '+convert_ndp(RC.sig_r,2)+'\,mag\\\\\n')
    t.write('$m_{i,RC,\\rm{centroid}}$  & '+convert_ndp(RC.i,2)+' $\pm$ '+convert_ndp(RC.sig_i,2)+'\,mag\\\\\n')
    t.write('$(g-r)_{RC,\\rm{centroid}}$ & '+convert_ndp(RC.gr,2)+'  $\pm$ '+convert_ndp(RC.sig_gr,2)+'\,mag\\\\\n')
    t.write('$(r-i)_{RC,\\rm{centroid}}$ & '+convert_ndp(RC.ri,2)+' $\pm$ '+convert_ndp(RC.sig_ri,2)+'\,mag\\\\\n')
    t.write('$A_{g}$ & '+convert_ndp(RC.A_g,3)+' $\pm$ '+convert_ndp(RC.sig_A_g,3)+'\,mag\\\\\n')
    t.write('$A_{r}$ & '+convert_ndp(RC.A_r,3)+' $\pm$ '+convert_ndp(RC.sig_A_r,3)+'\,mag\\\\\n')
    t.write('$A_{i}$ & '+convert_ndp(RC.A_i,3)+' $\pm$ '+convert_ndp(RC.sig_A_i,3)+'\,mag\\\\\n')
    t.write('$E(g-r)$ & '+convert_ndp(RC.Egr,3)+' $\pm$ '+convert_ndp(RC.sig_Egr,3)+'\,mag\\\\\n')
    t.write('$E(r-i)$ & '+convert_ndp(RC.Eri,3)+' $\pm$ '+convert_ndp(RC.sig_Eri,3)+'\,mag\\\\\n')
    t.write('\\hline\n')
    t.write('\\end{tabular}\n')
    t.write('\\end{table}\n')

    t.close()

    log.info('\n')
    log.info('Output red clump data in laTex table to '+file_path)

def convert_ndp(value,ndp):
    """Function to convert a given floating point value to a string,
    rounded to the given number of decimal places, and suffix with zero
    if the value rounds to fewer decimal places than expected"""

    value = str(round(value,ndp))

    dp = value.split('.')[-1]

    while len(dp) < ndp:

        value = value + '0'

        dp = value.split('.')[-1]

    return value
