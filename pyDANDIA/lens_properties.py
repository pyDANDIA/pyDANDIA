####################################################################################
# CALC ANGULAR EINSTEIN RADIUS

# Import modules:
import math
import numpy as np
from astropy import constants
#import pyslalib

class Lens:
    """Class describing the properties of a lensing body"""

    def __init__(self):

        self.tE = None
        self.sig_tE = None
        self.t0 = None
        self.sig_t0 = None
        self.rho = None
        self.sig_rho = None
        self.pi_E = None
        self.sig_pi_E = None

    def rads_to_muas(self,value,sigma):

        value = (value * (180.0/np.pi)) * 3600.0 * 1e6
        sigma = (sigma * (180.0/np.pi)) * 3600.0 * 1e6

        return value, sigma

    def muas_to_rads(self,value,sigma):

        value = ((value / 1e6) / 3600.0)*(np.pi/180.0)
        sigma = ((sigma / 1e6) / 3600.0)*(np.pi/180.0)

        return value, sigma

    def calc_angular_einstein_radius(self,thetaS,sig_thetaS,log=None):
        """Function to calculate the angular Einstein radius and geometric projected
        motion while propagating errors formally."""

        if log!=None:
            log.info('\n')
            log.info('Calculating the Einstein radius')

        # Convert tE in days -> years
        tE = self.tE / 365.25
        sig_tE = self.sig_tE / 365.25

        # pi_E is a vector of N,E components. Its magnitude is required later on:
        (pi_E_mag,sig_pi_E_mag) = self.calc_mag_piE()

        if log!=None:
            log.info('Magnitude of parallax vector: '+str(pi_E_mag)+'+/-'+str(sig_pi_E_mag))
            log.info('Source angular radius: '+str(thetaS)+' +/- '+str(sig_thetaS)+' muas')
            log.info('Rho: '+str(self.rho)+' +/- '+str(self.sig_rho))
        else:
            print('Magnitude of parallax vector: '+str(pi_E_mag)+'+/-'+str(sig_pi_E_mag))

        if log!= None:
            log.info('Source angular radius = '+str(thetaS)+'+/-'+str(sig_thetaS)+' microarcsec')

        # Calculate fractional errors for use in propagating errors later:
        fthetaS = sig_thetaS / thetaS
        frho = self.sig_rho / self.rho
        ftE = self.sig_tE / self.tE
        fpiE = sig_pi_E_mag / pi_E_mag

        # Calculate the angular Einstein radius:
        self.thetaE = thetaS / self.rho
        fthetaE = math.sqrt( fthetaS*fthetaS + frho*frho )
        self.sig_thetaE = fthetaE * self.thetaE

        # Lens-source relative parallax: THIS PI_E IS NOT A VECTOR
        self.pi_rel = self.thetaE * pi_E_mag
        fthetaE = self.sig_thetaE/self.thetaE
        self.sig_pi_rel = math.sqrt( (fthetaE*fthetaE) + \
                                    (fpiE*fpiE) ) * self.pi_rel

        if log!= None:
            log.info('Angular Einstein radius = '+str(self.thetaE)+'+/-'+str(self.sig_thetaE)+' microarcsec')
        else:
            print('Angular Einstein radius = '+str(self.thetaE)+'+/-'+str(self.sig_thetaE)+' microarcsec')

        if log!=None:
            log.info('Lens-source relative parallax: ' + str(self.pi_rel)+'+/-'+str(self.sig_pi_rel)+' microarcsec')
        else:
            print('Lens-source relative parallax: ' + str(self.pi_rel)+'+/-'+str(self.sig_pi_rel)+' microarcsec')

                # Projected velocities:  pi_E IS A VECTOR
        def calc_v_geo( t, pi, pi_mag ):
            """t must be in seconds, pi dimensionless"""

            v = ( au_km / t ) * ( pi / (pi_mag*pi_mag) )

            return v

    def calc_mag_piE(self):

        pi_E_mag = math.sqrt( (self.pi_E*self.pi_E).sum() )
        sig_pi_E_mag = math.sqrt( (self.sig_pi_E*self.sig_pi_E).sum() )

        return pi_E_mag, sig_pi_E_mag

    def calc_helio_veliocity():

        #earth_position = pyslalib.slalib.sla_epv(self.t0par-2400000.0)

        v_earth = earth_position[1]     # Earth's heliocentric velocity vector

        au_km = constants.au.value / 1000.0
        tE_s = self.tE * 365.25 * 24.0 * 60.0 * 60.0
        sig_tE_s = self.sig_tE * 365.25 * 24.0 * 60.0 * 60.0
        v_geo = []
        sig_v_geo = []

        # Calculate mu_geo:
        self.mu_geo = self.thetaE / tE
        self.sig_mu_geo = math.sqrt( fthetaE*fthetaE + ftE*ftE ) * self.mu_geo


        for i in range(0,2,1):
            v = calc_v_geo( tE_s, self.pi_E[i], pi_E_mag )
            v_min = calc_v_geo( (tE_s-sig_tE_s), (self.pi_E[i]-self.sig_pi_E[i]), (pi_E_mag-sig_pi_E_mag) )
            v_max = calc_v_geo( (tE_s+sig_tE_s), (self.pi_E[i]+self.sig_pi_E[i]), (pi_E_mag+sig_pi_E_mag) )
            sig_v = (v_max - v_min) / 2.0
            v_geo.append(v)
            sig_v_geo.append(sig_v)

        if log!=None:
            log.info('Projected velocity in the geocentric frame (N) [kms^-1]: '+str(v_geo[0])+'+/-'+str(sig_v_geo[0]))
            log.info('Projected velocity in the geocentric frame (E) [kms^-1]: '+str(v_geo[1])+'+/-'+str(sig_v_geo[1]))
        else:
            print('Projected velocity in the geocentric frame (N) [kms^-1]: '+str(v_geo[0])+'+/-'+str(sig_v_geo[0]))
            print('Projected velocity in the geocentric frame (E) [kms^-1]: '+str(v_geo[1])+'+/-'+str(sig_v_geo[1]))

        # Heliocentric velocity: v_earth IS A VECTOR
        # Error on Earth's position is assumed to be neglible compared with the error on v_geo
        v_hel = v_geo + v_earth
        sig_v_hel = sig_v_geo

        if log!=None:
            log.info('Projected velocity in the heliocentric frame (N): '+str(v_hel[0])+'+/-'+str(sig_v_hel[0]))
            log.info('Projected velocity in the heliocentric frame (E): '+str(v_hel[1])+'+/-'+str(sig_v_hel[1]))
        else:
            print('Projected velocity in the heliocentric frame (N): '+str(v_hel[0])+'+/-'+str(sig_v_hel[0]))
            print('Projected velocity in the heliocentric frame (E): '+str(v_hel[1])+'+/-'+str(sig_v_hel[1]))

    def calc_distance(self,DS,sig_DS,log):
        """Function to calculate the distance to the source"""

        (pi_E_mag,sig_pi_E_mag) = self.calc_mag_piE()

        (thetaE,sig_thetaE) = self.muas_to_rads(self.thetaE,self.sig_thetaE)

        (pi_rel,sig_pi_rel) = self.muas_to_rads(self.pi_rel,self.sig_pi_rel)

        DS_m = DS * 1000.0 * constants.pc.value     # kpc -> m
        sig_DS_m = sig_DS * 1000.0 * constants.pc.value

        pi_S = constants.au.value / DS_m
        if sig_DS > 0.0:
            sig_pi_S = constants.au.value / sig_DS
        else:
            sig_pi_S = 0.0

        pi_L = pi_rel + pi_S                    # rads
        sig_pi_L = np.sqrt( sig_pi_rel*sig_pi_rel + sig_pi_S*sig_pi_S )


        DL = constants.au.value / pi_L
        sig_DL = (sig_pi_L / pi_L) * DL

        if log!=None:
            log.info('Distance to the lens: '+str(DL)+' +/- '+str(sig_DL)+' m')

        self.D = DL / constants.pc.value / 1000.0
        self.sig_D = sig_DL / constants.pc.value / 1000.0

        if log!=None:
            log.info('Distance to the lens: '+str(self.D)+' +/- '+\
                                         str(self.sig_D)+' kpc')

    def calc_distance_modulus(self,log):
        """Method to calculate the distance modulus to the lens, given
        its distance from the observer
        Note: assumes input distance is in kiloparsecs
        """

        self.dist_mod = 5.0 * np.log10(self.D * 1000.0) - 5.0

        self.sig_dist_mod = (self.sig_D/self.D)*self.dist_mod

        if log!=None:
            log.info('Lens distance modulus = '+str(self.dist_mod)+\
                                        ' +/- '+str(self.sig_dist_mod))

    def calc_einstein_radius(self,log):
        """Function to calculate the Einstein radius in physical units"""

        (thetaE, sig_thetaE) = self.muas_to_rads(self.thetaE, self.sig_thetaE)

        delta_thetaE = np.cos(thetaE) * sig_thetaE

        # Kpc -> AU
        DL = self.D * 1000.0 * constants.pc.value / constants.au.value
        sig_DL = self.sig_D * 1000.0 * constants.pc.value / constants.au.value

        self.RE = DL * np.sin(thetaE)
        self.sig_RE = np.sqrt( (sig_DL/DL)*(sig_DL/DL) + \
                        (delta_thetaE/thetaE)*(delta_thetaE/thetaE) ) * self.RE

        if log!=None:
            log.info('Einstein radius = '+str(self.RE)+' +/- '+str(self.sig_RE)+' AU')

    def calc_masses(self,log):
        """Function to calculate the component masses of a binary lens"""

        (pi_E_mag,sig_pi_E_mag) = self.calc_mag_piE()

        (thetaE, sig_thetaE) = self.muas_to_rads(self.thetaE, self.sig_thetaE)
#        thetaE = (self.thetaE / 1e6 / 3600.0) * (np.pi/180.0)        # mu-as -> rads
#        sig_thetaE = (self.sig_thetaE / 1e6 / 3600.0) * (np.pi/180.0)

        if log!=None:
            log.info('Magnitude of pi_E: '+str(pi_E_mag)+' +/- '+str(sig_pi_E_mag))
            log.info('Theta_E: '+str(thetaE)+' +/- '+str(sig_thetaE)+' rads')

        kappa = ((constants.c*constants.c * constants.au)/(4.0*constants.G)).value

        self.ML = (kappa * ( thetaE / pi_E_mag ))
        dthetaE = sig_thetaE/thetaE
        dpiE = sig_pi_E_mag/pi_E_mag
        self.sig_ML = np.sqrt( dthetaE*dthetaE + dpiE*dpiE ) * self.ML
        self.ML = self.ML / constants.M_sun.value
        self.sig_ML = self.sig_ML / constants.M_sun.value

        if 'q' in dir(self):
            self.M1 = self.ML * ( 1.0 / (1.0 + self.q) )
            self.sig_M1 = (self.sig_ML/self.ML)*self.M1

            self.M2 = self.ML * ( self.q / (1.0 + self.q) )
            self.sig_M2 = (self.sig_ML/self.ML)*self.M2

        if log!=None:
            log.info('Total lens mass = '+str(self.ML)+' +/- '+str(self.sig_ML)+' Msol')
            if 'q' in dir(self):
                log.info('M1 mass = '+str(self.M1)+' +/- '+str(self.sig_M1)+' Msol')
                log.info('M2 mass = '+str(self.M2)+' +/- '+str(self.sig_M2)+' Msol')

    def calc_projected_separation(self,log):
        """Method to calculate the projected separation of the components of a
        binary lens at the time of an event"""

        (thetaE, sig_thetaE) = self.muas_to_rads(self.thetaE, self.sig_thetaE)

        # D is in kpc -> AU, s is in units of thetaE
        DL = self.D * 1000.0 * constants.pc.value / constants.au.value
        sig_DL = self.sig_D * 1000.0 * constants.pc.value / constants.au.value

        self.a_proj = self.s * DL * thetaE

        self.sig_a_proj = np.sqrt( (self.sig_s/self.s)*(self.sig_s/self.s) + \
                            (sig_DL/DL)*(sig_DL/DL) + \
                              (sig_thetaE/thetaE)*(sig_thetaE/thetaE) ) * self.a_proj

        if log!=None:
            log.info('Projected separation of lens masses = '+\
                    str(self.a_proj)+' +/- '+str(self.sig_a_proj)+' AU')

    def calc_orbital_energies(self,log):
        """Method to calculate the ratio of the kinetic and potential energy
        of the binary orbit as a test of whether the object could be bound"""

        sre = self.s * self.RE

        sig_sre = np.sqrt( (self.sig_s/self.s)*(self.sig_s/self.s) + \
                    (self.sig_RE/self.RE)*(self.sig_RE/self.RE) ) * sre

        # Units d^-1 -> year^-2
        dsdt = self.dsdt * 365.24
        sig_dsdt = self.sig_dsdt * 365.24
        dadt = self.dalphadt * 365.24
        sig_dadt = self.sig_dalphadt * 365.24

        gamma_sq = (dsdt / self.s)*(dsdt / self.s) + \
                    (dadt*dadt)

        sig_gamma_sq = np.sqrt( (sig_dsdt/dsdt)*(sig_dsdt/dsdt) + \
                        (self.sig_s/self.s)*(self.sig_s/self.s) + \
                         (sig_dadt/dadt)*(sig_dadt/dadt) ) * gamma_sq

        self.kepe = (sre*sre*sre * gamma_sq) / ( 8.0 * np.pi*np.pi * self.ML )

        self.sig_kepe = np.sqrt( (3 * (sig_sre/sre))*(3 * (sig_sre/sre)) + \
                    (2*sig_gamma_sq/gamma_sq)*(2*sig_gamma_sq/gamma_sq) + \
                      (self.sig_ML/self.ML)*(self.sig_ML/self.ML) ) * self.kepe

        if log!=None:
            log.info('Binary lens ratio of KE/PE = '+\
                    str(self.kepe)+' +/- '+str(self.sig_kepe))

    def calc_rel_proper_motion(self,log=None):

        self.mu_rel = self.thetaE / self.tE
        self.sig_mu_rel = np.sqrt((self.sig_thetaE/self.thetaE)**2 + \
                                    (self.sig_tE/self.tE)**2) * self.mu_rel

        self.mu_rel = self.mu_rel * 365.24 / 1000.0
        self.sig_mu_rel = self.sig_mu_rel * 365.24 / 1000.0

        if log!=None:
            log.info('Relative source-lens proper motion = '+\
                    str(self.mu_rel)+' +/- '+str(self.sig_mu_rel)+'mas yr^-1')

if __name__ == '__main__':

    l = Lens()

    thetaS = float(raw_input('Please enter theta_S [mas]: '))
    sig_thetaS = float(raw_input('Please enter the uncertainty on theta_S [mas]: '))
    l.rho = float(raw_input('Please enter rho: '))
    l.sig_rho = float(raw_input('Please enter the uncertainty on rho: '))
    l.tE = float(raw_input('Please enter tE [days]: '))
    l.sig_tE = float(raw_input('Please enter the uncertainty on tE [days]: '))
    pi_E_N = float(raw_input('Please enter pi_E (N): '))
    sig_pi_E_N = float(raw_input('Please enter the uncertainty on pi_E (N): '))
    pi_E_E = float(raw_input('Please enter pi_E (E): '))
    sig_pi_E_E = float(raw_input('Please enter the uncertainty on pi_E (E): '))
    l.pi_E = np.array( [pi_E_N, pi_E_E] )
    l.sig_pi_E = np.array( [sig_pi_E_N, sig_pi_E_E] )
    ve1 = float(raw_input('Please enter the Earth N velocity: '))
    ve2 = float(raw_input('Please enter the Earth E velocity: '))
    v_earth = np.array( [ve1, ve2] )

    l.calc_angular_einstein_radius(thetaS,sig_thetaS,v_earth)
