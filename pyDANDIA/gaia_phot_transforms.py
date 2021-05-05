from sys import argv
import numpy as np
from scipy import optimize
from astropy.table import Table, Column

def gaia_flux_to_mag(Gflux, Gferr, passband="G"):
    """Function to convert Gaia flux measurements to photometric magnitudes
    on the VEGAMAG system, using the zeropoints produced from Gaia
    Data Release 2 published in Evans, D.W. et al., 2018, A&A, 616, A4.

    Passband options are:
        G       (default)   ZP=25.6884 +/- 0.0018
        G_BP                ZP=25.3514 +/- 0.0014
        G_RP                ZP=24.7619 +/- 0.0019
    """

    def phot_conversion(flux, ferr):
        mag = ZP - 2.5 * np.log10(flux)
        merr = (2.5 / np.log(10.0)) * ferr / flux
        return mag, merr

    if passband == "G":
        ZP = 25.6884
        sigZP = 0.0018
    elif passband == 'G_BP':
        ZP = 25.3514
        sigZP = 0.0014
    elif passband == 'G_RP':
        ZP = 24.7619
        sigZP = 0.0019
    else:
        raise ValueError('No Gaia photometric transform available for passband '+passband)

    if type(Gflux) == type(Column()):

        idx = np.where(Gflux > 0.0)
        Gmag = np.zeros(len(Gflux))
        Gmerr = np.zeros(len(Gflux))

        (Gmag[idx],Gmerr[idx]) = phot_conversion(Gflux[idx], Gferr[idx])

    else:

        if Gflux > 0.0:
            (Gmag,Gmerr) = phot_conversion(Gflux, Gferr)
        else:
            Gmag = 0.0
            Gmerr = 0.0

    return Gmag, Gmerr

def calc_gaia_colours(BPmag, BPmag_err, RPmag, RPmag_err):
    """Function to calculate the Gaia BP-RP colours and uncertainties"""

    BP_RP = Column(data=(BPmag - RPmag), name='BP-RP')

    BPRP_err = Column(data=(np.sqrt( (BPmag_err*BPmag_err)  + \
                            (RPmag_err*RPmag_err) ) ), name='BP-RP_err')

    return BP_RP, BPRP_err

def transform_gaia_phot_to_SDSS(Gmag, Gmerr, BPRPmag, BPRPerr):
    """Function using the conversion transformations published in
    Evans, D.W. et al., 2018, A&A, 616, A4 to calculate the SDSS
    g, r and i magnitudes based on the Gaia photometry."""

    g_coeff = [0.13518, -0.46245, -0.25171, 0.021349]
    Gg_err = 0.16497
    r_coeff = [-0.12879, 0.24662, -0.027464, -0.049465]
    Gr_err = 0.066739
    i_coeff = [-0.29676, 0.64728, -0.10141]
    Gi_err = 0.098957

    G_g = Gmag - (g_coeff[0] + g_coeff[1]*BPRPmag + g_coeff[2]*BPRPmag**2 + g_coeff[3]*BPRPmag**3)
    g_err = np.sqrt( (Gmerr**2 + g_coeff[1]**2)*BPRPerr**2 + (4*g_coeff[2]**2*BPRPmag**2*BPRPerr**2) + (9*g_coeff[3]**2*BPRPmag**4*BPRPerr**2) + (Gg_err**2) )
    G_r = Gmag - (r_coeff[0] + r_coeff[1]*BPRPmag + r_coeff[2]*BPRPmag**2 + r_coeff[3]*BPRPmag**3)
    r_err = np.sqrt( (Gmerr**2 + r_coeff[1]**2)*BPRPerr**2 + (4*r_coeff[2]**2*BPRPmag**2*BPRPerr**2) + (9*r_coeff[3]**2*BPRPmag**4*BPRPerr**2) + (Gr_err**2) )
    G_i = Gmag - (i_coeff[0] + i_coeff[1]*BPRPmag + i_coeff[2]*BPRPmag**2)
    i_err = np.sqrt( (Gmerr**2 + i_coeff[1]**2)*BPRPerr**2 + (4*i_coeff[2]**2*BPRPmag**2*BPRPerr**2) + (Gi_err**2) )

    phot = {}

    phot['g'] = Column(data=G_g, name='gmag')
    phot['g_err'] = Column(data=g_err, name='gmag_err')

    phot['r'] = Column(data=G_r, name='rmag')
    phot['r_err'] = Column(data=r_err, name='rmag_err')

    phot['i'] = Column(data=G_i, name='imag')
    phot['i_err'] = Column(data=i_err, name='imag_err')

    return phot

def transform_gaia_phot_to_JohnsonCousins(Gmag, Gmerr, BPRPmag, BPRPerr):
    """Function using the conversion transformations published in
    Evans, D.W. et al., 2018, A&A, 616, A4 to calculate the Johnson-Cousins
    V, R and I magnitudes based on the Gaia photometry."""

    G_V = -0.01760 + -0.006860*BPRPmag + -0.1732*BPRPmag*BPRPmag
    GV_err = 0.045858

    G_R = -0.003226 + 0.3833*BPRPmag + -0.1345*BPRPmag*BPRPmag
    GR_err = 0.04840

    G_I = 0.02085 + 0.7419*BPRPmag + -0.09631*BPRPmag*BPRPmag
    GI_err = 0.04956

    phot = {}

    phot['V'] = Column(data=(Gmag - G_V), name='Vmag')
    phot['V_err'] = Column(data=(np.sqrt( (Gmerr*Gmerr) + (GV_err*GV_err) )),
                            name='Vmag_err')

    phot['R'] = Column(data=(Gmag - G_R), name='Rmag')
    phot['R_err'] = Column(data=(np.sqrt( (Gmerr*Gmerr) + (GR_err*GR_err) )),
                            name='Rmag_err')

    phot['I'] = Column(data=(Gmag - G_I), name='Imag')
    phot['I_err'] = Column(data=(np.sqrt( (Gmerr*Gmerr) + (GI_err*GI_err) )),
                            name='Imag_err')

    return phot
