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


    G_g = 0.13518 + -0.46245*BPRPmag + -0.25171*BPRPmag*BPRPmag + \
                    0.021349*BPRPmag*BPRPmag*BPRPmag
    Gg_err = 0.16497

    G_r = -0.12879 + 0.24662*BPRPmag + -0.027464*BPRPmag*BPRPmag + \
                    -0.049465*BPRPmag*BPRPmag*BPRPmag
    Gr_err = 0.066739

    G_i = -0.29676 + 0.64728*BPRPmag + -0.10141*BPRPmag*BPRPmag
    Gi_err = 0.098957

    phot = {}

    phot['g'] = Column(data=(Gmag - G_g), name='gmag')
    phot['g_err'] = Column(data=(np.sqrt( (Gmerr*Gmerr) + (Gg_err*Gg_err) )),
                            name='gmag_err')

    phot['r'] = Column(data=(Gmag - G_r), name='rmag')
    phot['r_err'] = Column(data=(np.sqrt( (Gmerr*Gmerr) + (Gr_err*Gr_err) )),
                            name='rmag_err')

    phot['i'] = Column(data=(Gmag - G_i), name='imag')
    phot['i_err'] = Column(data=(np.sqrt( (Gmerr*Gmerr) + (Gi_err*Gi_err) )),
                            name='imag_err')

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
