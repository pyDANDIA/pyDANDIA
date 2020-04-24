import numpy as np

def gaia_flux_to_mag(Gflux, Gferr, passband="G"):
    """Function to convert Gaia flux measurements to photometric magnitudes
    on the VEGAMAG system, using the zeropoints produced from Gaia
    Data Release 2 published in Evans, D.W. et al., 2018, A&A, 616, A4.

    Passband options are:
        G       (default)   ZP=25.6884 +/- 0.0018
        G_bp                ZP=25.3514 +/- 0.0014
        G_rp                ZP=24.7619 +/- 0.0019
    """

    if passband == "G":
        ZP = 25.6884
        sigZP = 0.0018
    elif passband == 'G_bp':
        ZP = 25.3514
        sigZP = 0.0014
    elif passband == 'G_rp':
        ZP = 24.7619
        sigZP = 0.0019
    else:
        raise ValueError('No Gaia photometric transform available for passband '+passband)

    Gmag = ZP - 2.5 * log10(Gflux)
    Gmerr = (2.5 / np.log(10.0)) * Gferr / Gflux

    return Gmag, Gmerr
