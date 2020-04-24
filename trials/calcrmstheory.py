##########################################
# IMPORTED MODULES
from numpy import arange, zeros, array, log10, exp
from numpy import sqrt as sqrtarray
from math import pi, sqrt

################################################################################
#     	      	    CALCRMSTHEORY
#
# Version     	Date  	    Author    	      Comments
# 0.1 	      	2012-05-15  Rachel Street     Initial version
################################################################################
def calcrmstheory(minmag,maxmag,ZP,exptime,skybkgd,airmass,aperdiam,G,RDN,teldiam,telheight,debug=True):
    '''Function to calculate the theoretical photometric scatter in lightcurves
    of the apparent magnitude range of the data due to Poisson, readout, sky and
    scintillation noise.  
    Inputs:
      	minmag	    float	  Minimum of the flux range
	maxmag	    float	  Maximum of the flux range
	ZP    	    float     	  Zeropoint magnitude for mag scale
      	exptime     float array   Exposure times [secs] of all frames
	skybkgd     float array   Skybackground flux measurements [ADU]
	airmass     float array   Airmass of each frame
	aperdiam    float     	  Diameter of the photometric aperture [pixels]
	G     	    float     	  CCD Gain [e-/ADU]
	RDN   	    float     	  CCD readout noise [e-]
	teldiam     float     	  Telescope diameter [m]
	telheight   float     	  Telescope altitude above sealevel [m]
    
    Outputs:
      	fluxarray   float array   Flux points for which noise is calculated
	readnoise   float array   Readout noise 
	skynoise    float array   Sky noise
	starnoise   float array   Star noise
	scintnoise  float array   Scintillation noise
	totalnoise  float array   Combined total noise
    '''
    
    # Define constants used in calculations:
    height_o = 8000.0 	  # Scale height of observatory in metres
    
    if debug==True:
	fileobj = open('calcrms.dbg','w')
      	fileobj.write('ZEROPOINT='+str(ZP)+'\n')
    
    # Range of fluxes to calculate theoretical noise for:
    dmag = (maxmag - minmag)/1000.0
    magarray = arange(minmag,maxmag,dmag)
    readnoise = zeros(len(magarray))
    skynoise = zeros(len(magarray))
    starnoise = zeros(len(magarray))
    scintnoise = zeros(len(magarray))
    totalnoise = zeros(len(magarray))
    for j,mag in enumerate(magarray.tolist()):
    
      	# Convert mag to flux [ADU] and convert to [e-].  Calculate the factor used in most noise calculations. 
	flux = (10**((mag-ZP)/-2.5)) * G
      	if debug==True: fileobj.write('FLUX = '+str(flux)+' eqv mag='+str(mag)+'\n')
	if debug==True: fileobj.write('MEAN EXPTIME: '+str(exptime.mean())+'\n')
	if debug==True: fileobj.write('GAIN: '+str(G)+'\n')
	logfactor = 2.5 * (1.0 / flux) * log10(exp(1.0))
      	if debug==True: fileobj.write('LOGFACTOR: '+str(logfactor)+'\n')
	
      	# Calculate the read out noise; RDN [e-] then x logfactor to convert to mags:
      	aperradius = aperdiam/2.0
      	npix_aper = pi*aperradius*aperradius
      	sig_Read = sqrt(RDN*RDN*npix_aper)*logfactor
	var_Read = zeros([len(skybkgd)])
	var_Read[:] = sig_Read*sig_Read
	#var_Read = sig_Read*sig_Read
	invvar = 1.0/var_Read
      	readnoise[j] = 1.0/sqrt( invvar.sum() )
	
      	if debug==True: fileobj.write('READNOISE: aperdiam='+str(aperdiam)+'pix aperrad='+str(aperradius)+'pix Npixels='+str(npix_aper)+'\n')
	if debug==True: fileobj.write('READNOISE: RDN='+str(RDN)+'e- GAIN='+str(G)+' e-/ADU\n')
	if debug==True: fileobj.write('READNOISE summed: '+str(readnoise[j])+'\n')
	if debug==True: fileobj.write('READNOISE per frame: '+str(sig_Read)+'\n')
	
      	# Calculate the sky noise:
	#var_Sky = skybkgd.mean() * G * npix_aper
      	var_Sky = skybkgd * G * npix_aper
	sig_Sky = sqrtarray(var_Sky)*logfactor
	#sig_Sky = sqrt(var_Sky)*logfactor
	var_Sky = sig_Sky*sig_Sky
	if debug==True: fileobj.write('SKYNOISE '+str(sig_Sky.mean())+'  '+str(skybkgd.mean())+'  '+str(G)+'  '+str(npix_aper)+'  '+str(logfactor)+'\n')
	if debug==True: fileobj.write('SKY VARIENCE: '+str(var_Sky.mean())+'\n')
	invvar = 1.0/var_Sky
      	skynoise[j] = 1.0/sqrt( invvar.sum() )
	if debug==True: fileobj.write('SKYNOISE summed:'+str(skynoise[j])+'\n')
	if debug==True: fileobj.write('SKYNOISE mean:'+str(1.0/sqrt( invvar.mean() ))+'\n')
	
      	# Calculate the Poisson noise for the star flux [already in e-]:
      	sig_Star = sqrt(flux)*logfactor
	var_Star = zeros(len(skybkgd))
	var_Star[:] = sig_Star * sig_Star
	#var_Star = sig_Star * sig_Star
	invvar = 1.0/var_Star
      	starnoise[j] = 1.0/sqrt( invvar.sum() )
	if debug==True: 
	    fileobj.write('STARNOISE '+str(flux)+' '+str(logfactor)+' '+str(sig_Star.mean())+'  '+\
	    str(var_Star.mean())+'  '+str(invvar.mean())+'\n')
	    fileobj.write('STARNOISE summed: '+str(starnoise[j])+'\n')
	    fileobj.write('STARNOISE mean: '+str(1.0/sqrt( invvar.mean() ))+'\n')
      	
	# Calculate the scintillation noise [Young 1993Obs...113...41Y, following Gilliland, Brown PASP 104 582 1992]:
	# S = S_o d^-2/3 X^3/2 exp(-h/h_o) (delta f)^1/2 where
	# teldiam must be in [cm] and S_o = 0.09
	# delta f = 1/(2*exptime) in [sec]
	if debug==True: fileobj.write('TELDIAM = '+str(teldiam)+'\n')
	if debug==True: fileobj.write('AIRMASS = '+str(airmass)+'\n')
	if debug==True: fileobj.write('EXPTIME = '+str(exptime)+'\n')
	sig_Scint = 0.09*((teldiam*100.0)**-0.67)*(airmass**1.5)*exp(-telheight/height_o)*((2.0*exptime)**-0.5)
	#sig_Scint = 0.09*((teldiam*100.0)**-0.67)*(airmass.mean()**1.5)*exp(-telheight/height_o)*((2.0*exptime.mean())**-0.5)
	var_Scint = sig_Scint * sig_Scint
	if debug==True: fileobj.write('SCINTILLATION VARIENCE: '+str(var_Scint.mean())+'\n')
	invvar = 1.0/var_Scint
      	scintnoise[j] = 1.0/sqrt( invvar.mean() )
	if debug==True: 
	    fileobj.write('SCINTNOISE summed:'+str(scintnoise[j])+'\n')
	    fileobj.write('SCINTNOISE mean:'+str(1.0/sqrt( invvar.mean() ))+'\n')
    
      	# Calculate the combined noise:
	totalnoise[j] = sqrt( (readnoise[j]*readnoise[j]) + (skynoise[j]*skynoise[j]) + \
	      	(starnoise[j]*starnoise[j]) + (scintnoise[j]*scintnoise[j]) )
	if debug==True: fileobj.write('TOTAL noise: '+str(totalnoise[j])+'\n')
    
    if debug== True: fileobj.close()
    
    return magarray, readnoise, skynoise, starnoise, scintnoise, totalnoise
    
