import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
import os
from scipy.odr import *

def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]
    
    
def gp_ip_from_G_BP_RP(G,BP_RP,delta_G=0,delta_BPRP=0):
#https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
    i = G+0.29676-0.64728*BP_RP+0.10141*BP_RP**2
    ei = (delta_G**2+0.64728**2*delta_BPRP**2+4*0.10141**2*BP_RP**2*delta_BPRP**2+0.098957**2)**0.5
    g = G-0.135189+0.46245*BP_RP+0.25171*BP_RP**2-0.021349*BP_RP**3
    eg =  (delta_G**2+0.46245**2*delta_BPRP**2+4*0.25171**2*BP_RP**2*delta_BPRP**2+9*0.021349**2*BP_RP**4*delta_BPRP**2+0.16497**2)**0.5
    return g,eg,i,ei

Vizier.ROW_LIMIT = -1

radius = 5 #arcmin
dataset = 'COJB_gp.dat'
fff = np.loadtxt('../'+dataset,dtype=str)

meta_file = 'pyDANDIA_metadata_'+dataset[:-4]+'.fits'
band = meta_file.split('.')[0][-2:]

if band=='gp':

    kkk = np.loadtxt('../LCO_g_coj1m011.dat',dtype=str)
else:

    kkk = np.loadtxt('../LCO_i_coj1m003.dat',dtype=str)
all_data = [i for i in os.listdir('../') if ('_gp.dat' in i) | ('_ip.dat' in i)]



metadata = fits.open(meta_file)


Stars = metadata[9].data
star_index = ((184.61816-Stars['ra'])**2 +(-63.49726-Stars['dec'])**2).argmin()

ra = Stars[star_index]['ra']
dec = Stars[star_index]['dec']
mask = (Stars['ra']-ra)**2+(Stars['dec']-dec)**2<(radius/60)**2


stars = Stars[mask]


center = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')

#result = Vizier.query_region(center,radius=radius*u.arcmin,catalog='I/345/gaia2',column_filters={'Gmag': '<19'})[0]
result = Vizier.query_region(center,radius=radius*u.arcmin,catalog='J/ApJ/867/105/refcat2')[0]

x = np.arange(0,25,0.1)
match = []
for ind in range(len(stars)):

    dist = (stars[ind]['ra']-result['RA_ICRS'])**2+(stars[ind]['dec']-result['DE_ICRS'])**2
    
    if dist.min()<(0.25/3600)**2:
        match.append([ind,dist.argmin()])
        
        if (stars[ind]['ra']==ra) & (stars[ind]['dec']==dec):
            target = len(match)-1
match = np.array(match)
#e_BPRP = (result[match[:,1]]['e_BPmag'].data.data**2+result[match[:,1]]['e_RPmag'].data.data**2)**0.5
#catalog_g,ecatalog_g,catalog_i,ecatalog_i = gp_ip_from_G_BP_RP(result[match[:,1]]['Gmag'].data.data,result[match[:,1]]['BP-RP'].data.data,result[match[:,1]]['e_Gmag'].data.data,e_BPRP)

#for data in all_data:
#    #import pdb; pdb.set_trace()
#    dat = np.loadtxt('../'+data,dtype=str)
#    plt.scatter(dat[:,0].astype(float)-2450000,dat[:,1].astype(float),label=data)
#    print(data,np.max(dat[:,1].astype(float)))
#    np.savetxt(data,dat[:,:3], delimiter=" ", header="# HJD    Instrumental mag, mag_error", 
#           fmt="%s", comments='')
#plt.legend()
#plt.gca().invert_yaxis()
#plt.show()


fstars = stars[match[:,0]]
catalog_i = result[match[:,1]]['imag'] 
catalog_g = result[match[:,1]]['gmag'] 
ecatalog_i = np.array([0.1]*len(result[match[:,1]]['imag'] ))
ecatalog_g = np.array([0.037]*len(result[match[:,1]]['imag'] ))
#import pdb; pdb.set_trace()

if band=='ip':
    try:
        mask = (~np.isnan(catalog_i)) & (catalog_i<17.5) & (np.arange(0,len(catalog_i)) != target)
    except:
        mask = (~np.isnan(catalog_i)) & (catalog_i<17.5) 
    calib,cov_calib = np.polyfit(fstars['ref_mag'][mask],catalog_i[mask],1,cov=True)
    linear = Model(f)

    mydata = Data(fstars['ref_mag'][mask], catalog_i[mask])
    myodr = ODR(mydata, linear, beta0=[1., 2.])
    myoutput = myodr.run()
    plt.scatter(fstars['ref_mag'][mask], catalog_i[mask])
    plt.plot(x,calib[0]*x+calib[1])
    plt.plot(x,myoutput.beta[0]*x+myoutput.beta[1])
    plt.show()
    aa,bb = np.polyfit(fstars['ref_flux'][mask], 10**((25-catalog_i[mask])/2.5),1)
 
else:
    try:
        mask = (~np.isnan(catalog_g)) & (catalog_g<18.5) & (np.arange(0,len(catalog_i)) != target)
    except:
        mask = (~np.isnan(catalog_g)) & (catalog_g<18.5)
        
    calib,cov_calib = np.polyfit(fstars['ref_mag'][mask],catalog_g[mask],1,cov=True)
    linear = Model(f)

    mydata = Data(fstars['ref_mag'][mask], catalog_g[mask])

    myodr = ODR(mydata, linear, beta0=[1., 2.])
    myoutput = myodr.run()
    plt.scatter(fstars['ref_mag'][mask], catalog_g[mask])
    plt.plot(x,calib[0]*x+calib[1])
    plt.plot(x,myoutput.beta[0]*x+myoutput.beta[1])
    plt.show()
    aa,bb,cc = np.polyfit(fstars['ref_flux'][mask], 10**((25-catalog_g[mask])/2.5),2)
    
#import pdb; pdb.set_trace()
#myoutput.pprint()

#plt.scatter(fff[:,0].astype(float)-2450000,fff[:,1].astype(float)*calib[0]+calib[1])
#plt.scatter(fff[:,0].astype(float)-2450000,fff[:,1].astype(float)*myoutput.beta[0]+myoutput.beta[1])




# calibrate
ccalib = np.eye(3)
#ccalib[:2,:2] = cov_calib
#ccalib[2,2] = calib[0]**2
jac = np.c_[fff[:,1].astype(float),[1]*len(fff),fff[:,2].astype(float)]

#cov_calib = myoutput.cov_beta*myoutput.res_var
#calib = myoutput.beta
#m_prime = fff[:,1].astype(float)*calib[0]+calib[1]
#res = jac@np.dot(ccalib,jac.T)
#em_prime = res.diagonal()**0.5

ccalib[:2,:2] = cov_calib
ccalib[2,2] = calib[0]**2
jac = np.c_[fff[:,1].astype(float),[1]*len(fff),fff[:,2].astype(float)]
m_prime = fff[:,1].astype(float)*calib[0]+calib[1]
res = jac@np.dot(ccalib,jac.T)
em_prime = res.diagonal()**0.5

plt.errorbar(fff[:,0].astype(float)-2450000,fff[:,1].astype(float),fmt='.')
plt.errorbar(fff[:,0].astype(float)-2450000,m_prime,em_prime,fmt='.')
plt.errorbar(fff[:,0].astype(float)-2450000,fff[:,1].astype(float)*myoutput.beta[0]+myoutput.beta[1],em_prime,fmt='.')
plt.errorbar(kkk[:,0].astype(float)-2450000,kkk[:,1].astype(float),kkk[:,2].astype(float),fmt='.')


try:
    
    diff_flux = 10**((25-fff[:,1].astype(float))/2.5)-fstars['ref_flux'][target]


    new_mag = 25-2.5*np.log10((fstars['ref_flux'][target]+diff_flux)**2*aa+(fstars['ref_flux'][target]+diff_flux)*bb+cc)
    plt.scatter(fff[:,0].astype(float)-2450000,new_mag,marker='^')
except:
   
    diff_flux = 10**((25-fff[:,1].astype(float))/2.5)-Stars[star_index]['ref_flux']


    new_mag = 25-2.5*np.log10((Stars[star_index]['ref_flux']+diff_flux)**2*aa+bb*(Stars[star_index]['ref_flux']+diff_flux)+cc)
    plt.scatter(fff[:,0].astype(float)-2450000,new_mag,marker='^')
    
plt.gca().invert_yaxis()
plt.show()

fff[:,3:5] = np.c_[m_prime,em_prime]
np.savetxt(dataset[:-4]+'.calib', fff, delimiter=" ", header="# HJD    Instrumental mag, mag_error   Calibrated mag, mag_error", 
           fmt="%s", comments='')
import pdb; pdb.set_trace()
