# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:38:38 2016

@author: ebachelet
"""
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord
 
 
v = Vizier(columns=['_RAJ2000', '_DEJ2000', 'Vmag'],column_filters={'Vmag':'<20'})
result=v.query_region(coord.SkyCoord(ra=270, dec=-28,unit=(u.deg, u.deg),frame='icrs'),width="2400m",catalog=['Tycho'])
RA=[]
DEC=[]
V=[]
for i in result['I/239/hip_main'] :
    RA.append(i['_RAJ2000'])
    DEC.append(i['_DEJ2000'])
    V.append(i['Vmag'])

import pdb; pdb.set_trace()
