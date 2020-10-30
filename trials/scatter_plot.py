import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib	import pyplot as plt

data = np.loadtxt('test_quality.dat',dtype=np.str)

#dataset = pd.DataFrame({'SITE':data[:,9].astype(np.str),'MEAN_LC_MAG':data[:,0].astype(np.float),'MEDIAN_LC_MAG':data[:,1].astype(np.float),'RMS_LC_MAG':data[:,2].astype(np.float),'REPORTED_ERR_LC_MAG':data[:,3].astype(np.float),'N_DATA':data[:,4].astype(np.float),'MEDIAN_PSCALE':data[:,5].astype(np.float),'MEDIAN_PSCALE_ERR':data[:,6].astype(np.float),'MEDIAN_BKG':data[:,7].astype(np.float),'MEDIAN_BKG_ERR':data[:,8].astype(np.float) })

dataset = pd.DataFrame({'SITE':data[:,9].astype(np.str),'MEDIAN_LC_MAG':data[:,1].astype(np.float),'LOG10_RMS_LC_MAG':np.log10(data[:,2].astype(np.float)),'LOG10_REPORTED_ERR_LC_MAG':np.log10(data[:,3].astype(np.float)),'N_DATA':data[:,4].astype(np.float),'MEDIAN_PSCALE':data[:,5].astype(np.float),'MEDIAN_BKG':data[:,7].astype(np.float)})

#'MEDIAN_LC_MAG':,'BAND':data[:,4],'FIELD':data[:,5],'FWHM':data[:,7].astype(np.float),'YEAR':data[:,8].astype(np.float)})


dataset['SITE'] = dataset['SITE'].astype('category')
#dataset['YEAR'] = dataset['YEAR'].astype('category')
#dataset['INSTRUMENT'] = dataset['INSTRUMENT'].astype('category')

sns.set(style="whitegrid")
sns.set(font_scale=3)
#ax = sns.violinplot(y = dataset["FWHM"],x = dataset["SITEINST"],  hue = dataset["YEAR"], inner="quartile")
#plt.show()
#scale="count", # if you don't need normalized kernel density violins...

sns.set(style="ticks")

sns.pairplot(dataset, hue="SITE")

#g = sns.PairGrid(dataset, diag_sharey=False, hue = "SITE")
#g.map_upper(sns.scatterplot)
#g.map_lower(sns.kdeplot, colors="C0")
#g.map_diag(sns.kdeplot, lw=2)
plt.savefig('scatter.png')

