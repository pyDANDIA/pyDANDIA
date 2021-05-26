import numpy as np

file = '/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_colour_photometry.txt'
file_lines = open(file, 'r').readlines()

gmin = 15.0
gmax = 18.0
gimin = 2.5
gimax = 3.5

data = []
for line in file_lines:
    if line[0:1] != '#':
        entries = line.replace('\n','').split()
        row = []
        for item in entries:
            row.append(float(item))
        data.append(row)
data = np.array(data)

idx1 = np.where(np.logical_and(np.greater_equal(data[:,3], gmin), np.less_equal(data[:,3], gmax)))[0]
idx2 = np.where(np.logical_and(np.greater_equal(data[:,9], gimin), np.less_equal(data[:,9], gimax)))[0]
idx = list(set(idx1).intersection(set(idx2)))

f = open('/Users/rstreet1/ROMEREA/test_data/parallel_ms_stars.txt', 'w')
f.write('# All measured floating point quantities in units of magnitude\n')
f.write('# Selected indicates whether a star lies within the selection radius of a given location, if any.  1=true, 0=false\n')
f.write('# Field_ID   ra_deg   dec_deg   g  sigma_g    r  sigma_r    i  sigma_i   (g-i)  sigma(g-i) (g-r)  sigma(g-r)  (r-i) sigma(r-i)  Selected  Gaia_ID parallax parallax_error proper_motion\n')

for j in range(0,len(idx),1):
    row = str(int(data[idx[j],0]))
    for item in data[idx[j],1:15]:
        row = row + ' ' + str(item)
    row = row + ' ' + str(int(data[idx[j],15]))
    row = row + ' ' + str(int(data[idx[j],16]))
    for item in data[idx[j],17:]:
        row = row + ' ' + str(item)
    f.write(row+'\n')

f.close()
