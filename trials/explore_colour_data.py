import numpy as np
import matplotlib.pyplot as plt

def analyze_parallel_ms_stars():

    data = read_data()

    plot_spacial_distribution(data)
    plot_phot_quality(data)
    plot_parallax(data)

def read_data():
    file = '/Users/rstreet1/ROMEREA/test_data/parallel_ms_stars.txt'
    file_lines = open(file, 'r').readlines()

    data = []
    for line in file_lines:
        if line[0:1] != '#':
            entries = line.replace('\n','').split()
            row = []
            for item in entries:
                row.append(float(item))
            data.append(row)
    data = np.array(data)
    return data

def plot_spacial_distribution(data):
    fig = plt.figure(1)

    plt.plot(data[:,1], data[:,2], 'k.')

    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    plt.grid()

    plt.savefig('/Users/rstreet1/ROMEREA/test_data/space_distro_parallel_MS_stars.png', bbox_inches='tight')
    plt.close(1)

def plot_phot_quality(data):
    fig = plt.figure(1)

    plt.plot(data[:,3], data[:,4], 'b.', label='g-band data')
    plt.plot(data[:,5], data[:,6], 'g.', label='r-band data')
    plt.plot(data[:,7], data[:,8], 'r.', label='i-band data')

    plt.xlabel('Calibrated magnitude')
    plt.ylabel('Photometric uncertainty [mag]')
    plt.legend()
    plt.grid()

    plt.savefig('/Users/rstreet1/ROMEREA/test_data/phot_quality_parallel_MS_stars.png', bbox_inches='tight')
    plt.close(1)

def plot_parallax(data):

    idx1 = np.where(~np.isnan(data[:,17]))[0]
    idx2 = np.where(data[:,17] > 0)[0]
    idx = list(set(idx1).intersection(set(idx2)))

    fig = plt.figure(1)
    plt.hist(data[idx,17], bins=50)
    plt.xlabel('Parallax [mas]')
    plt.ylabel('Frequency')
    plt.savefig('/Users/rstreet1/ROMEREA/test_data/parallax_parallel_MS_stars.png', bbox_inches='tight')
    plt.close(1)

    fig = plt.figure(1)
    plt.hist(data[idx,17], bins=50)
    [xmin, xmax, ymin, ymax] = plt.axis()
    plt.axis([0.0, 2.0, ymin, ymax])
    plt.xlabel('Parallax [mas]')
    plt.ylabel('Frequency')
    plt.savefig('/Users/rstreet1/ROMEREA/test_data/parallax_parallel_MS_stars_distant.png', bbox_inches='tight')
    plt.close(1)

    fig = plt.figure(1)
    plt.plot(data[idx,17], data[idx,18], 'k.')
    plt.xlabel('Parallax [mas]')
    plt.ylabel('Parallax uncertainty [mas]')
    plt.grid()

    plt.savefig('/Users/rstreet1/ROMEREA/test_data/parallax_quality_parallel_MS_stars.png', bbox_inches='tight')
    plt.close(1)

if __name__ == '__main__':
    analyze_parallel_ms_stars()
