from os import path
from sys import argv
from astropy.io import fits

def determine_image_struture(file_path, log=None):
    """Function to determine the structure of a FITS image file, identifying
    the list indices which contain the main image header, the science image
    and the Bad Pixel Mask"""

    if path.isfile(file_path) == False:
        raise IOError('Cannot find file '+file_path)

    image_structure = {'sci': None, 'bpm': None}

    hdu = fits.open(file_path)

    for i in range(0,len(hdu),1):
        if hdu[i].name == 'SCI':
            image_structure['sci'] = i
        elif hdu[i].name == 'BPM':
            image_structure['bpm'] = i

    if image_structure['sci'] == None:
        raise IOError('Cannot find any science data in image '+file_path)
    if image_structure['bpm'] == None:
        raise IOError('Cannot find a BPM for image '+file_path)

    if log != None:
        log.info('Determined that image '+path.basename(file_path)+\
                 ' has the following structure: '+repr(image_structure))

    hdu.close()

    return image_structure

def get_science_header(file_path, image_structure={}):

    if len(image_structure) == 0:
        image_structure = determine_image_struture(file_path)

    hdu = fits.open(file_path)

    header = hdu[image_structure['sci']].header

    hdu.close()

    return header

def get_science_image(file_path, image_structure={}):

    if len(image_structure) == 0:
        image_structure = determine_image_struture(file_path)

    hdu = fits.open(file_path)

    data = hdu[image_structure['sci']].data

    hdu.close()

    return data

if __name__ == '__main__':

    if len(argv) > 1:
        file_path = argv[1]
    else:
        file_path = input('Please enter the file path to an image: ')

    image_structure = determine_image_struture(file_path, log=None)
    print(image_structure)
