from sys import argv
from os import path
from pyDANDIA import crossmatch

def load_xmatch(xmatch_file):

    if path.isfile(xmatch_file):
        xmatch = crossmatch.CrossMatchTable()
        xmatch.load(xmatch_file)
    else:
        raise IOError('Cannot find crossmatch file '+xmatch_file)

    return xmatch

def reinit_stars_table(xmatch, xmatch_file):
    xmatch.create_stars_table()
    xmatch.init_stars_table()
    xmatch.save(xmatch_file)

def init_gaia_source_id(xmatch, xmatch_file):
    xmatch.field_index['gaia_source_id'] = 'None'
    xmatch.stars['gaia_source_id'] = 0.0
    xmatch.save(xmatch_file)

if __name__ == '__main__':

    if len(argv) == 1:
        xmatch_file = input('Please enter the path to the crossmatch file: ')
        print("""Main menu:
                Reinitialize the stars table        1
                Initialize Gaia source ID columns   2
                Cancel                               Any other key""")
        opt = input('Please select an option: ')
    else:
        xmatch_file = argv[1]
        opt = argv[2]

    xmatch = load_xmatch(xmatch_file)

    if opt == '1':
        reinit_stars_table(xmatch, xmatch_file)
    elif opt == '2':
        init_gaia_source_id(xmatch, xmatch_file)
