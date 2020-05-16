from sys import argv
from os import getcwd, path, remove, environ, mkdir
import numpy as np
from pyDANDIA import  phot_db
from pyDANDIA import  hd5_utils
from pyDANDIA import  pipeline_setup
from pyDANDIA import  logs
from pyDANDIA import  RoboNoise
from pyDANDIA import  config_utils
import matplotlib.pyplot as plt
from pyDANDIA import  hd5_utils
from astropy import table

VERSION = 'analyse_red_noise_v0.1.0'

def run_analyse_red_noise(setup):
    """Function to run the RoboNoise code by E. Bachelet within the context
    of pyDANDIA data products"""

    config = get_args()

    log = logs.start_pipeline_log(setup.red_dir, 'analyse_red_noise',
                                  version=VERSION)

    conn = db_phot.get_connection(dsn=setup.phot_db_path)
    image_data = fetch_image_data(conn,photometry,log)
    conn.close()

    photometry_file = path.join(setup.red_dir, 'photometry.hdf5')

    if path.isfile(photometry_file) == False:
        report = 'No timeseries photometry found at '+photometry_file+\
                    ' so no red noise analysis performed'

        log.info('Analyse_red_noise: '+report)
        logs.close_log(log)
        status = 'OK'

        return status, report

    photometry = hd5_utils.load_dataset_timeseries_photometry(setup,log,27)

    (dataset,dico) = compile_phot_data(image_data,photometry,log)

    Solver = RoboNoise.RedNoiseSolver(dataset,dico)
    Solver.clean_bad_data()
    Solver.clean_bad_stars(config['exclude_stars'])
    Solver.clean_magnitude_data(config['faint_limit'])
    Solver.define_continuous_quantities(config['fit_parameters'])
    Solver.CCD_fit_degree=config['CCD_fit_degree']
    Solver.construct_continuous_matrices(config['fit_parameters'])
    Solver.solve()

    plot_red_noise_model(setup,config,Solver,dico,log)

    report = 'Analysis complete'
    log.info('Analyse_red_noise: '+report)
    logs.close_log(log)
    status = 'OK'

    return status, report

def get_args():

    if len(argv) == 1:

        config_file = input('Please enter the path to the configuration file: ')

    else:

        config_file = argv[1]

    config = config_utils.build_config_from_json(config_file,
                                                list_keywords=['exclude_stars',
                                                                'fit_parameters',
                                                                'output_star_indices'])

    return config

def fetch_image_data(conn,photometry,log):
    """Function to query the fields photometric database for the information
    on the images in this dataset"""

    image_pks = photometry[:,:,2].flatten().unique().astype('str').tolist()

    query = 'select img_id, filename, exposure_time, airmass, fwhm from images where img_id in ("' + \
                '","'.join(image_pks)+'")'
    image_data = phot_db.query_to_astropy_table(conn, query, args=())

    log.info('Fetched information on images from photometry DB')

    return image_data

def compile_phot_data(image_data, photometry, config, log):
    """Function to compile all data required for red noise analysis into
    a single 2D array"""

    # Parse and record the option of which photmetry column to use:
    mag_col = { 11: "raw", 13: "dataset calibrated",
                23: "field calibrated", 25: "revised noise model" }
    mag_err_col = { 12: "raw", 14: "dataset calibrated",
                    24: "field calibrated", 26: "revised noise model" }

    if config['mag_column'] in mag_col.keys():
        log.info('Using the '+mag_col[config['mag_column']]+' magnitude measurements for this analysis')
    else:
        raise IOError('Configured magnitude column ('+str(config['mag_column'])+'\
                        ) is not a valid option.  Need one of: '+repr(mag_col)

    if config['mag_err_column'] in mag_err_col.keys():
        log.info('Using the '+mag_col[config['mag_column']]+' magnitude measurements for this analysis')
    else:
        raise IOError('Configured magnitude error column ('+str(config['mag_err_column'])+\
                        ') is not a valid option.  Need one of: '+repr(mag_err_col)

    # Need to efficiently extract the image data to fill in the missing entries:
    dico = {'stars' : 0, 'frames':1, 'time' : 2 ,
            'mag' : 3, 'err_mag' : 4,
            'exposure' : 5, 'airmass' : 6,
            'seeing': 7, 'background':8,
            'CCD_X':9,'CCD_Y':10,
            'phot_scale_factor' :11 }

    phot_data = photometry[:,:,[0,2,9,config['mag_column'],config['mag_err_column'],
    EXPTIME, AIRMASS, SEEING, 21, 7, 8, 19]]

    return phot_data, dico

def plot_red_noise_model(setup,config,Solver,dico,log):
    """Function to produce the standard diagnostic plots"""

    def build_model(Solver,index):
        model=Solver.x1[0]
        count=0
        for i in choice :
        	quantities = Solver.find_model_quantities(i).T

        	if quantities.ndim == 1 :
        		model += quantities[index]*Solver.x2[count]
        		count += 1
        	else :

        		for j in Solver.find_model_quantities(i).T:

        			model += j[index]*Solver.x2[count]
        			count += 1
        return model

    output_dir = path.join(setup.red_dir,'red_noise')
    if path.isdir(output_dir) == False:
        mkdir(output_dir)

    for star in config['output_star_indices']:
        model = build_model(Solver,index)

        fig=plt.figure(1)
        plt.errorbar(Solver.data[index,dico['time']].astype(float)-2450000.0,Solver.data[index,dico['mag']].astype(float),yerr=Solver.data[index,dico['err_mag']].astype(float),fmt='.k')
        plt.plot(Solver.data[index,dico['time']].astype(float)-2450000.0,model,'r',lw=2)
        plt.title(Solver.data[0,dico['stars']]+': m = '+str(Solver.x1[0])+' '+str(Solver.x2[0])+'*'+choice[0])
        plt.gca().invert_yaxis()

        plot_file = path.join(output_dir,'star_'+str(index)+'_red_noise.png')
        plt.savefig(plot_file)
        plt.close(1)
        log.info('Output analysis of star '+str(index)+' to '+plot_file)
