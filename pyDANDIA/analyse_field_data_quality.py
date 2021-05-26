from os import path
from sys import argv
from pyDANDIA import hd5_utils
from pyDANDIA import pipeline_setup
from pyDANDIA import pipeline_control
from pyDANDIA import quality_control
from pyDANDIA import logs
from pyDANDIA import plot_rms
from astropy.table import Table
from astropy.table import Column
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib	import pyplot as plt

def analyse_datasets(setup):
    """Function to evaluate photometry quality metrics for a randome sample of
    stars in a set of datasets for a given field"""

    log = logs.start_pipeline_log(setup.log_dir, 'field_data_quality_analysis')

    datasets = pipeline_control.get_datasets_for_reduction(setup,log)

    metrics = initialize_metrics_table()

    for data_dir,data_status in datasets.items():
        log.info('Calculating metrics for '+path.basename(data_dir))
        photometry = plot_rms.fetch_dataset_photometry({'red_dir': data_dir},log)
        site = str(path.basename(data_dir)).split('_')[1].split('-')[0]
        dataset_metrics = quality_control.calc_phot_qc_metrics(photometry,site,n_selection=5000)
        metrics = append_metric_data(metrics, dataset_metrics)

    plot_field_qc_metrics(setup, metrics,log)

    logs.close_log(log)

def get_args():

    params = {}

    if len(argv) > 1:
        params['base_dir'] = argv[1]
    else:
        params['base_dir'] = input('Please enter the path to the base directory: ')
    params['log_dir'] = path.join(params['base_dir'],'logs')
    params['pipeline_config_dir'] = path.join(params['base_dir'],'config')
    params['verbosity'] = 0
    setup = pipeline_setup.pipeline_setup(params)
    setup.phot_db_path = None

    return setup

def initialize_metrics_table():
    table_data = [  Column(name='mean_cal_mag', data=[]),
                    Column(name='median_cal_mag', data=[]),
                    Column(name='std_dev_cal_mag', data=[]),
                    Column(name='median_cal_mag_error', data=[]),
                    Column(name='n_valid_points', data=[]),
                    Column(name='median_ps_factor', data=[]),
                    Column(name='median_ps_error', data=[]),
                    Column(name='median_sky_background', data=[]),
                    Column(name='median_sky_background_error', data=[]),
                    Column(name='site', data=[]) ]
    metrics = Table(data=table_data)

    return metrics

def append_metric_data(master_table, dataset_table):

    table_data = []

    for col in master_table.colnames:
        new_data = np.concatenate([master_table[col].data, dataset_table[col].data])
        table_data.append(Column(name=col, data=new_data))

    new_table = Table(data=table_data)
    return new_table

def plot_field_qc_metrics(setup, metrics,log):

    dataset = pd.DataFrame({'SITE':metrics['site'].data.astype(np.str),
                            'MEDIAN_LC_MAG':metrics['median_cal_mag'].data.astype(np.float),
                            'LOG10_RMS_LC_MAG':np.log10(metrics['std_dev_cal_mag'].data.astype(np.float)),
                            'LOG10_REPORTED_ERR_LC_MAG':np.log10(metrics['median_cal_mag_error'].data.astype(np.float)),
                            'N_DATA':metrics['n_valid_points'].data.astype(np.float),
                            'MEDIAN_PSCALE':metrics['median_ps_factor'].data.astype(np.float),
                            'MEDIAN_BKG':metrics['median_sky_background'].data.astype(np.float)})
    dataset['SITE'] = dataset['SITE'].astype('category')

    sns.set(style="whitegrid")
    sns.set(font_scale=3)
    sns.set(style="ticks")
    sns.pairplot(dataset, hue="SITE")
    #g = sns.PairGrid(dataset, diag_sharey=False, hue = "SITE")
    #g.map_upper(sns.scatterplot)
    #g.map_lower(sns.kdeplot, colors="C0")
    #g.map_diag(sns.kdeplot, lw=2)

    file_path = path.join(setup.log_dir,'field_qc_metrics.png')
    plt.savefig(file_path)

    log.info('Plotted quality metrics for the field to '+file_path)

if __name__ == '__main__':
    setup = get_args()
    analyse_datasets(setup)
