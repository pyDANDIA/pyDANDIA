import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from os import path

def plot_interactive_lightcurve(lc, filter_list, plot_file, title=None):
    """Function to create an interactive lightcurve plot of a multi-dataset
    lightcurve dictionary
    lc : dict : {dataset1: array, dataset2: array, ...}
    where array is a three-column set of datapoints with HJD, mag, mag error
    plot_file : str : path to output html file
    Expected dataset labels have the format site-enclosure_filter
    """
    colors = {'lsc-doma': '#332288',
              'lsc-domb': '#88CCEE',
              'lsc-domc': '#44AA99',
              'cpt-doma': '#117733',
              'cpt-domb': '#999933',
              'cpt-domc': '#DDCC77',
              'coj-doma': '#CC6677',
              'coj-domb': '#882255',
              'coj-domc': '#AA4499'}
    symbols = {'lsc-doma': 'circle',
              'lsc-domb': 'circle',
              'lsc-domc': 'circle',
              'cpt-doma': 'diamond',
              'cpt-domb': 'diamond',
              'cpt-domc': 'diamond',
              'coj-doma': 'square',
              'coj-domb': 'square',
              'coj-domc': 'square'}
    hjd_offset = 2450000.0

    data_list = []
    for col_idx,f in enumerate(filter_list):
        for dset, data in lc.items():
            # Expected dataset labels are site-enclosure_filter
            if f in dset:
                (sitecode,passband) = dset.split('_')
                data_list.append(pd.DataFrame(
                            {
                            'HJD': pd.Series(data[:,0]-hjd_offset, dtype='float64'),
                            'mag': pd.Series(data[:,1], dtype='float64'),
                            'mag_error': pd.Series(data[:,2], dtype='float64'),
                            'datacode': pd.Series([dset]*len(data), dtype='str'),
                            'filter': pd.Series([f.replace('p','')]*len(data), dtype='str')
                            }
                            ))

    df = pd.concat(data_list)

    fig = px.scatter(df, x='HJD', y='mag', color='datacode', error_y='mag_error',
                    facet_col='filter',
                    labels=dict(HJD="HJD-"+str(hjd_offset),
                                mag="Mag",
                                datacode="Dataset"))

    if title:
        fig.update_layout(height=600, width=600*len(filter_list),
                        title=title)
    else:
        fig.update_layout(height=600, width=200*len(filter_list))
    fig.update_xaxes(title_font=dict(size=18),
                     tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=18),
                     tickfont=dict(size=18))
    fig.write_html(plot_file)

def plot_from_lc_file(file_path):

    if not path.isfile(file_path):
        raise IOError('Cannot find lightcurve file '+file_path)

    # Default format of field lightcurve files is:
    # HJD Instrumental mag, mag_error   Calibrated mag, mag_error
    # Corrected mag, mag_error   Normalized mag, mag_error  QC_Flag
    data = np.loadtxt(file_path, skiprows=1)
    idx = np.where(data[:,0] > 0.0)[0]
    data = data[idx,:]
    data = data[:,[0,5,6]]

    datacode = 'lsc-doma_ip'
    lc = {datacode: data}
    filter_list = ['ip']
    plot_file = file_path.replace('.dat','.html')

    plot_interactive_lightcurve(lc, filter_list, plot_file)

if __name__ == '__main__':
    file_path = '/Users/rstreet1/ROMEREA/ROME-FIELD-01/DR1/lcs/star_172333_ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip.dat'
    plot_from_lc_file(file_path)
