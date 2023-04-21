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
              'coj-domc': '#AA4499',
              'bad-data': '#000000'}
    symbols = {'lsc-doma': 'circle',
              'lsc-domb': 'circle',
              'lsc-domc': 'circle',
              'cpt-doma': 'diamond',
              'cpt-domb': 'diamond',
              'cpt-domc': 'diamond',
              'coj-doma': 'square',
              'coj-domb': 'square',
              'coj-domc': 'square',
              'bad-data': 'cross'}
    hjd_offset = 2450000.0

    data_list = []
    for col_idx,f in enumerate(filter_list):
        bad_data = []
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
                            'filter': pd.Series([f.replace('p','')]*len(data), dtype='str'),
                            'qc_flag': pd.Series(data[:,3], dtype='float64')
                            }
                            ))

                # Filter datapoints that failed QC so they can be highlighted
                # on the plot
                bad_idx = np.where(data[:,3] != 0.0)[0]
                print(dset+' '+str(len(data))+' datapoints, '+str(len(bad_idx))+' flagged as bad')
                for i in bad_idx:
                    bad_data.append([data[i,0], data[i,1], data[i,2], data[i,3]])

        #bad_data = np.array(bad_data)
        #data_list.append(pd.DataFrame(
        #        {
        #        'HJD': pd.Series(bad_data[:,0]-hjd_offset, dtype='float64'),
        #        'mag': pd.Series(bad_data[:,1], dtype='float64'),
        #        'mag_error': pd.Series(bad_data[:,2], dtype='float64'),
        #        'datacode': pd.Series(['bad-data']*len(bad_data), dtype='str'),
        #        'filter': pd.Series([f.replace('p','')]*len(bad_data), dtype='str'),
        #        'qc_flag': pd.Series(bad_data[:,3], dtype='float64')
        #        }
        #        ))

    if len(data_list) > 0:
        df = pd.concat(data_list)

        fig = px.scatter(df, x='HJD', y='mag', color='datacode', error_y='mag_error',
                        facet_col='filter',
                        labels=dict(HJD="HJD-"+str(hjd_offset),
                                    mag="Mag",
                                    datacode="Dataset"),
                        hover_data=['HJD','mag','mag_error','qc_flag'])

        if title:
            fig.update_layout(height=600, width=600*len(filter_list),
                            title=title)
        else:
            fig.update_layout(height=600, width=200*len(filter_list))
        fig['layout']['yaxis']['autorange'] = "reversed"
        fig.update_xaxes(title_font=dict(size=18),
                         tickfont=dict(size=18))
        fig.update_yaxes(title_font=dict(size=18),
                         tickfont=dict(size=18))
        fig.write_html(plot_file)
    else:
        print('Warning: data list empty; no valid lightcurve to plot')

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

def plot_interactive(data, plot_file, axis_labels, target_params,
                        title=None, logy=False, xreverse=False,
                        xrange=None, yrange=None):
    """General purpose function to create an interactive graph with a single dataset
    data : array: a three-column set of datapoints with field ID (not index), mag, color
    plot_file : str : path to output html file
    axis_labels: list of str : labels for X, Y axes in that order
    """

    text_labels = np.array(['Field ID='+str(int(id)) for id in list(data[:,0])])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
                mode='markers',
                x=data[:,1],
                y=data[:,2],
                marker=dict(
                    color='Black',
                    size=2
                ),
                marker_symbol='circle',
                name='Field stars',
                text=text_labels,
                hoverinfo='x+y+text',
                showlegend=False,
            )
        )

    if target_params[0]:
        fig.add_trace(
            go.Scatter(
                    mode='markers',
                    x=np.array([target_params[1]]),
                    y=np.array([target_params[2]]),
                    marker=dict(
                        color='LightSkyBlue',
                        size=5
                    ),
                    marker_symbol='diamond',
                    name='Target',
                    text='Field ID='+str(int(target_params[0])),
                    hoverinfo='x+y+text',
                    showlegend=False,
                )
            )

    if title:
        fig.update_layout(height=600, width=600,
                        title=title)
    else:
        fig.update_layout(height=600, width=600)
    fig.update_xaxes(title_font=dict(size=18),
                     tickfont=dict(size=18))
    fig.update_yaxes(title_font=dict(size=18),
                     tickfont=dict(size=18))
    if xrange:
        fig.update_xaxes(range=xrange)
    if yrange:
        fig.update_xaxes(range=yrange)
    if xreverse:
        fig['layout']['xaxis']['autorange'] = "reversed"
        
    fig.write_html(plot_file)


if __name__ == '__main__':
    file_path = '/Users/rstreet1/ROMEREA/ROME-FIELD-01/DR1/lcs/star_172333_ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip.dat'
    plot_from_lc_file(file_path)
