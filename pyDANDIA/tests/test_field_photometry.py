import numpy as np
from pyDANDIA import field_photometry
from pyDANDIA import crossmatch
from pyDANDIA import logs
from pyDANDIA import metadata
from astropy.table import Table, Column
from astropy.io import fits
import h5py
from os import path

def test_params():
    params = {'primary_ref': 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip',
              'datasets': { 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/', 'ip'],
                            'ROME-FIELD-01_lsc-doma-1m0-05-fa15_rp' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_rp/', 'rp' ],
                            'ROME-FIELD-01_lsc-domb-1m0-09-fa15_gp' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/ROME-FIELD-01_lsc-domb-1m0-09-fa15_gp/', 'gp' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.'}

    return params

def test_field_index(xmatch):

    xmatch.field_index.add_row([1, 267.61861696019145, -29.829605383706895, 4, 1, None, 1, 0, 0])
    xmatch.field_index.add_row([2, 267.70228408545813, -29.83032824102953, 4, 2, None, 2, 0, 0])
    xmatch.field_index.add_row([3, 267.9873108673885, -29.829734325692858, 3, 1, None, 3, 0, 0])
    xmatch.field_index.add_row([4, 267.9585073984874, -29.83002538112054, 3, 2, None, 4, 0, 0])
    xmatch.field_index.add_row([5, 267.9623466389135, -29.82994179424344, 3, 3, None, 5, 0, 0])
    xmatch.field_index.add_row([6, 267.9315803167322, -29.830983939264463, 3, 35, None, 6, 105, 13])
    xmatch.field_index.add_row([7, 267.94313361856774, -29.830855906070912, 3, 36, None, 7, 66, 0])
    xmatch.field_index.add_row([8, 267.9641908032068, -29.83105008127081, 3, 67, None, 8, 138, 27])
    xmatch.field_index.add_row([9, 267.96937719314764, -29.831020635327544, 3, 68, None, 9, 139, 28])
    xmatch.field_index.add_row([10, 267.97122056100426, -29.83096064913158, 3, 69, None, 10, 108, 15])
    xmatch.field_index.add_row([11, 267.9934578741869, -29.83090679218163, 3, 70, None, 11, 143, 0])

    return xmatch

def test_stars_table(xmatch):

    for star in xmatch.field_index:
        xmatch.stars.add_row([star['field_id'], star['ra'], star['dec']] + [0.0]*36)

    return xmatch

def test_star_catalog(meta):

    nstars = 11
    star_catalog = np.zeros((nstars,37))
    #star_catalog[0] = [1, 346.7463278676221, 0.31769726490029015, 267.61861696019145, -29.829605383706895, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056436121079692160, 267.6188626384401, 8.781465530395508, -29.829782879978076, 7.436490535736084, 286.91259765625, 2.5169389247894287, -9999.999, -9999.999, -9999.999, -9999.999, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10]
    #star_catalog[1] = [2, 1017.3340457345882, 8.379706811851744, 267.70228408545813, -29.83032824102953, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10]
    star_catalog[2] = [3, 3301.8400837300433, 11.255029683618215, 267.9873108673885, -29.829734325692858, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444539267431040, 267.98727887564667, 0.045853544026613235, -29.829912824096596, 0.03865557536482811, 23195.255859375, 10.54239273071289, 11357.7841796875, 28.62028694152832, 18024.580078125, 40.84840774536133, None, 267.98729, -29.82988, 15.5, 0.0, 14.619999885559082, 0.0, 13.989999771118164, 0.0, 0.0, 1.0E10]
    star_catalog[3] = [4, 3070.9685725169215, 12.842726035986505, 267.9585073984874, -29.83002538112054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444878525743616, 267.95849533448626, 0.12481509894132614, -29.830130309157735, 0.1085614413022995, 7796.68701171875, 9.380874633789062, 783.80517578125, 17.880268096923828, 13934.6943359375, 136.3885955810547, None, 267.95851, -29.83011, 20.25, 0.029999999329447746, 16.850000381469727, 0.009999999776482582, 14.739999771118164, 0.009999999776482582, 0.0, 1.0E10]
    star_catalog[4] = [5, 3101.7435341021883, 12.21388380362664, 267.9623466389135, -29.82994179424344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444917224603904, 267.9623198050811, 0.22476719319820404, -29.830078194809662, 0.18630900979042053, 11072.041015625, 46.79934310913086, 1684.195068359375, 49.38384246826172, 19294.3125, 126.06063079833984, None, 267.96233, -29.83006, 19.600000381469727, 0.019999999552965164, 16.270000457763672, 0.009999999776482582, 14.380000114440918, 0.009999999776482582, 0.0, 1.0E10]
    star_catalog[5] = [50, 2855.1098226019617, 20.722212446330264, 267.9315803167322, -29.830983939264463, 72.83591222434477, 1.2387420738083867, 20.344136090341717, 0.018465439325429298, 17.999026801794326, 0.018465439325429298, 631.5231558024683, 10.740502587064343, 1085.0, 5179.085342983518, 4056444844165932544, 267.93156186163185, 0.3745168447494507, -29.831130541812637, 0.3244278132915497, 599.3142700195312, 2.3796072006225586, 332.97344970703125, 16.70027732849121, 1802.3887939453125, 43.210113525390625, None, 267.93157, -29.8311, 21.959999084472656, 0.10999999940395355, 18.959999084472656, 0.029999999329447746, 17.450000762939453, 0.019999999552965164, 0.0, 0.0]
    star_catalog[6] = [51, 2947.713447326438, 19.952506222613422, 267.94313361856774, -29.830855906070912, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444947245158016, 267.94308300945704, 0.8784804344177246, -29.83089780473822, 0.6652926206588745, 475.24163818359375, 2.5971155166625977, -9999.999, -9999.999, -9999.999, -9999.999, None, 267.9431, -29.83087, 21.969999313354492, 0.11999999731779099, 19.329999923706055, 0.029999999329447746, 17.809999465942383, 0.019999999552965164, 0.0, 1.0E10]
    star_catalog[7] = [90, 3116.475668245582, 22.523958538511042, 267.9641908032068, -29.83105008127081, 67.91323518000698, 1.241420694285689, 20.420113951429993, 0.01984672633470334, 18.087107346368825, 0.01984672633470334, 582.3137719332096, 10.644410697404933, 1085.0, 5179.085342983518, 4056444912885400832, 267.96420196637877, 0.4867507517337799, -29.831163227678985, 0.4509122669696808, 874.8128662109375, 7.869678020477295, 415.13970947265625, 33.829490661621094, 2185.14501953125, 55.643558502197266, None, 267.96421, -29.83113, 21.739999771118164, 0.18000000715255737, 18.719999313354492, 0.019999999552965164, 17.100000381469727, 0.019999999552965164, 0.0, 0.0]
    star_catalog[8] = [91, 3158.045169151278, 22.44803844400929, 267.96937719314764, -29.831020635327544, 66.76936467853554, 1.2428182815630278, 20.438556890325344, 0.020209459992990574, 18.10848810226519, 0.020209459992990574, 570.9587843722417, 10.627598729645248, 1085.0, 5179.085342983518, 4056444912885534976, 267.96940117953943, 0.2777192294597626, -29.831159569166665, 0.24106402695178986, 1009.09765625, 3.348869800567627, 470.4875183105469, 19.688159942626953, 3108.443359375, 116.98040008544922, None, 267.96941, -29.83114, 20.81999969482422, 0.05000000074505806, 18.3799991607666, 0.019999999552965164, 17.010000228881836, 0.009999999776482582, 1.0, 0.0]
    star_catalog[9] = [92, 3172.822182455274, 21.96392772157342, 267.97122056100426, -29.83096064913158, 1389.5969629009883, 1.3872958687360546, 17.14277785938164, 0.001083938286863857, 14.287717055480321, 0.001083938286863857, 19271.39609627774, 19.23948375169784, 1085.0, 5179.085342983518, 4056444912869653504, 267.97119999632883, 0.051752395927906036, -29.831062674301922, 0.04540014639496803, 27424.91015625, 19.671449661254883, 7664.77392578125, 49.09895706176758, 32068.123046875, 50.958839416503906, None, 267.97121, -29.83104, 16.479999542236328, 0.009999999776482582, 14.539999961853027, 0.0, 13.479999542236328, 0.009999999776482582, 0.0, 0.0]
    star_catalog[10] = [93, 3351.0537899352307, 22.33141678451542, 267.9934578741869, -29.83090679218163, 2.810801157316224, 1.2340045184832593, 23.877924690797105, 0.4766624558500083, 22.095720494942896, 0.4766624558500083, 14.511483154459642, 6.370865379746347, 1085.0, 5179.085342983518, 4056444534928464128, 267.9934558649592, 0.794064462184906, -29.83104086736632, 0.9368880391120911, 394.9053039550781, 3.421269655227661, 247.20770263671875, 20.210479736328125, 1386.758544921875, 	44.8889045715332, None, 267.99345, -29.83099, 22.610000610351562, 0.3100000023841858, 19.510000228881836, 0.05000000074505806, 17.8700008392334, 0.029999999329447746, 0.0, 0.0]

    layer_header = fits.Header()
    layer_header.update({'NAME': 'star_catalog'})
    names = ['index', 'x', 'y', 'ra', 'dec',
            'ref_flux', 'ref_flux_error', 'ref_mag', 'ref_mag_error',
            'cal_ref_mag', 'cal_ref_mag_error', 'cal_ref_flux', 'cal_ref_flux_error',
            'sky_background', 'sky_background_error',
            'gaia_source_id',
            'gaia_ra', 'gaia_ra_error', 'gaia_dec', 'gaia_dec_error', 'phot_g_mean_flux',
            'phot_g_mean_flux_error', 'phot_bp_mean_flux',
            'phot_bp_mean_flux_error', 'phot_rp_mean_flux',
            'phot_rp_mean_flux_error',
            'vphas_source_id', 'vphas_ra', 'vphas_dec', 'gmag', 'gmag_error',
            'rmag', 'rmag_error', 'imag', 'imag_error', 'clean',
            'psf_star']
    data_format = ['int','float', 'float', 'float', 'float',
                    'float', 'float','float', 'float',
                    'float', 'float','float', 'float',
                    'float', 'float',
                    'int',
                    'float', 'float','float', 'float','float',
                    'float', 'float',
                    'float', 'float',
                    'float',
                    'int', 'float','float', 'float', 'float',
                    'float', 'float','float', 'float', 'int',
                    'int']
    layer_table = Table(star_catalog, names=names, dtype=data_format)
    setattr(meta, 'star_catalog', [layer_header, layer_table])

    return meta

def test_dataset_timeseries_photometry(meta, xmatch):

    photometry = []

    hjds = []
    for i,image in enumerate(xmatch.images):
        hjds.append(24521032.0 + 0.01*i)

    for star in meta.star_catalog[1]:
        image_data = []
        for i,image in enumerate(xmatch.images):
            data = [star['index'], 0, i, 0, 0, 0, 0, star['x'], star['y'], hjds[i], 8.0,\
                    star['ref_mag'] + np.random.standard_normal()*0.001,\
                    star['ref_mag_error'] + np.random.standard_normal()*0.00001, \
                    star['cal_ref_mag'] + np.random.standard_normal()*0.001,\
                    star['cal_ref_mag_error'] + np.random.standard_normal()*0.00001, \
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            image_data.append(data)
        photometry.append(image_data)

    return np.array(photometry)

def test_headers_summary(meta):
    names = np.array(['IMAGES', 'EXPKEY', 'OBJKEY', 'OBSKEY', 'DATEKEY', \
                        'TIMEKEY', 'RAKEY', 'DECKEY', 'FILTKEY', \
                        'MOONDKEY', 'MOONFKEY'])
    formats = np.array(['S200']*11)
    data = np.array([['lsc1m005-fa15-20190612-0216-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:36:47.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005'],
            ['lsc1m005-fa15-20190612-0217-e91.fits', '300.0', 'ROME-FIELD-01',
        'EXPOSE', '2019-06-13T02:37:00.449', '02:36:47.449', '17:51:22.5696',
        '-30:03:37.550', 'ip', '60.7240446', '0.799005'],
        ['lsc1m005-fa15-20190612-0218-e91.fits', '300.0', 'ROME-FIELD-01',
    'EXPOSE', '2019-06-13T02:38:00.449', '02:36:47.449', '17:51:22.5696',
    '-30:03:37.550', 'ip', '60.7240446', '0.799005']
            ])
    meta.create_headers_summary_layer(names, formats, units=None, data=data)

    return meta

def test_init_field_data_table():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    photometry = field_photometry.init_field_data_table(xmatch, log)

    assert(type(photometry) == type([]))
    logs.close_log(log)

def test_populate_images_table():
    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    meta = metadata.MetaData()
    meta = test_headers_summary(meta)

    (xmatch, image_index) = field_photometry.populate_images_table(xmatch.datasets[0], meta, xmatch, log)

    assert(len(xmatch.images) == 1)
    assert(xmatch.images['filename'] == data[0])
    assert(xmatch.images['filter'] == data[8])
    assert(xmatch.images['dataset_code'] == xmatch.datasets[0]['dataset_code'])
    assert(xmatch.images['hjd'] == 0.0)

    logs.close_log(log)

def test_populate_stars_table():
    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_field_index(xmatch)
    xmatch = test_stars_table(xmatch)

    meta = metadata.MetaData()
    meta = test_star_catalog(meta)

    (xmatch, field_array_idx, dataset_array_idx) = field_photometry.populate_stars_table(xmatch.datasets[0],xmatch,meta,log)

    filter_name = xmatch.datasets[0]['dataset_filter'].replace('p','')
    dataset_id = '_'.join(xmatch.datasets[0]['dataset_code'].split('_')[1].split('-')[0:2])
    mag_column = 'cal_'+filter_name+'_mag_'+dataset_id
    mag_error_column = 'cal_'+filter_name+'_magerr_'+dataset_id

    assert(field_array_idx == dataset_array_idx).all()
    for j in range(0,nstars,1):
        assert(xmatch.stars[mag_column][j] == star_catalog[j,9])
        assert(xmatch.stars[mag_error_column][j] == star_catalog[j,10])

    logs.close_log(log)

def test_populate_photometry_array():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_field_index(xmatch)
    xmatch = test_stars_table(xmatch)

    meta = metadata.MetaData()
    meta = test_star_catalog(meta)
    meta = test_headers_summary(meta)

    # Image table must be populated before the timeseries can be generated
    (xmatch, dataset_image_idx) = field_photometry.populate_images_table(xmatch.datasets[0], meta, xmatch, log)
    dataset_photometry = test_dataset_timeseries_photometry(meta, xmatch)

    # Create the dataset_photometry as an HDF5 object
    file_path = path.join(params['log_dir'],'test.hdf5')
    with h5py.File(file_path, "w") as f:
        dataset_photometry = f.create_dataset('dataset_photometry',
                                    dataset_photometry.shape,
                                    dtype='float64',
                                    data=dataset_photometry)
    f.close()
    f = h5py.File(file_path, "r")
    dataset_photometry = f['dataset_photometry']

    (xmatch, field_array_idx, dataset_array_idx) = field_photometry.populate_stars_table(xmatch.datasets[0],xmatch,meta,log)

    # Can be initialized only after the images and stars tables have been
    # populated with all datasets
    photometry = field_photometry.init_field_data_table(xmatch,log)

    (xmatch, photometry) = field_photometry.populate_photometry_array(field_array_idx, dataset_array_idx,
                                    dataset_image_idx, photometry, dataset_photometry, xmatch, log)

    for i,iimage in enumerate(dataset_image_idx):
        assert(xmatch.images['hjd'][iimage] == dataset_photometry[0,i,9])

    for j in field_array_idx:
        for i,iimage in enumerate(dataset_image_idx):
            print('Phot array: ',photometry[j,iimage,0])
            print('Dataset: ',dataset_photometry[dataset_array_idx[j],i,9])
            assert( photometry[j,iimage,0] == dataset_photometry[dataset_array_idx[j],i,9])
            assert( photometry[j,iimage,3] == dataset_photometry[dataset_array_idx[j],i,13])
            assert( photometry[j,iimage,4] == dataset_photometry[dataset_array_idx[j],i,14])

    logs.close_log(log)

def test_build_array_index_3D():

    idx1 = [1, 2]
    idx2 = [0, 1, 2]
    nentries = len(idx1)*len(idx2)

    index = field_photometry.build_array_index([idx1, idx2])

    assert( [type(idx) == type(np.zeros(1)) for idx in index] )
    assert( [len(idx) == nentries for idx in index])
    assert( (index[0] == np.array([1,1,1,2,2,2]) ).all() )
    assert( (index[1] == np.array([0,1,2,0,1,2]) ).all() )

    idx1 = [ 3, 4, 5 ]
    idx2 = [ 0, 1 ]
    idx3 = [ 2, 7, 9 ]
    nentries = len(idx1)*len(idx2)*len(idx3)

    index = field_photometry.build_array_index([idx1, idx2, idx3])

    assert( [type(idx) == type(np.zeros(1)) for idx in index] )
    assert( (index[0] == np.array([3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5]) ).all() )
    assert( (index[1] == np.array([0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1]) ).all() )
    assert( (index[2] == np.array([2,7,9,2,7,9,2,7,9,2,7,9,2,7,9,2,7,9]) ).all() )

def test_update_array_col_index():

    nentries = 10
    index = (np.arange(0,nentries,1), np.arange(0,nentries,1), np.arange(0,nentries,1))
    print(index)
    new_col_value = 4
    new_index = field_photometry.update_array_col_index(index, new_col_value)
    print(new_index)

    assert((new_index[2] == new_col_value).all())

if __name__ == '__main__':
    #test_init_field_data_table()
    #test_populate_images_table()
    #test_populate_stars_table()
    #test_populate_photometry_array()
    #test_build_array_index_3D()
    test_update_array_col_index()
