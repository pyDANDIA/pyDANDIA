from os import path, remove
from pyDANDIA import crossmatch
from pyDANDIA import crossmatch_datasets
from pyDANDIA import crossmatch_field_gaia
from pyDANDIA import match_utils
from pyDANDIA import metadata
from pyDANDIA import logs
import test_field_photometry
from astropy.table import Table, Column
from astropy import units as u
import numpy as np

def test_params():
    params = {'primary_ref': 'primary_ref_dataset',
              'datasets': { 'primary_ref_dataset': ['primary_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset_p/', 'none'],
                            'dataset0' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset0/', 'none' ],
                            'dataset1' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset1/', 'none' ],
                            'dataset2' : [ 'non_ref', '/Users/rstreet1/OMEGA/test_data/non_ref_dataset1/', 'none' ]},
              'file_path': 'crossmatch_table.fits',
              'log_dir': '.',
              'gaia_dr': 'Gaia_DR2',
              'separation_threshold': (2.0/3600.0)*u.deg}

    return params

def test_matched_stars():
    matched_stars = match_utils.StarMatchIndex()
    p = {'cat1_index': 1,
         'cat1_ra': 250.0,
         'cat1_dec': -27.5,
         'cat1_x': 1000.0,
         'cat1_y': 1000.0,
         'cat2_index': 1,
         'cat2_ra': 250.1,
         'cat2_dec': -27.5,
         'cat2_x': 1000.0,
         'cat2_y': 1000.0,
         'separation': 0.1}
    matched_stars.add_match(p)
    return matched_stars

def test_orphans():
    orphans = match_utils.StarMatchIndex()
    p = {'cat1_index': None,
         'cat1_ra': 0.0,
         'cat1_dec': 0.0,
         'cat1_x': 0.0,
         'cat1_y': 0.0,
         'cat2_index': 1,
         'cat2_ra': 252.0,
         'cat2_dec': -27.2,
         'cat2_x': 0.0,
         'cat2_y': 0.0,
         'separation': -1.0}
    orphans.add_match(p)
    return orphans

def test_field_index(xmatch):

    xmatch.field_index.add_row([1,267.61861696019145, -29.829605383706895, 4, 1, None, 1, 0, 0, 0])
    xmatch.field_index.add_row([2,267.70228408545813, -29.83032824102953, 4, 2, None, 2, 0, 0, 0])
    xmatch.field_index.add_row([3,267.9873108673885, -29.829734325692858, 3, 1, None, 3, 0, 0, 0])
    xmatch.field_index.add_row([4,267.9585073984874, -29.83002538112054, 3, 2, None, 4, 0, 0, 0])
    xmatch.field_index.add_row([5,267.9623466389135, -29.82994179424344, 3, 3, None, 5, 0, 0, 0])
    xmatch.field_index.add_row([6,267.943683356543, -29.830113202355186, 3, 4, None, 6, 0, 0, 0])
    xmatch.field_index.add_row([7,267.90449275089594, -29.830465810573223, 3, 5, None, 7, 0, 0, 0])
    xmatch.field_index.add_row([8,267.9504950018423, -29.830247462548577, 3, 6, None, 8, 0, 0, 0])
    xmatch.field_index.add_row([9,267.9778110411362, -29.83012645385565, 3, 7, None, 9, 0, 0, 0])
    xmatch.field_index.add_row([10,267.7950771349625, -29.830849947501875, 4, 3, None, 10, 0, 0, 0])
    xmatch.field_index.add_row([11,268.06583501505446, -29.83070761362742, 3, 84, 4056397427727492224, 11, 0, 0, 0])
    xmatch.field_index.add_row([12,268.0714302057775, -29.830599528895256, 3, 85, 4056403303242709888, 12, 0, 0, 0])
    xmatch.field_index.add_row([13,268.07569655803013, -29.83064274854432, 3, 86, 4056403307573006208, 13, 0, 0, 0])
    xmatch.field_index.add_row([14,268.07663104709775, -29.830575490772073, 3, 87, 4056403303204313344, 14, 0, 0, 0])
    xmatch.field_index.add_row([15,268.07816636284224, -29.830684523662065, 3, 88, 4056403307572525184, 15, 0, 0, 0])

    return xmatch

def test_gaia_catalog():
    nstars = 5
    table_data = [  Column(name='source_id', data = np.array([4056397427727492224, 4056403303242709888, 4056403307573006208, 4056403307618099840, 4056403307572525184])),
                    Column(name='ra', data = np.array([268.0657435285, 268.07133070742, 268.07560517009, 268.07658092476, 268.07819547689])),
                    Column(name='ra_error', data = np.array([1.023, 0.1089, 0.4107, 38.0801, 2.0125])),
                    Column(name='dec', data = np.array([-29.83088151459, -29.83073507254, -29.83081220358, -29.83107857205, -29.83092724519])),
                    Column(name='dec_error', data = np.array([0.9754, 0.0946, 0.3265, 17.3181, 1.3702])) ]

    colnames = ['phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
                'phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'proper_motion', 'pm_ra', 'pm_dec',
                'parallax', 'parallax_error']

    for col in colnames:
        table_data.append( Column(name=col, data=np.zeros(nstars)) )

    gaia_star_field_ids = [ 11, 12, 13, 14, 15 ]

    return Table(table_data), gaia_star_field_ids

def test_stars_table(xmatch):

    xmatch.stars.add_row([1,267.61861696019145, -29.829605383706895]+[0.0]*36)
    xmatch.stars.add_row([2,267.70228408545813, -29.83032824102953]+[0.0]*36)
    xmatch.stars.add_row([3,267.9873108673885, -29.829734325692858]+[0.0]*36)
    xmatch.stars.add_row([4,267.9585073984874, -29.83002538112054]+[0.0]*36)
    xmatch.stars.add_row([5,267.9623466389135, -29.82994179424344]+[0.0]*36)
    xmatch.stars.add_row([6,267.943683356543, -29.830113202355186]+[0.0]*36)
    xmatch.stars.add_row([7,267.90449275089594, -29.830465810573223]+[0.0]*36)
    xmatch.stars.add_row([8,267.9504950018423, -29.830247462548577]+[0.0]*36)
    xmatch.stars.add_row([9,267.9778110411362, -29.83012645385565]+[0.0]*36)
    xmatch.stars.add_row([10,267.7950771349625, -29.830849947501875]+[0.0]*36)
    xmatch.stars.add_row([11,268.06583501505446, -29.83070761362742]+[0.0]*36)
    xmatch.stars.add_row([12,268.0714302057775, -29.830599528895256]+[0.0]*36)
    xmatch.stars.add_row([13,268.07569655803013, -29.83064274854432]+[0.0]*36)
    xmatch.stars.add_row([14,268.07663104709775, -29.830575490772073]+[0.0]*36)
    xmatch.stars.add_row([15,268.07816636284224, -29.830684523662065]+[0.0]*36)

    return xmatch

def test_images_table(xmatch):

    xmatch.images.add_row([0, 'lsc1m005-fa15-20190610-0205-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 2458645.7883432973, '2019-06-11T06:43:16.626', 300.027, '17:51:20.1818', '-30:03:22.197',	86.3219901, 0.6088179, 1.0564295, 1.8786789649990936, 1.7507988428740193, 2180.902135178605, -52.60656380124374, 4.273383547487297, -0.09175497562167338, 20, 0.0, 0.06594043970108032, 1, 1, 18.7, 51.4, 1.2996041667737925, 8.981180114969149E-9, 0.020750108473197144, 915856.0, -12.166806753664963, 8.981180114969149E-9, -18.43304034681362, 1.0000858199674545, -1.8557049457455335E-4, -1.0745034639825235E-8, 6.0699617554504925E-9, 4.0514136490088504E-10, -51.788471377742, 1.4638855537290325E-4, 1.000011339214754, -3.2257331419698054E-9, -1.0272403732974542E-8, 1.5950476934807438E-8, 0.0, 0.0, 0.0, 0.0, 0])
    xmatch.images.add_row([1, 'lsc1m005-fa15-20190309-0220-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 2458552.876477259, '2019-03-10T08:59:50.594', 300.0, '17:51:20.612', '-30:03:38.93', 121.9294215, 0.1288553, 1.1626025, 2.383945680326521, 2.54539354930874, 2045.2229424690008, -52.60656380124374, 5.8038534133512645, 0.1128353104292111, 12, 0.0, 0.1263662725687027, 0, 0, 8.0, 86.7, 2.167622294961082, 8.606163164331833E-9, 0.050299781565681136, 915856.0, -14.050093579585035,	8.606163164331833E-9, -7.605396720686155, 0.9999055334976249, -4.261665386078603E-5, 1.5106492723404585E-9, -2.0576077952583205E-8, 6.65680444100758E-9, -86.60983141547689, 1.6219744233289237E-4, 0.9998920953940439, -1.5224476203242734E-8, -8.032568058510652E-10, -2.3931440996580438E-8, 0.0, 0.0, 0.0, 0.0, 0])
    xmatch.images.add_row([2, 'lsc1m005-fa15-20190610-0183-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 2458645.6937665534, '2019-06-11T04:27:05.332', 300.027, '17:51:20.615', '-30:03:38.96', 87.6382446, 0.5982815, 1.0176447, 1.9404641389250232, 1.7089054606217395, 2005.4675269716945, -52.60656380124374, 4.296804342369643, -0.17466349715857915, 48, 0.0, 0.09608365595340729, 1, 1, 3.3, 72.4, 2.2837070043503624, 9.296043721330528E-9, 0.06700179222479302, 915856.0, -11.447228364283028,	9.296043721330528E-9, -3.4212810775671136, 1.0001679141032906, -2.049611596522576E-5, -2.4784541308164876E-8, -8.482666791209681E-9, -4.160338407110942E-9, -72.59869174381467, 8.396996369182337E-5, 1.0000973476604513, -1.3550914526279986E-8, -2.856476388554013E-8, 2.2704431524146838E-9, 0.0, 0.0, 0.0, 0.0, 0])
    xmatch.images.add_row([3, 'lsc1m005-fa15-20190312-0280-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 0.0, '2019-03-13T08:45:39.159', 300.0, '17:51:20.613', '-30:03:38.92', 158.2386538, 0.3840027, 0.0, 1.5830877462924673, 1.8747266352511662, 2067.0767588242697, -1.0, 4.071265308827622, 0.18246048853211813, 149, 0.0, 0.12208664417266846, 1, 1, 17.4, 134.5, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -16.976012749332366, 0.9999320870095413, -1.055604871449739E-4, 2.4381809904383545E-9, 1.1600962745994536E-8, 1.8700963000384263E-9, -134.42219826015767, 2.3069702204337277E-4, 0.9998957453208073, -2.541285830815615E-8, 9.942425205182459E-9, -2.8921774750340166E-8, 0.0, 0.0, 0.0, 0.0, -1])
    xmatch.images.add_row([4, 'lsc1m005-fa15-20190612-0218-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 2458647.6250936626, '2019-06-13T02:48:09.619', 299.979, '17:51:20.615', '-30:03:38.97', 60.648983, 0.7994724, 1.1580157, 2.734686023243875, 3.0468637644071723, 4341.27760716018, -52.60656380124374, 6.807254665652557, 0.22584269990532424, 10, 0.0, 0.0836634635925293, 0, 0, 94.7, 95.4, 2.181248483172304, 8.433982081279351E-9, 0.04748215494726888, 915856.0, -11.377647582983133,	8.433982081279351E-9, -94.59374288331995, 0.9999353595726668, 8.714176973207945E-5, -1.275820782353776E-8, -1.561313600628722E-8, 2.591369896443041E-9, -94.87767976537738, 3.806876464872887E-5, 0.9998777978549307, -1.8625173618316967E-8, -1.144134305363937E-9, -3.49902426984342E-8, 0.0, 0.0, 0.0, 0.0, 0])
    xmatch.images.add_row([5, 'lsc1m005-fa15-20190316-0251-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 0.0, '2019-03-17T08:32:23.503', 300.0, '17:51:20.614', '-30:03:38.97', 144.3447244, 0.8024332, 0.0, 1.6002050979758113, 1.735772755725037, 1870.7611016069714, -1.0, 3.927813759837041, 0.04486344886017897, 149, 0.0, 0.09461426734924316, 1, 1, 20.3, 114.3, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -20.04138937425921, 0.9999828649617244, -3.11191148846533E-5, -6.991440204551225E-9, 1.887235762110251E-8, 2.0590240623619138E-10, -114.3434601391381, 1.8875008366463252E-4, 0.999927921245501, -1.843532533682586E-8, -9.531182729481192E-9, -1.7979112030110447E-8, 0.0, 0.0, 0.0, 0.0, -1])
    xmatch.images.add_row([6, 'lsc1m005-fa15-20190618-0266-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 2458653.7904698513, '2019-06-19T06:46:13.974', 299.995, '17:51:21.6306', '-30:03:29.199', 22.6234763, 0.9623925, 1.1146143, 1.8653719961514892, 2.158378424823819, 10271.436654994244, -52.60656380124374, 4.737604073757189, 0.04564365146673354, 59, 0.0, 0.0215844064950943, 0, 0, 67.5, 69.7, 2.2503506661458665, 9.417413677299046E-9, 0.1102061449254334, 915856.0, -4.08859730977737,	9.417413677299046E-9, -67.11227015985745, 1.0001088769075508, -2.1141436197475073E-4, -2.110108787700682E-8, -8.423468644824794E-9, -3.847627660924502E-9, -70.23479655841429, 3.2613877428560147E-4, 0.9999979721916997, -1.5435044082429388E-8, -2.6443756620726268E-8, 1.2086094987773417E-9, 0.0, 0.0, 0.0, 0.0, 0])
    xmatch.images.add_row([7, 'lsc1m005-fa15-20190319-0260-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 2458562.846290716, '2019-03-20T08:14:57.476', 300.0, '17:51:20.613', '-30:03:39.01', 100.5201612, 0.9891269, 1.1773858, 5.6135987265357015, 3.493825725965821, 3912.550133564622, -52.60656380124374, 10.723172829677802, -0.18917868122701215, 7, 0.0, 0.12199093401432037, 0, 0, -5.0, 91.8, 1.7796332334944758, 8.886203634983367E-9, 0.44272269551448884, 915856.0, -5.361442602091852,	8.886203634983367E-9, 7.629657720527497, 0.9990309821695363, -3.682271077590945E-5, 4.658624164211034E-8, 4.657293867227352E-8, 7.061219897686044E-8, -91.56537302203863, -3.3856103197121536E-4, 0.9986170336819505, 3.958342035695155E-8, 3.0704756420299223E-7, 1.0154123962281315E-7, 0.0, 0.0, 0.0, 0.0, 0])
    xmatch.images.add_row([8, 'lsc1m005-fa15-20190619-0181-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 0.0, '2019-06-20T02:28:37.548', 300.06, '17:51:19.5737', '-30:03:44.854', 33.0383345, 0.9215497, 0.0, 1.5610495361073482, 1.856372201419589, 14421.926760705355, -1.0, 4.023706604926464, 0.056896837635554795, 105, 0.0, 0.02690390683710575, 0, 0, -1.7, 108.3, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 1.7243318110850725, 1.000048201125242, -4.84561175098343E-5, 7.350036468345422E-9, -1.2305004237767037E-8, 1.3153943489996323E-8, -108.27762126601613, 1.6919205496381302E-4, 1.000075466965181, -2.9691352902222945E-8, -5.458042208570258E-10, -4.6731779284161015E-8, 0.0, 0.0, 0.0, 0.0, -1])
    xmatch.images.add_row([9, 'lsc1m005-fa15-20190330-0112-e91.fits', 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip', 'ip', 0.0, '2019-03-31T07:31:41.521', 300.0, '17:51:20.616', '-30:03:38.94', 47.2399092, 0.2119034, 0.0, 3.97886684754194, 4.128551831444028, 3734.665514229217, -1.0, 9.545756009367249, 0.07877908291332728, 1, 0.0, 0.07361027598381042, 0, 0, 43.8, 124.5, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -43.67524318366547, 0.999944531928178, 5.396892249585325E-5, -4.314970603047641E-9, -1.3897344386393229E-8, -2.0544379530917922E-9, -124.50570505938127, 6.48676607485464E-5, 0.9999701739769572, -7.31099722051814E-9, -1.906219996981222E-9, -2.3821769754901355E-8, 0.0, 0.0, 0.0, 0.0, -1])

    return xmatch

def test_metadata():
    dataset_metadata = metadata.MetaData()
    nstars = 10
    star_catalog = np.zeros((nstars,21))
    for j in range(1,len(star_catalog),1):
        star_catalog[j-1,0] = j
        star_catalog[j-1,1] = 250.0
        star_catalog[j-1,2] = -17.5
    star_catalog[1,13] = '4062470305390995584'
    dataset_metadata.create_star_catalog_layer(data=star_catalog)


    return dataset_metadata

def test_create():
    params = test_params()

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch.init_field_index(primary_metadata)
    for key in params['datasets'].keys():
        assert(key+'_index' in xmatch.field_index.colnames)
    assert(len(xmatch.datasets) == len(params['datasets']))

def test_add_dataset():

    params = {'primary_ref_dir': '/Users/rstreet1/OMEGA/test_data/primary_ref_dataset/',
              'primary_ref_filter': 'ip',
              'red_dir_list': [ '/Users/rstreet1/OMEGA/test_data/non_ref_dataset/' ],
              'red_dataset_filters': [ 'rp' ],
              'file_path': 'crossmatch_table.fits'}

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    test_keys = ['PRIMARY', 'PRIMFILT', 'DATASET0', 'FILTER0']
    assert(x in xmatch.header.keys() for x in test_keys)
    assert(len(xmatch.matched_stars) == 1)

    xmatch.add_dataset('/Users/rstreet1/OMEGA/test_data/non_ref_dataset2/', 'rp')
    assert(len(xmatch.matched_stars) == 2)

def test_dataset_index():

    params = {'primary_ref_dir': '/Users/rstreet1/OMEGA/test_data/primary_ref_dataset',
              'primary_ref_filter': 'ip',
              'red_dir_list': [ '/Users/rstreet1/OMEGA/test_data/non_ref_dataset' ],
              'red_dataset_filters': [ 'rp' ],
              'file_path': 'crossmatch_table.fits'}

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    dataset_idx = xmatch.dataset_index(params['red_dir_list'][0]+'/')
    assert(dataset_idx == 0)

def test_save():

    params = test_params()

    if path.isfile(params['file_path']):
      remove(params['file_path'])

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    row = [1, 'cpt1m010-fl16-20170720-0104-e91.fits', 'i', 2456655.5000]
    xmatch.images.add_row(row)

    row = [1, 256.5, -27.2, \
            17.0, 0.02, 16.7, 0.02, 16.0, 0.02, \
            17.0, 0.02, 16.7, 0.02, 16.0, 0.02, \
            17.0, 0.02, 16.7, 0.02, 16.0, 0.02, \
            '4062470305390987584', 256.5, 0.001, -27.2, 0.001] + [0.0]*13
    xmatch.stars.add_row(row)

    xmatch.save(params['file_path'])

    assert(path.isfile(params['file_path']))

def test_load():
    params = test_params()
    test_save()

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['file_path'])

    assert(xmatch.datasets != None)
    assert(xmatch.field_index != None)
    assert(type(xmatch.datasets) == type(Table()))
    assert(type(xmatch.field_index) == type(Table()))
    assert(len(xmatch.datasets) > 0)
    assert(len(xmatch.field_index) > 0)

def test_init_field_index():
    params = test_params()
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    nstars = 10
    m = metadata.MetaData()
    star_catalog = np.zeros((nstars,21))
    for j in range(1,len(star_catalog),1):
        star_catalog[j-1,0] = j
        star_catalog[j-1,1] = 250.0
        star_catalog[j-1,2] = -17.5
    m.create_star_catalog_layer(data=star_catalog)

    xmatch.init_field_index(m, 'ip')

    assert len(xmatch.field_index) == nstars

def test_init_stars_table():
    params = test_params()
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    xmatch.init_stars_table()

    assert(len(xmatch.field_index) == len(xmatch.stars))

def test_update_field_index():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)
    dataset_code = 'dataset0'
    matched_stars = test_matched_stars()
    orphans = test_orphans()
    dataset_metadata = test_metadata()

    xmatch.update_field_index(dataset_code, matched_stars, orphans,
                             dataset_metadata, log)

    # Check that the index of matched star from dataset0 has been added to the
    # row entry for the corresponding star:
    assert(xmatch.field_index[dataset_code+'_index'][1] == 1)
    # Check that an additional entry has been added to the field index for
    # the orphan object:
    assert(len(xmatch.field_index) == 2)

    logs.close_log(log)

def test_assign_quadrants():
    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )
    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    row = [1, 250.0, -27.5, 0, 0, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    xmatch.assign_stars_to_quadrants()

    assert(xmatch.field_index['quadrant'][0] != 0)
    assert(xmatch.field_index['quadrant_id'][0] != 0)

    logs.close_log(log)

def test_cone_search():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    row = [1, 250.0, -27.5, 1, 1, '4062470305390987584', 1, 0, 0, 0]
    xmatch.field_index.add_row(row)

    idx = xmatch.cone_search(250.0, -27.5, 2.0, log=log)
    print(idx)
    assert(len(idx) == 1)
    logs.close_log(log)

def test_match_field_index_with_gaia_catalog():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_field_index(xmatch)
    xmatch = test_stars_table(xmatch)

    (gaia_data, gaia_star_field_ids) = test_gaia_catalog()

    xmatch.match_field_index_with_gaia_catalog(gaia_data, params, log)

    for g, gaia_field_id in enumerate(gaia_star_field_ids):
        assert(int(xmatch.field_index['gaia_source_id'][gaia_field_id-1]) == int(gaia_data['source_id'][g]))
        assert(int(xmatch.stars['gaia_source_id'][gaia_field_id-1]) == int(gaia_data['source_id'][g]))
        assert(xmatch.stars['gaia_ra'][gaia_field_id-1] == gaia_data['ra'][g])
        assert(xmatch.stars['gaia_dec'][gaia_field_id-1] == gaia_data['dec'][g])

    logs.close_log(log)

def test_load_gaia_catalog_file():

    params = {}
    params['gaia_catalog_file'] = '/Users/rstreet1/ROMEREA/test_data/config/ROME-FIELD-01_Gaia_EDR3.fits'
    params['log_dir'] = '.'

    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    gaia_data = crossmatch_field_gaia.load_gaia_catalog(params,log)

    assert(type(gaia_data) == type(Table()))

    logs.close_log(log)

def test_record_dataset_stamps():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    meta = metadata.MetaData()
    meta = test_field_photometry.test_star_catalog(meta)
    meta = test_field_photometry.test_headers_summary(meta)
    meta = test_field_photometry.test_images_stats(meta)
    meta = test_field_photometry.test_stamps_table(meta)

    xmatch.record_dataset_stamps('dataset0', meta, log)

    assert('stamps' in dir(xmatch))
    assert(type(xmatch.stamps) == type(Table()))
    columns = ['dataset_code', 'filename', 'stamp_id', 'xmin', 'xmax', 'ymin', 'ymax',\
                'warp_matrix_0', 'warp_matrix_1', 'warp_matrix_2', \
                'warp_matrix_3', 'warp_matrix_4', 'warp_matrix_5', \
                'warp_matrix_6', 'warp_matrix_7', 'warp_matrix_8']
    for column in columns:
        assert(column in xmatch.stamps.colnames)
    assert(len(xmatch.stamps) == len(meta.images_stats[1])*len(meta.stamps[1]))

def test_find_matching_images():
    params = test_params()

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_field_index(xmatch)
    xmatch = test_stars_table(xmatch)
    xmatch = test_images_table(xmatch)

    search_criteria = {'filter': 'ip', 'exposure': 300.0}

    test_idx = []
    for i in range(0,len(xmatch.images),1):
        select = True
        for key, value in search_criteria.items():
            if xmatch.images[i][key] != value:
                select = False
        if select:
            test_idx.append(i)
    test_idx = np.array(test_idx)

    image_index = xmatch.find_matching_images(search_criteria, log=None)

    assert( (np.unique(image_index)==test_idx).all() )

if __name__ == '__main__':
    #test_create()
#    test_add_dataset()
#    test_dataset_index()
    #test_save()
    #test_load()
    #test_init_field_index()
    #test_update_field_index()
    #test_assign_quadrants()
    #test_cone_search()
    #test_init_stars_table()
    #test_match_field_index_with_gaia_catalog()
    #test_load_gaia_catalog_file()
    #test_record_dataset_stamps()
    test_find_matching_images()
