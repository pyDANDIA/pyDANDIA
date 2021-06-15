import numpy as np
from pyDANDIA import field_photometry
from pyDANDIA import crossmatch
from pyDANDIA import logs
from pyDANDIA import metadata
from astropy.table import Table, Column
from astropy.io import fits
import astropy.units as u
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

def test_images_table(xmatch):
    nimages = 10

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

def test_xmatch_stamps_table(xmatch):

    dataset_code = 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip'
    images = ['lsc1m005-fa15-20190610-0205-e91.fits',
              'lsc1m005-fa15-20190610-0183-e91.fits',
              'lsc1m005-fa15-20190612-0218-e91.fits']
    stamps = [ [0, 0, 1010, 0, 1010],
               [1, 990, 2010, 0, 1010],
               [2, 1990, 3010, 0, 1010],
               [3, 2990, 4096, 0, 1010],
               [4, 0, 1010, 990, 2010],
               [5, 990, 2010, 990, 2010],
               [6, 1990, 3010, 990, 2010],
               [7, 2990, 4096, 990, 2010],
               [8, 0, 1010, 1990, 3010],
               [9, 990, 2010, 1990, 3010],
               [10, 1990, 3010, 1990, 3010],
               [11, 2990, 4096, 1990, 3010],
               [12, 0, 1010, 2990, 4096],
               [13, 990, 2010, 2990, 4096],
               [14, 1990, 3010, 2990, 4096],
               [15, 2990, 4096, 2990, 4096] ]

    for image in images:
        for params in stamps:
            xmatch.stamps.add_row( [dataset_code, image] + params + [0.0]*9 )

    return xmatch

def test_star_catalog(meta):

    nstars = 11
    star_catalog = np.zeros((nstars,37))
    star_catalog[0] = [1, 346.7463278676221, 0.31769726490029015, 267.61861696019145, -29.829605383706895, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056436121079692160, 267.6188626384401, 8.781465530395508, -29.829782879978076, 7.436490535736084, 286.91259765625, 2.5169389247894287, -9999.999, -9999.999, -9999.999, -9999.999, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10]
    star_catalog[1] = [2, 1017.3340457345882, 8.379706811851744, 267.70228408545813, -29.83032824102953, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10]
    star_catalog[2] = [3, 3301.8400837300433, 11.255029683618215, 267.9873108673885, -29.829734325692858, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444539267431040, 267.98727887564667, 0.045853544026613235, -29.829912824096596, 0.03865557536482811, 23195.255859375, 10.54239273071289, 11357.7841796875, 28.62028694152832, 18024.580078125, 40.84840774536133, None, 267.98729, -29.82988, 15.5, 0.0, 14.619999885559082, 0.0, 13.989999771118164, 0.0, 0.0, 1.0E10]
    star_catalog[3] = [4, 3070.9685725169215, 12.842726035986505, 267.9585073984874, -29.83002538112054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444878525743616, 267.95849533448626, 0.12481509894132614, -29.830130309157735, 0.1085614413022995, 7796.68701171875, 9.380874633789062, 783.80517578125, 17.880268096923828, 13934.6943359375, 136.3885955810547, None, 267.95851, -29.83011, 20.25, 0.029999999329447746, 16.850000381469727, 0.009999999776482582, 14.739999771118164, 0.009999999776482582, 0.0, 1.0E10]
    star_catalog[4] = [5, 3101.7435341021883, 12.21388380362664, 267.9623466389135, -29.82994179424344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444917224603904, 267.9623198050811, 0.22476719319820404, -29.830078194809662, 0.18630900979042053, 11072.041015625, 46.79934310913086, 1684.195068359375, 49.38384246826172, 19294.3125, 126.06063079833984, None, 267.96233, -29.83006, 19.600000381469727, 0.019999999552965164, 16.270000457763672, 0.009999999776482582, 14.380000114440918, 0.009999999776482582, 0.0, 1.0E10]
    star_catalog[5] = [6, 2855.1098226019617, 20.722212446330264, 267.9315803167322, -29.830983939264463, 72.83591222434477, 1.2387420738083867, 20.344136090341717, 0.018465439325429298, 17.999026801794326, 0.018465439325429298, 631.5231558024683, 10.740502587064343, 1085.0, 5179.085342983518, 4056444844165932544, 267.93156186163185, 0.3745168447494507, -29.831130541812637, 0.3244278132915497, 599.3142700195312, 2.3796072006225586, 332.97344970703125, 16.70027732849121, 1802.3887939453125, 43.210113525390625, None, 267.93157, -29.8311, 21.959999084472656, 0.10999999940395355, 18.959999084472656, 0.029999999329447746, 17.450000762939453, 0.019999999552965164, 0.0, 0.0]
    star_catalog[6] = [7, 2947.713447326438, 19.952506222613422, 267.94313361856774, -29.830855906070912, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E10, 0.0, 0.0, 0.0, 4056444947245158016, 267.94308300945704, 0.8784804344177246, -29.83089780473822, 0.6652926206588745, 475.24163818359375, 2.5971155166625977, -9999.999, -9999.999, -9999.999, -9999.999, None, 267.9431, -29.83087, 21.969999313354492, 0.11999999731779099, 19.329999923706055, 0.029999999329447746, 17.809999465942383, 0.019999999552965164, 0.0, 1.0E10]
    star_catalog[7] = [8, 3116.475668245582, 22.523958538511042, 267.9641908032068, -29.83105008127081, 67.91323518000698, 1.241420694285689, 20.420113951429993, 0.01984672633470334, 18.087107346368825, 0.01984672633470334, 582.3137719332096, 10.644410697404933, 1085.0, 5179.085342983518, 4056444912885400832, 267.96420196637877, 0.4867507517337799, -29.831163227678985, 0.4509122669696808, 874.8128662109375, 7.869678020477295, 415.13970947265625, 33.829490661621094, 2185.14501953125, 55.643558502197266, None, 267.96421, -29.83113, 21.739999771118164, 0.18000000715255737, 18.719999313354492, 0.019999999552965164, 17.100000381469727, 0.019999999552965164, 0.0, 0.0]
    star_catalog[8] = [9, 3158.045169151278, 22.44803844400929, 267.96937719314764, -29.831020635327544, 66.76936467853554, 1.2428182815630278, 20.438556890325344, 0.020209459992990574, 18.10848810226519, 0.020209459992990574, 570.9587843722417, 10.627598729645248, 1085.0, 5179.085342983518, 4056444912885534976, 267.96940117953943, 0.2777192294597626, -29.831159569166665, 0.24106402695178986, 1009.09765625, 3.348869800567627, 470.4875183105469, 19.688159942626953, 3108.443359375, 116.98040008544922, None, 267.96941, -29.83114, 20.81999969482422, 0.05000000074505806, 18.3799991607666, 0.019999999552965164, 17.010000228881836, 0.009999999776482582, 1.0, 0.0]
    star_catalog[9] = [10, 3172.822182455274, 21.96392772157342, 267.97122056100426, -29.83096064913158, 1389.5969629009883, 1.3872958687360546, 17.14277785938164, 0.001083938286863857, 14.287717055480321, 0.001083938286863857, 19271.39609627774, 19.23948375169784, 1085.0, 5179.085342983518, 4056444912869653504, 267.97119999632883, 0.051752395927906036, -29.831062674301922, 0.04540014639496803, 27424.91015625, 19.671449661254883, 7664.77392578125, 49.09895706176758, 32068.123046875, 50.958839416503906, None, 267.97121, -29.83104, 16.479999542236328, 0.009999999776482582, 14.539999961853027, 0.0, 13.479999542236328, 0.009999999776482582, 0.0, 0.0]
    star_catalog[10] = [11, 3351.0537899352307, 22.33141678451542, 267.9934578741869, -29.83090679218163, 2.810801157316224, 1.2340045184832593, 23.877924690797105, 0.4766624558500083, 22.095720494942896, 0.4766624558500083, 14.511483154459642, 6.370865379746347, 1085.0, 5179.085342983518, 4056444534928464128, 267.9934558649592, 0.794064462184906, -29.83104086736632, 0.9368880391120911, 394.9053039550781, 3.421269655227661, 247.20770263671875, 20.210479736328125, 1386.758544921875, 	44.8889045715332, None, 267.99345, -29.83099, 22.610000610351562, 0.3100000023841858, 19.510000228881836, 0.05000000074505806, 17.8700008392334, 0.029999999329447746, 0.0, 0.0]

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
            data.append(data[13])
            data.append(data[14])
            data.append(0.0)
            image_data.append(data)
        photometry.append(image_data)

    return np.array(photometry)

def test_headers_summary(meta):
    names = np.array(['IMAGES', 'EXPKEY', 'OBJKEY', 'OBSKEY', 'DATEKEY', \
                        'TIMEKEY', 'RAKEY', 'DECKEY', 'FILTKEY', \
                        'MOONDKEY', 'MOONFKEY', 'AIRMASS', 'HJD'])
    formats = np.array(['S200']*11 + ['float']*2)
    data = np.array([['lsc1m005-fa15-20190610-0205-e91.fits', '300.0', 'ROME-FIELD-01',
            'EXPOSE', '2019-06-13T02:36:47.449', '02:36:47.449', '17:51:22.5696',
            '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.91', '2452103.2'],
            ['lsc1m005-fa15-20190610-0183-e91.fits', '300.0', 'ROME-FIELD-01',
        'EXPOSE', '2019-06-13T02:37:00.449', '02:36:47.449', '17:51:22.5696',
        '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.92', '2452103.21'],
        ['lsc1m005-fa15-20190612-0218-e91.fits', '300.0', 'ROME-FIELD-01',
    'EXPOSE', '2019-06-13T02:38:00.449', '02:36:47.449', '17:51:22.5696',
    '-30:03:37.550', 'ip', '60.7240446', '0.799005', '0.93', '2452103.22']
            ])
    meta.create_headers_summary_layer(names, formats, units=None, data=data)

    return meta

def test_data_architecture(meta):

    table_data = [
                Column(name='METADATA_NAME', data=np.array(['pyDANDIA_metadata.fits']), unit=None, dtype='S100'),
                Column(name='OUTPUT_DIRECTORY', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/']), unit=None, dtype='S100'),
                Column(name='IMAGES_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/data']), unit=None, dtype='S100'),
                Column(name='BPM_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/data']), unit=None, dtype='S100'),
                Column(name='REF_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/ref']), unit=None, dtype='S100'),
                Column(name='REF_IMAGE', data=np.array(['lsc1m005-fl15-20180710-0088-e91.fits']), unit=None, dtype='S100'),
                Column(name='KERNEL_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/kernel']), unit=None, dtype='S100'),
                Column(name='DIFFIM_PATH', data=np.array(['/Users/rstreet1/ROMEREA/test_data/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/diff_images']), unit=None, dtype='S100'),
                ]

    layer_header = fits.Header()
    layer_header.update({'NAME': 'data_architecture'})
    layer_table = Table(table_data)
    layer = [layer_header, layer_table]

    setattr(meta, 'data_architecture', layer)

    return meta

def test_images_stats(meta):

    table_data = [ Column(name='IM_NAME', data=np.array(['lsc1m005-fa15-20190610-0205-e91.fits','lsc1m005-fa15-20190610-0183-e91.fits','lsc1m005-fa15-20190612-0218-e91.fits']), unit=None, dtype='S100'),
                   Column(name='SIGMA_X', data=np.array([1.8786789649990936,2.383945680326521,1.9404641389250232]), unit=u.pix, dtype='float'),
                   Column(name='SIGMA_Y', data=np.array([1.7507988428740193,2.54539354930874,1.7089054606217395]), unit=u.pix, dtype='float'),
                   Column(name='FWHM', data=np.array([4.273383547487297, 5.8038534133512645, 4.296804342369643]), unit=u.pix, dtype='float'),
                   Column(name='SKY', data=np.array([2180.902135178605, 2045.2229424690008,2005.4675269716945]), unit=u.adu, dtype='float'),
                   Column(name='CORR_XY', data=np.array([-0.09175497562167338, 0.1128353104292111,-0.17466349715857915]), unit=None, dtype='float'),
                   Column(name='NSTARS', data=np.array([20, 12, 48]), unit=None, dtype='int'),
                   Column(name='FRAC_SAT_PIX', data=np.array([0.0, 0.0, 0.0]), unit=None, dtype='float'),
                   Column(name='SYMMETRY', data=np.array([0.06594043970108032, 0.1263662725687027,0.09608365595340729]), unit=None, dtype='float'),
                   Column(name='USE_PHOT', data=np.array([1,0,1]), unit=None, dtype='int'),
                   Column(name='USE_REF', data=np.array([1,0,1]), unit=None, dtype='int'),
                   Column(name='SHIFT_X', data=np.array([18.7, 8.0, 3.3]), unit=None, dtype='float'),
                   Column(name='SHIFT_Y', data=np.array([51.4, 86.7, 72.4]), unit=None, dtype='float'),
                   Column(name='PSCALE', data=np.array([1.304527793209856, 2.1691397412945386, 2.282448295973614]), unit=None, dtype='float'),
                   Column(name='PSCALE_ERR', data=np.array([9.88518365653761E-9, 1.1109174691900272E-8, 9.202312487729357E-9]), unit=None, dtype='float'),
                   Column(name='MEDIAN_SKY', data=np.array([-73.95321752298457, -73.95321752298457, -73.95321752298457]), unit=None, dtype='float'),
                   Column(name='VAR_PER_PIX_DIFF', data=np.array([0.0457529108700811, 0.15137651432510715, 0.11725083370344544]), unit=None, dtype='float'),
                   Column(name='N_UNMASKED', data=np.array([1013100.0, 1013100.0, 1013100.0]), unit=None, dtype='float'),
                   Column(name='SKEW_DIFF', data=np.array([6.172574787633106, -0.03582948547777698, 7.67133980818663]), unit=None, dtype='float'),
                   Column(name='KURTOSIS_DIFF', data=np.array([9.88518365653761E-9, 1.1109174691900272E-8, 9.202312487729357E-9]), unit=None, dtype='float'),
                   ]

    layer_header = fits.Header()
    layer_header.update({'NAME': 'images_stats'})
    layer_table = Table(table_data)
    layer = [layer_header, layer_table]

    setattr(meta, 'images_stats', layer)

    return meta

def test_reduction_status(meta):
    nimages = len(meta.images_stats[1]['IM_NAME'].data)

    table_data = [Column(name='IMAGES', data=meta.images_stats[1]['IM_NAME'].data),
                  Column(name='STAGE_0', data=np.ones(nimages)),
                  Column(name='STAGE_1', data=np.ones(nimages)),
                  Column(name='STAGE_2', data=np.ones(nimages)),
                  Column(name='STAGE_3', data=np.ones(nimages)),
                  Column(name='STAGE_4', data=np.ones(nimages)),
                  Column(name='STAGE_5', data=np.ones(nimages)),
                  Column(name='STAGE_6', data=np.ones(nimages)),
                  Column(name='STAGE_7', data=np.zeros(nimages))]

    layer_table = Table(table_data)
    layer_table['STAGE_4'][1] = -1
    layer_table['STAGE_5'][1] = -1
    layer_table['STAGE_6'][1] = -1
    layer_table['STAGE_7'][1] = -1
    layer_header = fits.Header()
    layer_header.update({'NAME': 'reduction_status'})
    layer = [layer_header, layer_table]

    setattr(meta, 'reduction_status', layer)

    return meta

def test_stamps_table(meta):

    table_data = [Column(name='PIXEL_INDEX', data=np.arange(0,16,1), unit=None, dtype='int'),
                  Column(name='Y_MIN', data=np.array([0,0,0,0,990,990,990,990,1990,1990,1990,1990,2990,2990,2990,2990]), unit=None, dtype='int'),
                  Column(name='Y_MAX', data=np.array([1010,1010,1010,1010,2010,2010,2010,2010,3010,3010,3010,3010,4096,4096,4096,4096]), unit=None, dtype='int'),
                  Column(name='X_MIN', data=np.array([0,990,1990,2990,0,990,1990,2990,0,990,1990,2990,0,990,1990,2990]), unit=None, dtype='int'),
                  Column(name='X_MAX', data=np.array([1010,2010,3010,4096,1010,2010,3010,4096,1010,2010,3010,4096,1010,2010,3010,4096]), unit=None, dtype='int'),
                   ]
    layer_header = fits.Header()
    layer_header.update({'NAME': 'stamps'})
    layer_table = Table(table_data)
    layer = [layer_header, layer_table]

    setattr(meta, 'stamps', layer)

    return meta

def test_photometry_array():
    """Function to build a test data photometric array comprised of the following
    columns:

    hjd, instrumental_mag, instrumental_mag_err,
    calibrated_mag, calibrated_mag_err, corrected_mag, corrected_mag_err,
    phot_scale_factor, phot_scale_factor_err, stamp_index,
    sub_image_sky_bkgd, sub_image_sky_bkgd_err,
    residual_x, residual_y, qc_flag
    """

    nstars = 100
    nimages = 10
    ncol = 15
    photometry = np.zeros( (nstars, nimages, ncol) )

    for i in range(0,nimages,1):
        photometry[:,i,0] = 2455678.0 + i*0.01

    for j in range(0,nstars,1):
        base_mag = 14.0 + j*0.01
        photometry[j,:,1] = base_mag + np.random.rand(nimages)
        photometry[j,:,2] = np.random.rand(nimages)
        photometry[j,:,3] = photometry[j,:,1]
        photometry[j,:,4] = photometry[j,:,2]
        photometry[j,:,5] = photometry[j,:,1]
        photometry[j,:,6] = photometry[j,:,2]
        photometry[j,:,7] = np.random.rand(nimages)
        photometry[j,:,8] = np.random.rand(nimages)*0.1
        photometry[j,:,9] = np.random.randint(0,15)
        photometry[j,:,10] = 500.0 + np.random.rand(nimages)
        photometry[j,:,11] = np.random.rand(nimages)
        photometry[j,:,12] = np.random.rand(nimages)
        photometry[j,:,13] = np.random.rand(nimages)
        photometry[j,:,14] = np.random.randint(0,2)

    return photometry

def test_init_field_data_table():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    photometry = field_photometry.init_field_data_table(xmatch, log)

    assert(type(photometry) == type(np.array([])))
    logs.close_log(log)

def test_populate_images_table():
    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)

    meta = metadata.MetaData()
    meta = test_headers_summary(meta)
    meta = test_images_stats(meta)
    meta = test_reduction_status(meta)
    meta = test_data_architecture(meta)

    (xmatch, image_index) = field_photometry.populate_images_table(xmatch.datasets[0], meta, xmatch, log)

    assert(len(xmatch.images) == len(meta.headers_summary[1]))

    keys = ['index', 'filename', 'dataset_code', 'filter', 'hjd', 'datetime', \
            'exposure', 'RA', 'Dec', 'moon_ang_separation', 'moon_fraction', 'airmass',\
            'sigma_x', 'sigma_y', 'sky', 'median_sky', 'fwhm', \
            'corr_xy', 'nstars', 'frac_sat_pix', 'symmetry',  \
            'use_phot', 'use_ref', 'shift_x', 'shift_y', \
            'pscale', 'pscale_err', 'var_per_pix_diff', 'n_unmasked',\
            'skew_diff', 'kurtosis_diff']
    for key in keys:
        assert(key in xmatch.images.colnames)

    for image in meta.headers_summary[1]:
        i = np.where(xmatch.images['filename'] == image['IMAGES'])[0]
        assert(len(i) > 0)
        assert(xmatch.images['filter'][i] == image['FILTKEY'])
        assert(xmatch.images['dataset_code'][i] == xmatch.datasets[0]['dataset_code'])
        assert(xmatch.images['airmass'][i] == image['AIRMASS'])

    stats_keys = ['sigma_x', 'sigma_y', 'sky', 'median_sky', 'fwhm', \
                    'corr_xy', 'nstars', 'frac_sat_pix', 'symmetry',  \
                    'use_phot', 'use_ref', 'shift_x', 'shift_y', \
                    'pscale', 'pscale_err', 'var_per_pix_diff', 'n_unmasked',\
                    'skew_diff', 'kurtosis_diff']
    for image in meta.images_stats[1]:
        i = np.where(xmatch.images['filename'] == image['IM_NAME'])[0]
        for key in stats_keys:
            assert(xmatch.images[key][i] == image[key.upper()])

    red_dir = meta.data_architecture[1]['OUTPUT_DIRECTORY'][0]
    for image in meta.reduction_status[1]:
        i = np.where(xmatch.images['filename'] == image['IMAGES'])[0]
        for k in range(0,7,1):
            image['STAGE_'+str(k)], xmatch.images[i]['qc_flag']
            if image['STAGE_'+str(k)] == -1:
                assert(xmatch.images[i]['qc_flag'] == int(image['STAGE_'+str(k)]))
        matrix_file = path.join(red_dir, 'resampled', image['IMAGES'], 'warp_matrice_image.npy')
        if path.isfile(matrix_file):
            matrix = np.load(matrix_file)
            transformation = matrix.ravel()
        else:
            transformation = np.zeros(9)
        for j in range(0,len(transformation),1):
            assert(xmatch.images[i]['warp_matrix_'+str(j)] == transformation[j])

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
    for j in range(0,len(xmatch.stars),1):
        assert(xmatch.stars[mag_column][j] == meta.star_catalog[1]['cal_ref_mag'][j])
        assert(xmatch.stars[mag_error_column][j] == meta.star_catalog[1]['cal_ref_mag_error'][j])

    logs.close_log(log)

def test_populate_photometry_array():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_field_index(xmatch)
    xmatch = test_stars_table(xmatch)
    xmatch.create_images_table()

    meta = metadata.MetaData()
    meta = test_star_catalog(meta)
    meta = test_headers_summary(meta)
    meta = test_stamps_table(meta)
    meta = test_data_architecture(meta)
    meta = test_images_stats(meta)
    meta = test_reduction_status(meta)

    # Image table must be populated before the timeseries can be generated
    (xmatch, dataset_image_idx) = field_photometry.populate_images_table(xmatch.datasets[0], meta, xmatch, log)
    dataset_photometry = test_dataset_timeseries_photometry(meta, xmatch)

    # Create the dataset_photometry as an HDF5 object
    # Deactivated as no longer required for this test - the indexing requires
    # that this array be a numpy array not an HDF5
    use_array = False
    if use_array:
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

    for j in range(0,len(field_array_idx),1):
        print('Dataset star '+str(dataset_array_idx[j])+' corresponds to field star '+str(field_array_idx[j]))

    # Can be initialized only after the images and stars tables have been
    # populated with all datasets
    photometry = field_photometry.init_field_data_table(xmatch,log)

    (xmatch, photometry) = field_photometry.populate_photometry_array(field_array_idx, dataset_array_idx,
                                    dataset_image_idx, photometry, dataset_photometry, xmatch, log, meta)

    for i,iimage in enumerate(dataset_image_idx):
        assert(xmatch.images['hjd'][iimage] == dataset_photometry[0,i,9])

    for j in field_array_idx:
        star_x = meta.star_catalog[1]['x'][dataset_array_idx[j]]
        star_y = meta.star_catalog[1]['y'][dataset_array_idx[j]]
        stamp_id = calc_star_stamp(star_x, star_y, meta)
        print('Star in stamp '+str(stamp_id),dataset_array_idx[j],star_x, star_y)
        for i,iimage in enumerate(dataset_image_idx):
            print('Phot array: ',photometry[j,iimage,0])
            print('Dataset: ',dataset_photometry[dataset_array_idx[j],i,9])
            assert( photometry[j,iimage,0] == dataset_photometry[dataset_array_idx[j],i,9])
            assert( photometry[j,iimage,3] == dataset_photometry[dataset_array_idx[j],i,13])
            assert( photometry[j,iimage,4] == dataset_photometry[dataset_array_idx[j],i,14])

            print(photometry[j,iimage,9], stamp_id)
            assert( photometry[j,iimage,9] == float(stamp_id) )

    logs.close_log(log)

def calc_star_stamp(star_x, star_y, meta):

    stamp_id = -1
    for stamp in meta.stamps[1]:
        if star_x > stamp['X_MIN'] and star_x < stamp['X_MAX'] and \
            star_y > stamp['Y_MIN'] and star_y < stamp['Y_MAX']:
            stamp_id = stamp['PIXEL_INDEX']

    return stamp_id

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

def test_populate_stamps_table():

    params = test_params()
    log = logs.start_stage_log( params['log_dir'], 'test_crossmatch' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_xmatch_stamps_table(xmatch)

    meta = metadata.MetaData()
    meta = test_star_catalog(meta)
    meta = test_headers_summary(meta)
    meta = test_images_stats(meta)
    meta = test_stamps_table(meta)
    meta = test_data_architecture(meta)

    dataset_code = 'ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip'

    xmatch = field_photometry.populate_stamps_table(xmatch, dataset_code, meta, log)

    # Verify that that the coefficients 0--5 & 8 are non-zero;
    # coefficients 6 & 7 are likely supposed to be zero
    for stamp in xmatch.stamps:
        for j in range(0,6,1):
            assert(abs(stamp['warp_matrix_'+str(j)]) > 0.0)
        assert(abs(stamp['warp_matrix_'+str(8)]) > 0.0)

    logs.close_log(log)

def test_mask_phot_array_by_qcflag():

    phot_data = test_photometry_array()

    bad_data_idx = np.where(phot_data[:,:,14] > 0)

    masked_data = field_photometry.mask_phot_array_by_qcflag(phot_data)

    assert(type(masked_data) == type(np.ma.array([0])))
    mask = np.ma.getmask(masked_data)
    assert( (mask[bad_data_idx] == True).all() )

def test_extract_photometry_by_image_search():

    params = test_params()

    xmatch = crossmatch.CrossMatchTable()
    xmatch.create(params)
    xmatch = test_xmatch_stamps_table(xmatch)
    xmatch = test_field_index(xmatch)
    xmatch = test_stars_table(xmatch)
    xmatch = test_images_table(xmatch)

    phot_data = test_photometry_array()

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

    selected_phot = field_photometry.extract_photometry_by_image_search(xmatch, phot_data, search_criteria)

    assert( selected_phot.shape[1] == len(test_idx) )
    for k,i in enumerate(test_idx):
        assert( (phot_data[:,i,:] == selected_phot[:,k,:]).all() )

if __name__ == '__main__':
    #test_init_field_data_table()
    #test_populate_images_table()
    #test_populate_stars_table()
    #test_populate_photometry_array()
    #test_build_array_index_3D()
    #test_update_array_col_index()
    #test_populate_stamps_table()
    #test_mask_phot_array_by_qcflag()
    test_extract_photometry_by_search_criteria()
