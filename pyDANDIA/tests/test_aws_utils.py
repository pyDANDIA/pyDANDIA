from pyDANDIA import aws_utils
from pyDANDIA import logs
from os import getcwd, path
from datetime import datetime

CWD = getcwd()

def load_default_config():

    config = {
            'red_dir': '/Users/rstreet1/ROMEREA/test_data/MOA-2021-BLG-067_ip',
            'aws_bucket': 'robonet.lco.global',
            'awsid': 'roboarchiver',
            'local_data_dir': '/Users/rstreet1/ROMEREA/test_data/incoming',
    }

    config = aws_utils.get_credentials(config)

    return config


def test_list_available_lightcurves():

    config = load_default_config()

    s3_client = aws_utils.start_s3_client(config)

    file_list = aws_utils.list_available_lightcurves(config, s3_client)

    assert type(file_list) == type([])
    assert len(file_list) > 0

def test_search_for_event_data():

    config = load_default_config()

    log = logs.start_stage_log( CWD, 'test_aws' )

    s3_client = aws_utils.start_s3_client(config)

    event_name = 'MOA-2021-BLG-067'

    event_files = aws_utils.search_for_event_data(config, s3_client, event_name, log)

    assert type(event_files) == type([])
    for entry in event_files:
        assert event_name in entry

    logs.close_log(log)

def test_remove_files():

    config = load_default_config()

    log = logs.start_stage_log( CWD, 'test_aws' )

    test_file = 'AWS_TEST_UPLOAD.txt'
    f = open(test_file,'w')
    f.write('AWS TEST UPLOAD')
    f.write(datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))
    f.close()

    s3_client = aws_utils.start_s3_client(config)

    aws_utils.upload_file_to_aws(config, test_file, s3_client, log=log)

    event_files = aws_utils.search_for_event_data(config, s3_client, test_file, log)
    assert(len(event_files) > 0)

    aws_utils.remove_files(config, s3_client, event_files, log=log)

    event_files = aws_utils.search_for_event_data(config, s3_client, test_file, log)
    assert(len(event_files) == 0)

    logs.close_log(log)

def test_remove_old_reduction_data_products():

    config = load_default_config()
    event_name = 'TEST-2020-BLG-XXXX_ip'
    config['red_dir'] = path.join('/Users/rstreet1/ROMEREA/test_data/',event_name)

    log = logs.start_stage_log( CWD, 'test_aws' )
    test_file = event_name+'.txt'
    f = open(test_file,'w')
    f.write('AWS TEST UPLOAD')
    f.write(datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))
    f.close()

    s3_client = aws_utils.start_s3_client(config)

    aws_utils.upload_file_to_aws(config, test_file, s3_client, log=log)
    key_list = aws_utils.search_for_event_data(config, s3_client, event_name, log=log)
    assert(len(key_list) > 0)

    aws_utils.remove_old_reduction_data_products(config, log=log)
    key_list = aws_utils.search_for_event_data(config, s3_client, event_name, log=log)
    assert(len(key_list) == 0)

if __name__ == '__main__':
    test_list_available_lightcurves()
    test_search_for_event_data()
    test_remove_files()
    test_remove_old_reduction_data_products()
