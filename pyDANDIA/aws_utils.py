from os import path
from sys import argv
from pyDANDIA import logs
from pyDANDIA import config_utils
import boto3

def upload_lightcurve_aws(config, lc_file, log=None):

    config = get_credentials(config, log=log)

    s3_client = start_s3_client(config, log=log)

    upload_file_to_aws(config, lc_file, s3_client, log=log)

def upload_file_to_aws(config, lc_file, s3_client, log=None):

    if not path.isfile(lc_file):
        if log!=None:
            log.info('Error uploading lightcurve: Cannot find lightcurve file '+lc_file)

    else:

        with open(lc_file, "rb") as f:
            s3_client.upload_fileobj(f, config['aws_bucket'],
                    path.join('OMEGA/realtime_lightcurves', path.basename(lc_file)) )

        if log:
            log.info('Uploaded '+lc_file+' to AWS')

def start_s3_client(config, log=None):

    s3_client = boto3.client(
                's3',
                aws_access_key_id=config['aws_access_key_id'],
                aws_secret_access_key=config['aws_secret_access_key']
            )

    if log:
        log.info('Started S3 client')

    return s3_client


def get_credentials(config, log=None):

    home_dir = path.expanduser("~")

    credentials_file = path.join(home_dir, '.aws', 'credentials')

    if not path.isfile(credentials_file):
        if log!=None:
            log.info('Error uploading lightcurve: No AWS credentials available at '+credentials_file)

    else:
        credentials = open(credentials_file, 'r').readlines()

        for i,line in enumerate(credentials):
            if config['awsid'] in line:
                config['aws_access_key_id'] = credentials[i+1].split('=')[-1].replace('\n','').lstrip()
                config['aws_secret_access_key'] = credentials[i+2].split('=')[-1].replace('\n','').lstrip()
                break

        if 'aws_access_key_id' not in config.keys():
            if log!=None:
                log.info('No AWS credentials found for user ID '+config['awsid'])

    return config


if __name__ == '__main__':
    log = logs.start_stage_log('.', 'upload_lc')

    if len(argv) == 1:
        config_file = input('Please enter the path to this scripts configuration file: ')
        lc_file = input('Please enter the path to a lightcurve file to upload: ')
    else:
        config_file = argv[1]
        lc_file = argv[2]

    config = config_utils.build_config_from_json(config_file)

    upload_lightcurve_aws(config, lc_file, log=log)

    logs.close_log(log)
