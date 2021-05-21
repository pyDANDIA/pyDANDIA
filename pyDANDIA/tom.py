from os import path
from sys import argv
import requests
from datetime import datetime
import json

def concat_urls(base_url,extn_url):
    """Function to concatenate URL components without unnecessary duplication
    of /"""

    if base_url[-1:] == '/':
        base_url = base_url[:-1]
    if extn_url[0:1] == '/':
        extn_url = extn_url[1:]

    return base_url+'/'+extn_url

def get_target_id(config, login, payload, log=None):
    """Function queries the TOM to find the unique primary key identifier for the
    target name given.  This parameter is required for data upload."""

    targetid_url = concat_urls(config['tom_url'],config['targets_endpoint'])

    target_pk = None
    target_groups = []
    ur = {'name': payload['name']}
    response = requests.get(targetid_url, auth=login, params=ur).json()

    if 'results' in response.keys() and len(response['results']) == 1:
        target_pk = response['results'][0]['id']
        for group in response['results'][0]['groups']:
            target_groups.append(group['id'])

        if log!=None:
            log.info('TOM identified target '+payload['name']+' as target ID='+str(target_pk))

    elif 'results' in response.keys() and len(response['results']) == 0:
        if log!=None:
            log.info('Targetname '+payload['name']+' unknown to TOM')

    elif 'results' in response.keys() and len(response['results']) > 1:
        if log!=None:
            log.info('Ambiguous targetname '+payload['name']+' multiple entries in TOM')

    else:
        if log!=None:
            log.info('No response from TOM.  Check login details and URL?')

    return target_pk, target_groups

def upload_datafile(config, login, payload, target_pk, target_groups, log=None):
    """Function uploads photometry data to the TOM"""

    ur = {'target': target_pk, 'data_product_type': 'photometry', 'groups': target_groups}
    file_data = {'file': (payload['file_path'], open(payload['file_path'],'rb'))}
    dataupload_url = concat_urls(config['tom_url'],config['dataproducts_endpoint'])
    response = requests.post(dataupload_url, data=ur, files=file_data, auth=login)

    if log!= None:
        log.info('Uploaded lightcurve file to TOM at URL: '+repr(response.url))
        log.info('with response: '+repr(response.text))

def list_dataproducts(config, login, payload, target_pk, log=None):
    """Function to return a list of dataproducts for the given target that
    have already been uploaded to the TOM"""

    dataupload_url = concat_urls(config['tom_url'],config['dataproducts_endpoint'])

    #ur = {'target': target_pk, 'data_product_type': 'photometry', 'page_size': 99999}
    ur = {'data_product_type': 'photometry', 'limit': 99999}

    # List endpoint does not currently support queries specific to target ID
    #response = requests.get(dataupload_url, params=ur, auth=login).json()
    response = requests.get(dataupload_url, params=ur, auth=login).json()

    existing_datafiles = {}
    for entry in response['results']:
        if entry['target'] == target_pk:
            existing_datafiles[path.basename(entry['data'])] = entry['id']

    if log != None:
        if len(existing_datafiles) > 0:
            log.info('Found existing datafiles for target '+payload['name']+\
                    ', ID='+str(target_pk)+' in the TOM:')
            log.info(repr(existing_datafiles))
        else:
            log.info('No existing datafiles in TOM for target '+payload['name'])

    return existing_datafiles

def delete_old_datafile_version(config, login, payload, existing_datafiles, log=None):
    """Function to find and delete any existing entry in the TOM for the
    datafile to be uploaded"""

    # Due to automatic suffixes added by the data ingest processor, the only
    # way to identify datasets from the same telescope is the first
    # section of the filename, which must be distinctively named.
    #filename = path.basename(payload['file_path']).split('.')[0]
    file_pk = None
    dataupload_url = concat_urls(config['tom_url'], config['dataproducts_endpoint'])

    if log!= None:
        log.info('Searching TOM system for previous similar datafiles')

    for fname, id in existing_datafiles.items():
        if payload['search_string'] in fname:
            file_pk = id
            delete_data_url = concat_urls(dataupload_url,str(file_pk))
            response = requests.delete(delete_data_url, auth=login)
            if log!=None:
                log.info('Attempted to remove old datafile from TOM with response: '+repr(response.text))
