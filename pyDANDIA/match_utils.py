# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:49:31 2019

@author: rstreet
"""
import numpy as np

class StarMatchIndex:

    def __init__(self):

        self.cat1_index = []
        self.cat1_ra = []
        self.cat1_dec = []
        self.cat1_x = []
        self.cat1_y = []
        self.cat2_index = []
        self.cat2_ra = []
        self.cat2_dec = []
        self.cat2_x = []
        self.cat2_y = []
        self.separation = []
        self.n_match = 0

    def add_match(self,params):

        for key, value in params.items():

            l = getattr(self,key)

            l.append(value)

            setattr(self,key,l)

        self.n_match += 1

    def remove_match(self,entry_index):

        def pop_entry(attribute,index):

            l = getattr(self,attribute)

            try:
                tmp = l.pop(index)
            except IndexError:
                pass

            setattr(self,attribute,l)

        pop_entry('cat1_index',entry_index)
        pop_entry('cat1_ra',entry_index)
        pop_entry('cat1_dec',entry_index)
        pop_entry('cat1_x',entry_index)
        pop_entry('cat1_y',entry_index)

        pop_entry('cat2_index',entry_index)
        pop_entry('cat2_ra',entry_index)
        pop_entry('cat2_dec',entry_index)
        pop_entry('cat2_x',entry_index)
        pop_entry('cat2_y',entry_index)

        pop_entry('separation',entry_index)

        self.n_match -= 1

    def summary(self,units='deg'):

        for j in range(0,self.n_match,1):

            if units=='deg':

                output = 'Catalog 1 star '+str(self.cat1_index[j])+' at ('+\
                    str(self.cat1_ra[j])+', '+str(self.cat1_dec[j])+\
                    ') matches Catalog 2 star '+str(self.cat2_index[j])+' at ('+\
                    str(self.cat2_ra[j])+', '+str(self.cat2_dec[j])+\
                    '), separation '+str(self.separation[j])+' '+units+'\n'

            else:

                output = 'Catalog 1 star '+str(self.cat1_index[j])+' at ('+\
                    str(self.cat1_x[j])+', '+str(self.cat1_y[j])+\
                    ') matches Catalog 2 star '+str(self.cat2_index[j])+' at ('+\
                    str(self.cat2_x[j])+', '+str(self.cat2_y[j])+\
                    '), separation '+str(self.separation[j])+' '+units+'\n'

        return output

    def summarize_last(self,units='deg'):

        j = self.n_match - 1

        if units == 'deg':

            output = 'Catalog 1 star '+str(self.cat1_index[j])+' at ('+\
                    str(self.cat1_ra[j])+', '+str(self.cat1_dec[j])+\
                    ') matches Catalog 2 star '+str(self.cat2_index[j])+' at ('+\
                    str(self.cat2_ra[j])+', '+str(self.cat2_dec[j])+\
                    '), separation '+str(self.separation[j])+' deg\n'

        elif units == 'both':

                output = 'Catalog 1 star '+str(self.cat1_index[j])+' at RA,Dec=('+\
                        str(self.cat1_ra[j])+', '+str(self.cat1_dec[j])+'), x,y=('+\
                        str(self.cat1_x[j])+', '+str(self.cat1_y[j])+\
                        ') matches Catalog 2 star '+str(self.cat2_index[j])+' at RA,Dec=('+\
                        str(self.cat2_ra[j])+', '+str(self.cat2_dec[j])+'), x,y=('+\
                        str(self.cat2_x[j])+', '+str(self.cat2_y[j])+\
                        '), separation '+str(self.separation[j])+' deg\n'

        else:

            output = 'Catalog 1 star '+str(self.cat1_index[j])+' at ('+\
                    str(self.cat1_x[j])+', '+str(self.cat1_y[j])+\
                    ') matches Catalog 2 star '+str(self.cat2_index[j])+' at ('+\
                    str(self.cat2_x[j])+', '+str(self.cat2_y[j])+\
                    '), separation '+str(self.separation[j])+' pixels\n'

        return output

    def find_star_match_index(self, catalog_index, cat2_star_id):
        """Method to find the array index of a star entry in the matched stars list,
        based on it's star ID number from either catalog.

        Inputs:
        :param str catalog_index: Name of catalog index attribute to search
                                    one of {cat1_index, cat2_index}
        :param int cat2_star_id: Star ID index to search for

        Outputs:
        :param int idx: Array index of star or -1 if not found
        """

        catalog_star_index = getattr(self,catalog_index)

        try:
            idx = catalog_star_index.index(cat2_star_id)
        except ValueError:
            idx = -1

        return idx

    def find_starlist_match_ids(self, catalog_index, star_ids, log,
                                verbose=False, expand_star_ids = False):
        """Method to find the array index of a star entry in the matched stars list,
        based on it's star ID number from either catalog.

        Inputs:
        :param str catalog_index: Name of catalog index attribute to search
                                    one of {cat1_index, cat2_index}
        :param list star_ids: Star ID indices to search for (from catalog_index)

        Outputs:
        :param array idx: Array index of star or -1 if not found
        """

        star_ids = np.array(star_ids)
        if verbose:
            log.info('Searching for '+str(len(star_ids))+' stars in index '+catalog_index)

        search_catalog_index = np.array( getattr(self,catalog_index) )
        if catalog_index == 'cat1_index':
            result_catalog = 'cat2_index'
        else:
            result_catalog = 'cat1_index'

        result_catalog_index = np.array( getattr(self,result_catalog) )

        # Rows in the list of star IDs where the star is present in the catalog
        present = np.isin(star_ids, search_catalog_index)

        if verbose:
            log.info('Stars present in search index: '+str(len(np.where(present)[0])))
            log.info('Length of present array='+str(len(present))+\
                ' should equal list of input star IDs='+str(len(star_ids)))

        # Indicies in the matched index of the sought-after stars
        entries = np.where(np.isin(search_catalog_index, star_ids))[0]

        if verbose:
            log.info('Identified '+str(len(entries))+\
                    ' array entries in the search index for these stars')

        # Identify any non-unique entries:
        (unique_search_ids, unique_search_index) = np.unique(search_catalog_index,return_index=True)
        non_unique_search_ids = np.delete(search_catalog_index, unique_search_index)
        non_unique_search_index = np.delete(np.arange(0,len(search_catalog_index),1), unique_search_index)

        if len(non_unique_search_ids) > 0:
            log.info('Found the following non-unique entries in the matched_stars index: '+repr(non_unique_search_ids))
            log.info('at array positions in the matched_stars index: '+repr(non_unique_search_index))
        else:
            log.info('Found no duplicates in the matched_stars index')

        non_unique_present = np.isin(star_ids, non_unique_search_ids)
        non_unique_star_id_entries = np.where(non_unique_present == True)[0]
        non_unique_star_ids = star_ids[non_unique_star_id_entries]

        # There are some circumstances where we would like to retain the
        # duplicated entries, e.g. where multiple stars are identified very
        # close together and effectively get the same photometry, so that
        # this issue can be more effectively addressed at a different stage of
        # the pipeline.
        if len(non_unique_star_id_entries) > 0 and expand_star_ids:
            # Add repeated star IDs to the end of the star_IDs list
            new_star_ids = np.array(star_ids.tolist() + non_unique_star_ids.tolist())

            # Expand the present array to match the length and entries of
            # the star IDs list
            new_present = np.array([True]*len(new_star_ids))
            new_present[0:len(present)] = present

            # Replace the original arrays with the expanded ones
            star_ids = new_star_ids
            present = new_present

        elif len(non_unique_star_id_entries) > 0 and not expand_star_ids:
            entries = np.delete(entries,non_unique_search_index)
            log.info('Removed entries '+repr(non_unique_search_index)+' from result catalog index')

        result_star_index = np.zeros(len(star_ids), dtype='int')
        result_star_index.fill(-1)

        result_star_index[present] = result_catalog_index[entries]

        return star_ids, result_star_index
