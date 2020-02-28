# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:49:31 2019

@author: rstreet
"""

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
