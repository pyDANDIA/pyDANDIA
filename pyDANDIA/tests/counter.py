# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:00:35 2017

@author: rstreet
"""

def counter():
    """Test function designed simply to run a long count for use in 
    testing spawned and parallel processes.
    """
    
    max_count = 1e9
    
    i = 0

    while i < max_count:
        i += 1


if __name__ == '__main__':

    counter()