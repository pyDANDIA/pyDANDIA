import numpy as np
import itertools


def construct_a_master_mask(original_master_mask, list_of_masks):
    master_mask = original_master_mask
    for index, mask in enumerate(list_of_masks[::-1]):
        master_mask += mask * 2 ** index

    return master_mask



def deconstruct_a_master_mask(master_mask, number_of_mask=4):
    masks = [np.zeros(master_mask.shape)] * number_of_mask

    list_of_possibles = all_possible_integers(number_of_mask)

    for possible in list_of_possibles:

        integer_value = int(possible, number_of_mask)

        index_mask = master_mask == integer_value

        if True in index_mask:
            for index, mask in enumerate(masks):

                if possible[index] == '1':
                    mask[index_mask] = 1

    return masks


def all_possible_integers(length):
    combinations = list(itertools.product(n, repeat=length))

    list_of_possibles = [''.join(str(e) for e in i) for i in combinations]

    return list_of_possibles


