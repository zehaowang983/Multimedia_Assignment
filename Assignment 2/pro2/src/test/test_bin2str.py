import argparse
import os
import math
import numpy as np
from utility import *
from scipy import fftpack
from PIL import Image
from scipy.misc import imread, imsave
from huffmantree import HuffmanTree 

def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else binstr_flip(binstr)

def binstr_flip(binstr):
    # check if binstr is a binary string
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))

def rlc_values(Zs):
    valuess = []
    for i in range(3):
        values = []
        blocks = len(Zs[i])
        for j in range(blocks):
            arr = Zs[i][j][1:]
            value = []
            # determine where the sequence is ending prematurely
            last_nonzero = -1
            # print(arr)
            for k, elem in enumerate(arr):
                if elem != 0:
                    last_nonzero = i
            # print(last_nonzero)
            run_length = 0

            for k, elem in enumerate(arr):
                if k > last_nonzero:
                    value.append(int_to_binstr(0))
                    break
                elif elem == 0 and run_length < 15:
                    run_length += 1
                else:
                    value.append(int_to_binstr(elem))
                    run_length = 0
            
            values.append(value)

        valuess.append(values)
    return valuess

def bin_value(arr):
    value = []
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    # print(arr)
    for k, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = k
    # print(last_nonzero)
    run_length = 0

    for k, elem in enumerate(arr):
        if k > last_nonzero:
            value.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            value.append(int_to_binstr(elem))
            run_length = 0
    return value

a = np.array([10,8,-6,24,18,5,-6,5,4,4,-17])
print(bin_value(a))
print(int_to_binstr(-3))
print(int_to_binstr(0))
print(int_to_binstr(-1))


