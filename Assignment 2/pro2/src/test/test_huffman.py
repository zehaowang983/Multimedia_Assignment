import argparse
import os
import math
import numpy as np
from utility import *
from scipy import fftpack
from PIL import Image
from scipy.misc import imread, imsave
from huffmantree import HuffmanTree 
 
#  H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
#     H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))

#     H_AC_Y = HuffmanTree(
#         flatten(run_length_encode(ac[i, :, 0])[0]
#                 for i in range(blocks_count)))

#     H_AC_C = HuffmanTree(
#             flatten(run_length_encode(ac[i, :, j])[0]
#                     for i in range(blocks_count) for j in [1, 2]))

# a = np.array([2,2,3,1,4,5,2,2,1,2,3,3])
a = np.array([1,1,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,6])

H = HuffmanTree(a)

print(H.value_to_bitstring_table())