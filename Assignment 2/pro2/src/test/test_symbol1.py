import argparse
import os
import math
import numpy as np
from utility import *
from scipy import fftpack
from PIL import Image
from scipy.misc import imread, imsave
from huffmantree import HuffmanTree 

def symbol2int(a,b):
    s1 = bin(a)[2:].zfill(4)
    s2 = bin(b)[2:].zfill(4)
    s = s1+s2
    return int(s,2)

print(symbol2int(0,0))


a = []
for j in range(3):
    b = []
    for i in range(3):
        b.append(i)
    a.append(b)

print(flatten(a))

