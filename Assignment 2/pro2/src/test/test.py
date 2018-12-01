import argparse
import os
import math
import numpy as np
from utility import *
from scipy import fftpack
from PIL import Image
from huffmantree import HuffmanTree

 
# BLACK = [0,0,0]
 
# img = cv2.imread('../动物卡通图片.jpg')

# height = img.shape[0]
# width = img.shape[1]

# m = math.ceil(height / 8) * 8 - height
# print(m)
# n = width % 8
# img = cv2.copyMakeBorder(img,m,0,n,0,cv2.BORDER_CONSTANT,value=BLACK)
# print(img.shape)

# cv2.imwrite('../动物卡通图片2.jpg',img)

# img = Image.open('../动物卡通图片.jpg')
# ycbcr = img.convert('YCbCr')
# print(img.size)

# npmat = np.array(ycbcr, dtype=np.uint8)

# rows, cols = npmat.shape[0], npmat.shape[1]

block = np.array([
                [200, 202, 189, 188, 189, 175, 175, 175],
                [200, 203, 198, 188, 189, 182, 178, 175],
                [203, 200, 200, 195, 200, 187, 185, 175],
                [200, 200, 200, 200, 197, 187, 187, 187],
                [200, 205, 200, 200, 195, 188, 187, 175],
                [200, 200, 200, 200, 200, 190, 187, 175],
                [205, 200, 199, 200, 191, 187, 187, 175],
                [210, 200, 200, 200, 188, 185, 187, 186]
            ])

block = block - 128

dct_b = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho').round().astype(np.int32)

q = load_quantization_table('lum')

dct_q = (dct_b / q).round().astype(np.int32)

i_q = dct_q * q

i_block = fftpack.idct(fftpack.idct(i_q.T, norm='ortho').T, norm='ortho').round().astype(np.int32)

i_block = i_block + 128


# print(block)
# print(dct_b)
# print(q)
# print(dct_q)
# print(i_q)
# print(i_block)

# print(n)

def reverse_str(str):
    res = ""
    for _,s in enumerate(str):
        if s=='0':
            res += '1'
        else:
            res += '0'
    return res

def int_size(number):
    str = ""
    if number < 0:
        number = abs(number)
        l = len(bin(number)) - 2
        str = bin(number)[-l:]
        str = reverse_str(str)
    else:
        l = len(bin(number)) - 2
        str = bin(number)[-l:]
    # return (l,str)
    return l 

# print(int2tuple(150))
# print(int2tuple(5))
# print(int2tuple(-6))
# print(int2tuple(3))
# print(int2tuple(-8))


def dcpm_size(dc_block):
    res = np.empty((dc_block.shape[0], 3), dtype=np.int32)
    for i in range(3):
        dc = dc_block[:,i]
        for j in range(dc_block.shape[0]):  
            if j == 0:
                res[j,i] = int_size(dc_block[j,i])
            else:
                res[j,i] = int_size(dc_block[j,i] - dc_block[j-1,i])
    return res

dc = np.array([[150,150,150],
                [155,155,155],
                [149,149,149],
                [152,152,152],
                [144,144,144]])

dc_ = dcpm_size(dc)
print(dc_)

def run_length_encode(arr):
    res = []
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    # values are binary representations of array elements using SIZE bits
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            res.append(int_size(size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values, res


a,b,c= run_length_encode([32,6,-1,-1,0,0,0,0,-1])

print(a)
print(c)
