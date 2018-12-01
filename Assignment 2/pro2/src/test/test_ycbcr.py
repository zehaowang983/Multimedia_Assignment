import argparse
import os
import math
import numpy as np
from scipy import fftpack
from PIL import Image
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import cv2
# from scipy.misc import imread,imsave
import imageio

def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    # rgb = copy.deepcopy(ycbcr)
    rgb = ycbcr
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)


img1 = imageio.imread('../动物照片.jpg')
img2 = imageio.imread('../动物卡通图片.jpg')


ycbcr1 = rgb2ycbcr(img1)
ycbcr2 = rgb2ycbcr(img2)
rgb1 = ycbcr2rgb(ycbcr1)
rgb2 = rgb2ycbcr(ycbcr2)

ycbcr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
ycbcr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCR_CB)
rgb1 = cv2.cvtColor(ycbcr1, cv2.COLOR_YCR_CB2RGB)
rgb2 = cv2.cvtColor(ycbcr2, cv2.COLOR_YCR_CB2RGB)

plt.subplot(2,2,1)
plt.imshow(ycbcr1)
plt.subplot(2,2,2)
plt.imshow(ycbcr2)
plt.subplot(2,2,3)
plt.imshow(rgb1)
plt.subplot(2,2,4)
plt.imshow(rgb2)
plt.show()