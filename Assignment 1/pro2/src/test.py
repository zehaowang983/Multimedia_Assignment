import numpy as np
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import cv2
from PIL import Image

img = Image.open("../redapple.jpg")

img = img.quantize()

img.show()

img.save("test.bmp")

# img = Image.open("mc_apple2.jpg")
# test = img.convert("RGB")
# s = test.getcolors(1000000)
# color = []
# for time,(r,g,b) in s:
#     color.append(r)
# color = list(set(color))
# print(len(color))

# print(dic)

# gray_img = cv2.imread('mc_apple.png', cv2.IMREAD_GRAYSCALE)
# histr = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# hist,bins = np.histogram(gray_img,256,[0,256])
# plt.hist(gray_img.ravel(),256,[0,256])
# plt.show()

# def adjust_gamma(image, gamma=1.0):
#     # build a lookup table mapping the pixel values [0, 255] to
#     # their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     print(table)
#     # apply gamma correction using the lookup table
#     return cv2.LUT(image, table)

# img = cv2.imread("redapple.jpg")

# new_img = adjust_gamma(img,2.2)

# imsave("gamma_img.jpg",new_img)

# img = imread("redapple.jpg")
# height = img.shape[0]
# width = img.shape[1]
# new_img = img
# red = (img[:,:,0])/32
# green = (img[:,:,1])/32
# blue = (img[:,:,2]*4)/64
# print(blue)

# for x in range(height):
#     for y in range(width):
#         # new_img[x][y] = (red[x][y] << 5) | (green[x][y] << 2) | blue[x][y]
#         new_img[x][y] = red[x][y] * 32 + green[x][y] * 4 + blue[x][y] 

# imsave("new_img.jpg",new_img)