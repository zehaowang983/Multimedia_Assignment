import numpy as np
from math import sqrt
from scipy.misc import imread, imsave
import imageio
import cv2

# read img from the disk
nobel_img = imread('诺贝尔.jpg')
lena_img = imread('lena.jpg')
print(nobel_img.dtype,nobel_img.shape)
print(lena_img.dtype,nobel_img.shape)

# convert to red channel 
nobel_img = nobel_img[:,:,0]
lena_img = lena_img[:,:,0]

row_size = nobel_img.shape[0]
col_size = nobel_img.shape[1]

# 最大半径大小
r_max = sqrt((row_size/2)**2+(col_size/2)**2)
print(r_max)

images = []
img = nobel_img
images.append(img)
imsave('Assets/img0.jpg', img)

for i in range(1,100):
    r_t = r_max*i/99
    for x in range(0,row_size):
        for y in range(0,col_size):
            r = sqrt((x-row_size/2)**2+(y-col_size/2)**2)
            if r< r_t:
                img[x,y] = lena_img[x,y]
            else:
                img[x,y] = nobel_img[x,y]
    images.append(img)
    # Write the tinted image back to disk     
    imsave('Assets/img'+str(i)+'.jpg', img)

# 设定视频每帧的图片格式
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
size = (row_size,col_size)
# 设定视频每秒帧数和每帧像素大小
vw = cv2.VideoWriter('file.avi', fourcc=fourcc, fps=20, frameSize=size)

# 设定所有帧序列
for i in range(0,100):
    f_read = cv2.imread('Assets/img'+str(i)+'.jpg')
    vw.write(f_read)
vw.release()