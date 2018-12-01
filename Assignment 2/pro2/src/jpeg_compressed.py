import argparse
import os
import math
import numpy as np
from scipy import fftpack
from PIL import Image
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import cv2
from utility import *
from huffmantree import HuffmanTree
# from scipy.misc import imread,imsave
import imageio

img1 = imageio.imread('../动物照片.jpg')
img2 = imageio.imread('../动物卡通图片.jpg')

# 块的大小
B=8 
# 图片的大小
height1,width1=(np.array(img1.shape[:2])/B * B).astype(np.int32)
img1=img1[:height1,:width1]
height2,width2=(np.array(img2.shape[:2])/B * B).astype(np.int32)
img2=img2[:height2,:width2]


# m = np.array([[ 65.481, 128.553, 24.966],
#                   [-37.797, -74.203, 112],
#                   [ 112, -93.786, -18.214]])
# shape1 = img1.shape
# shape2 = img2.shape
# # 转换成二维矩阵
# if len(shape1) == 3:
#     img1 = img1.reshape((shape1[0] * shape1[1], 3))
# if len(shape2) == 3:
#     img2 = img2.reshape((shape2[0] * shape2[1], 3))
# # 转换成YCbCr
# ycbcr1 = np.dot(img1, m.transpose() / 255.)
# ycbcr1[:,0] += 16.
# ycbcr1[:,1:] += 128.
# ycbcr1 = ycbcr1.reshape(shape1)

# ycbcr2 = np.dot(img2, m.transpose() / 255.)
# ycbcr2[:,0] += 16.
# ycbcr2[:,1:] += 128.
# ycbcr2 = ycbcr2.reshape(shape2)

# 转换成uint8
# ycbcr1 = ycbcr1.astype(np.uint8)
# ycbcr2 = ycbcr2.astype(np.uint8)


ycbcr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
ycbcr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCR_CB)
# 显示YCbCr图片
# plt.subplot(1,2,1)
# plt.imshow(ycbcr1)
# plt.subplot(1,2,2)
# plt.imshow(ycbcr2)
# plt.show()
# imageio.imsave('../ycbcr1.jpg',ycbcr1)
# imageio.imsave('../ycbcr2.jpg',ycbcr2)


# 每两个像素垂直采样一次
vertical_subsample=2
# 每两个像素水平采样一次
horizontal_subsample=2

# 2*2平滑滤波（取区域内的平均值）
crf1=cv2.boxFilter(ycbcr1[:,:,1],ddepth=-1,ksize=(2,2))
cbf1=cv2.boxFilter(ycbcr1[:,:,2],ddepth=-1,ksize=(2,2))
crf2=cv2.boxFilter(ycbcr2[:,:,1],ddepth=-1,ksize=(2,2))
cbf2=cv2.boxFilter(ycbcr2[:,:,2],ddepth=-1,ksize=(2,2))
# 每隔一行和一列采样，即2*2区域内采样一次
crsub1=crf1[::vertical_subsample,::horizontal_subsample]
cbsub1=cbf1[::vertical_subsample,::horizontal_subsample]
crsub2=crf2[::vertical_subsample,::horizontal_subsample]
cbsub2=cbf2[::vertical_subsample,::horizontal_subsample]
# 将三个通道下采样后的像素值存入列表
sub_img1=[ycbcr1[:,:,0],crsub1,cbsub1]
sub_img2=[ycbcr2[:,:,0],crsub2,cbsub2]

# 输出大小检验
print(ycbcr1[:,:,0].size)
print(crsub1.size)
print(cbsub1.size)

print(ycbcr2[:,:,0].size)
print(crsub2.size)
print(cbsub2.size)
# print(imSub)

# 亮度和色度量化表
QY=np.array([[16,11,10,16,24,40,51,61],
             [12,12,14,19,26,48,60,55],
             [14,13,16,24,40,57,69,56],
             [14,17,22,29,51,87,80,62],
             [18,22,37,56,68,109,103,77],
             [24,35,55,64,81,104,113,92],
             [49,64,78,87,103,121,120,101],
             [72,92,95,98,112,100,103,99]])

QC=np.array([[17,18,24,47,99,99,99,99],
             [18,21,26,66,99,99,99,99],
             [24,26,56,99,99,99,99,99],
             [47,66,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99]])

# 所有dct变换后的块
dct_blocks1=[]
dct_blocks2=[]
# 所有量化后的块
quan_blocks1=[]
quan_blocks2=[]
# YCrCb颜色通道
ch=['Y','Cr','Cb']
# 量化表
Q=[QY,QC,QC]
# 所有z扫描后的向量
Zs1 = []
Zs2 = []

# 遍历每个颜色通道
for index,channel in enumerate(sub_img1):
        # 行数
        rows=channel.shape[0]
        # 列数
        cols=channel.shape[1]
        # dct变换后的块
        dct_block = np.zeros((rows,cols), np.int32)
        # 量化后的块
        quan_block = np.zeros((rows,cols), np.int32)
        # 块的行数
        block_rows=int(rows/B)
        # 块的列数
        block_cols=int(cols/B)

        # z扫描后的向量
        # z = np.array((block_rows,block_cols), np.zeros(1,64))
        z = []

        block = np.zeros((rows,cols), np.float32)
        block[:rows, :cols] = channel
        # 整齐化，减128使Y分量（DC）成为均值为0。
        block=block-128
        # for debug
#         print(block_rows)
#         print(block_cols)
        # 处理每个块
        for row in range(block_rows):
                for col in range(block_cols):
                        # 当前块
                        # currentblock = cv2.dct(block[row*B:(row+1)*B,col*B:(col+1)*B])
                        currentblock = fftpack.dct(fftpack.dct(block[row*B:(row+1)*B,col*B:(col+1)*B].T, norm='ortho').T, norm='ortho').round().astype(np.int32)
                        # 存储二维dct变换后的块
                        dct_block[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                        # 存储量化后的块
                        quan_block[row*B:(row+1)*B,col*B:(col+1)*B]=np.round(currentblock/Q[index])
                        # z扫描
                        # z[row,col] = block_to_zigzag(quan_block[row*B:(row+1)*B,col*B:(col+1)*B])
                        z.append(block_to_zigzag(quan_block[row*B:(row+1)*B,col*B:(col+1)*B]))


        dct_blocks1.append(dct_block)
        quan_blocks1.append(quan_block)
        Zs1.append(z)

# print(len(dct_blocks1))
# print(len(quan_blocks1))
print("origin block 1:\n",sub_img1[0][0:8,0:8])
print("DCT block 1:\n",dct_blocks1[0][0:8,0:8])
print("QUAN block 1:\n",quan_blocks1[0][0:8,0:8])
print("Z array:\n",Zs1[0][0])

dcs1 = dcpm(Zs1)
dcs1_values = dc(Zs1)
acs1_symbol1 = rlc(Zs1)
acs1 = ac(Zs1)
acs1_bin = rlc_values(Zs1)
# print(len(dcs1[0]))
print("DC size block 1:",dcs1[0][0])
print("DC value block 1:",dcs1_values[0][0])
# print(len(acs1_symbol1[0][0]))
print("AC values block 1:",acs1[0][0])
print("AC sizes block 1:",acs1_symbol1[0][0])
print("AC bin_str block 1:",acs1_bin[0][0])
print(len(acs1_symbol1[0][0]))
print(len(acs1_bin[0][0]))
# DC_Y_Huffman = HuffmanTree(dcs1[0])
# DC_C_Huffman = HuffmanTree(dcs1[1]+dcs1[2])
# print(DC_Y_Huffman.value_to_bitstring_table())
# print(DC_C_Huffman.value_to_bitstring_table())

# Huffman_tables = {'dc_y': DC_Y_Huffman.value_to_bitstring_table(),
#         'ac_y': AC_Y_Huffman.value_to_bitstring_table(),
#         'dc_c': DC_C_Huffman.value_to_bitstring_table(),
#         'ac_c': AC_C_Huffman.value_to_bitstring_table()}

# AC_Y_Huffman = HuffmanTree(flatten(acs1_symbol1[0]))
# AC_C_Huffman = HuffmanTree(flatten(acs1_symbol1[1])+flatten(acs1_symbol1[2]))
# print(AC_Y_Huffman.value_to_bitstring_table())
# print(AC_C_Huffman.value_to_bitstring_table())
# print(len(dcs1[0]))
# print(len(dcs1[1]))
# print(len(dcs1[2]))
# write_to_file('encode1.txt', dcs1, acs1_symbol1, len(dcs1[0]), Huffman_tables)

# H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
H_DC_Y = HuffmanTree(dcs1[0])
# H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
H_DC_C = HuffmanTree(dcs1[1]+dcs1[2])

# H_AC_Y = HuffmanTree(
#     flatten(run_length_encode(ac[i, :, 0])[0]
#             for i in range(blocks_count)))
H_AC_Y = HuffmanTree(flatten(acs1_symbol1[0]))

# H_AC_C = HuffmanTree(
#         flatten(run_length_encode(ac[i, :, j])[0]
#                 for i in range(blocks_count) for j in [1, 2]))
H_AC_C = HuffmanTree(flatten(acs1_symbol1[1])+flatten(acs1_symbol1[2]))

tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
            'ac_y': H_AC_Y.value_to_bitstring_table(),
            'dc_c': H_DC_C.value_to_bitstring_table(),
            'ac_c': H_AC_C.value_to_bitstring_table()}

# print(H_AC_Y.value_to_bitstring_table())
count1 = write_to_file('encode1.b', dcs1, dcs1_values,acs1_symbol1, acs1_bin,tables)
print("动物图片写入位数：\n",count1)

for index,channel in enumerate(sub_img2):
        # 行数
        rows=channel.shape[0]
        # 列数
        cols=channel.shape[1]
        # dct变换后的块
        dct_block = np.zeros((rows,cols), np.int32)
        # 量化后的块
        quan_block = np.zeros((rows,cols), np.int32)
        # 块的行数
        block_rows=int(rows/B)
        # 块的列数
        block_cols=int(cols/B)

        # z扫描后的向量
        # z = np.array((block_rows,block_cols), np.zeros(1,64))
        z = []

        block = np.zeros((rows,cols), np.float32)
        block[:rows, :cols] = channel
        # 整齐化，减128使Y分量（DC）成为均值为0。
        block=block-128
        
        pass
        # for debug
        # print(block_rows)
        # print(block_cols)
        # print(block[0:8,0:8])
        # print(cv2.dct(block[0:8,0:8]))

        # 处理每个块        
        for row in range(block_rows):
                for col in range(block_cols):
                        # 当前块
                        # currentblock = cv2.dct(block[row*B:(row+1)*B,col*B:(col+1)*B])
                        currentblock = fftpack.dct(fftpack.dct(block[row*B:(row+1)*B,col*B:(col+1)*B].T, norm='ortho').T, norm='ortho').round().astype(np.int32)
                        # 存储二维dct变换后的块
                        dct_block[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                        # 存储量化后的块
                        quan_block[row*B:(row+1)*B,col*B:(col+1)*B]=np.round(currentblock/Q[index])

                        # z扫描
                        # z[row,col] = block_to_zigzag(quan_block[row*B:(row+1)*B,col*B:(col+1)*B])
                        z.append(block_to_zigzag(quan_block[row*B:(row+1)*B,col*B:(col+1)*B]))

        dct_blocks2.append(dct_block)
        quan_blocks2.append(quan_block)
        Zs2.append(z)
       
# print(len(dct_blocks1[0]))
# print(len(dct_blocks1[1]))
# print(len(dct_blocks1[2]))
print("origin block 1:\n",sub_img2[0][0:8,0:8])
print("DCT block 1:\n",dct_blocks2[0][0:8,0:8])
print("QUAN block 1:\n",quan_blocks2[0][0:8,0:8])
print("Z array:\n",Zs2[0][0])

# print(len(quan_blocks1))
# print(dct_blocks)
# print(quan_blocks)

dcs2 = dcpm(Zs2)
dcs2_values = dc(Zs2)
acs2_symbol1 = rlc(Zs2)
acs2 = ac(Zs2)
acs2_bin = rlc_values(Zs2)
# print(len(dcs1[0]))
print("DC size block 1:",dcs2[0][0])
print("DC value block 1:",dcs2_values[0][0])
# print(len(acs1_symbol1[0][0]))
print("AC values block 1:",acs2[0][0])
print("AC sizes block 1:",acs2_symbol1[0][0])
print("AC bin_str block 1:",acs2_bin[0][0])
# print(len(acs1_symbol1[0][0]))
# print(len(acs1_bin[0][0]))

H_DC_Y = HuffmanTree(dcs2[0])
H_DC_C = HuffmanTree(dcs2[1]+dcs2[2])
H_AC_Y = HuffmanTree(flatten(acs2_symbol1[0]))
H_AC_C = HuffmanTree(flatten(acs2_symbol1[1])+flatten(acs2_symbol1[2]))

tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
            'ac_y': H_AC_Y.value_to_bitstring_table(),
            'dc_c': H_DC_C.value_to_bitstring_table(),
            'ac_c': H_AC_C.value_to_bitstring_table()}

count2 = write_to_file('encode2.b', dcs2, dcs2_values,acs2_symbol1, acs2_bin,tables)
print("卡通图片写入位数：\n",count2)


# for debug
# 根据书上的例子对DCT和量化进行测试
# block = np.array([
#                 [200, 202, 189, 188, 189, 175, 175, 175],
#                 [200, 203, 198, 188, 189, 182, 178, 175],
#                 [203, 200, 200, 195, 200, 187, 185, 175],
#                 [200, 200, 200, 200, 197, 187, 187, 187],
#                 [200, 205, 200, 200, 195, 188, 187, 175],
#                 [200, 200, 200, 200, 200, 190, 187, 175],
#                 [205, 200, 199, 200, 191, 187, 187, 175],
#                 [210, 200, 200, 200, 188, 185, 187, 186]
#             ])

# block = block - 128
# # DCT
# dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho').round().astype(np.int32)
# # 量化
# dct_q = (dct_block / QY).round().astype(np.int32)
# # 反量化
# i_q = dct_q * QY
# # 逆DCT
# i_block = fftpack.idct(fftpack.idct(i_q.T, norm='ortho').T, norm='ortho').round().astype(np.int32)
# i_block = i_block + 128
# # 误差
# err = block-i_block+128

# print("原始块:\n",block+128)
# print("DCT变换系数:\n",dct_block)
# print("量化后:\n",dct_q)
# print("逆量化后:\n",i_q)
# print("重构后的:\n",i_block)
# print("误差:\n",err)

reCons1=np.zeros((height1,width1,3), np.uint8)
reCons2=np.zeros((height2,width2,3), np.uint8)
print(reCons1.shape)
print(reCons2.shape)

for index,channel in enumerate(quan_blocks1):
        rows=channel.shape[0]
        cols=channel.shape[1]
        block_rows=int(rows/B)
        block_cols=int(cols/B)

        block = np.zeros((rows,cols), np.uint8)
#         print(block.shape)
        for row in range(block_rows):
                for col in range(block_cols):
                        # 逆量化
                        dequantblock=channel[row*B:(row+1)*B,col*B:(col+1)*B]*Q[index]
                        # 逆DCT                        
                        # currentblock = np.round(cv2.idct(dequantblock))+128
                        currentblock = fftpack.idct(fftpack.idct(dequantblock.T, norm='ortho').T, norm='ortho').round().astype(np.int32)+128
                        # 设定阈值                        
                        currentblock[currentblock>255]=255
                        currentblock[currentblock<0]=0
            
                        block[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
        #         
        print(block.shape)
        back1=cv2.resize(block,(width1,height1))
        reCons1[:,:,index]=np.round(back1)
    
print(reCons1.shape)
print(reCons1[0:8,0:8,0])
# print(reCons1)

for index,channel in enumerate(quan_blocks2):
        rows=channel.shape[0]
        cols=channel.shape[1]
        block_rows=int(rows/B)
        block_cols=int(cols/B)

        block = np.zeros((rows,cols), np.uint8)
#         print(block.shape)
        for row in range(block_rows):
                for col in range(block_cols):
                        # 逆量化
                        dequantblock=channel[row*B:(row+1)*B,col*B:(col+1)*B]*Q[index]
                        # 逆DCT                        
                        # currentblock = np.round(cv2.idct(dequantblock))+128
                        currentblock = fftpack.idct(fftpack.idct(dequantblock.T, norm='ortho').T, norm='ortho').round().astype(np.int32)+128
                        # 设定阈值                        
                        currentblock[currentblock>255]=255
                        currentblock[currentblock<0]=0
            
                        block[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
        #         
        print(block.shape)
        back1=cv2.resize(block,(width2,height2))
        reCons2[:,:,index]=np.round(back1)
    
print(reCons2.shape)
# print(reCons2)


rgb1 = cv2.cvtColor(reCons1, cv2.COLOR_YCR_CB2RGB)
rgb2 = cv2.cvtColor(reCons2, cv2.COLOR_YCR_CB2RGB)

plt.subplot(2,1,1)
plt.imshow(rgb1)
plt.subplot(2,1,2)
plt.imshow(rgb2)
plt.show()

# 显示结果
# res_img1 = Image.fromarray(reCons1,'YCbCr')
# res_img1 = res_img1.convert('RGB')
# res_img1.save('jpeg_com1.jpg')

# res_img2 = Image.fromarray(reCons2,'YCbCr')
# res_img2 = res_img2.convert('RGB')
# res_img2.save('jpeg_com2.jpg')

# compressed_img1 = imageio.imread('jpeg_com1.jpg')
# compressed_img2 = imageio.imread('jpeg_com2.jpg')
# plt.subplot(1,2,1)
# plt.imshow(compressed_img1)
# plt.subplot(1,2,2)
# plt.imshow(compressed_img2)
# plt.show()