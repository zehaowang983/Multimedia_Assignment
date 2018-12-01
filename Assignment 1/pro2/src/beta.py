import math
import numpy as np
from scipy.misc import imread, imsave
from scipy.spatial.distance import cdist,pdist,squareform
from matplotlib import pyplot as plt
import cv2

# C64 color palette
# COLOR_BLACK 		= (0, 0, 0)
# COLOR_WHITE 		= (255, 255, 255)
# COLOR_RED 			= (104, 55, 43)
# COLOR_CYAN 			= (112, 164, 178)
# COLOR_PURPLE 		= (111, 61, 134)
# COLOR_GREEN 		= (88, 141, 67)
# COLOR_BLUE 			= (53, 40, 121)
# COLOR_YELLOW 		= (184, 199, 111)
# COLOR_ORANGE 		= (111, 79, 37)
# COLOR_BROWN 		= (67, 57, 0)
# COLOR_LIGHTRED 		= (154, 103, 89)
# COLOR_DARKGREY 		= (68, 68, 68)
# COLOR_GREY 			= (108, 108, 108)
# COLOR_LIGHTGREEN 	= (154, 210, 132)
# COLOR_LIGHTBLUE 	= (108, 94, 181)
# COLOR_LIGHTGREY 	= (149, 149, 149)

# palette = [ COLOR_BLACK, COLOR_WHITE, COLOR_RED, COLOR_CYAN, COLOR_PURPLE, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_ORANGE, COLOR_BROWN, COLOR_LIGHTRED, COLOR_DARKGREY, COLOR_GREY, COLOR_LIGHTGREEN, COLOR_LIGHTBLUE, COLOR_LIGHTGREY ]

apple_im = imread("redapple.jpg")

gray_img = cv2.imread('redapple.jpg', cv2.IMREAD_GRAYSCALE)
color = ('b','g','r')

histr = cv2.calcHist([gray_img],[0,1,2],None,[256,256,256],[0,256,0,256,0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.title('Histogram for color scale picture')
# plt.show()

# hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# hist,bins = np.histogram(gray_img,256,[0,256])
# plt.hist(gray_img.ravel(),256,[0,256])
# plt.show()

width = apple_im.shape[1]
height = apple_im.shape[0]
# 410x726
print(height,width)

red_channel = apple_im[:,:,0].reshape(height*width,1)
green_channel = apple_im[:,:,1].reshape(height*width,1)
blue_channel = apple_im[:,:2].reshape(2460,1)
# print(np.median(red_channel))
red_means = []
median_value = np.median(red_channel)	
red_means.append(np.mean([i for i in red_channel if i< median_value]))
red_means.append(np.mean([i for i in red_channel if i>= median_value]))
print(red_means)

green_means = []
median_value = np.median(green_channel)	
green_means.append(np.mean([i for i in green_channel if i< median_value]))
green_means.append(np.mean([i for i in green_channel if i>= median_value]))
print(green_means)

blue_means = []
median_value = np.median(blue_channel)	
blue_means.append(np.mean([i for i in blue_channel if i< median_value]))
blue_means.append(np.mean([i for i in blue_channel if i>= median_value]))
print(blue_means)

palette = []
for x in red_means:
	for y in green_means:
		for z in blue_means:
			palette.append((x,y,z))

dest1 = apple_im
count = 0
for x in range(height):
	for y in range(width):
		distances = []
		for c in palette:
			distances.append(np.sqrt(np.sum((apple_im[x,y]-c)**2)))
		dest1[x, y] = palette[np.argmin(distances)]
		count += 1
		if(count%1000==0):
			print(count)



imsave("paletteC64_first.png",dest1)
# def group_by(group):
# 	median_value = np.median(group)
# 	pre_group = group[group<median_value]
# 	post_group = group[group>=median_value]
# 	return median_value,pre_group,post_group

def pre_group_by(group,medians):
	median_value = np.median(group)	
	if(len(medians) >=4):
		return 
	medians.append(median_value)
	pre_group = group[group<median_value]
	pre_group_by(pre_group,medians)

def post_group_by(group,medians):
	median_value = np.median(group)	
	if(len(medians) >=8):
		return 
	medians.append(median_value)
	post_group = group[group>=median_value]
	post_group_by(post_group,medians)

def pre_group_by2(group,medians):
	median_value = np.median(group)	
	if(len(medians) >=2):
		return 
	medians.append(median_value)
	pre_group = group[group<median_value]
	pre_group_by2(pre_group,medians)

def post_group_by2(group,medians):
	median_value = np.median(group)	
	if(len(medians) >=4):
		return 
	medians.append(median_value)
	post_group = group[group>=median_value]
	post_group_by2(post_group,medians)


# red_medians = []
# pre_group_by(red_channel,red_medians)
# post_group_by(red_channel,red_medians)
# red_medians.sort()
# del red_medians[4]
# red_medians.append(255)
# red_medians.insert(0,0)
# red_means = []
# for i in range(8):
# 	bottom = red_medians[i]
# 	top = red_medians[i+1]
# 	red_means.append(np.mean([i for i in red_channel if i> bottom and i <= top]))
# print(red_medians)
# print(red_means)

# green_medians = []
# pre_group_by(green_channel,green_medians)
# post_group_by(green_channel,green_medians)
# green_medians.sort()
# del green_medians[4]
# green_medians.append(255)
# green_medians.insert(0,0)
# green_means = []
# for i in range(8):
# 	bottom = green_medians[i]
# 	top = green_medians[i+1]
# 	green_means.append(np.mean([i for i in green_channel if i> bottom and i <= top]))
# print(green_medians)
# print(green_means)
# print(green_medians)

# blue_medians = []
# pre_group_by2(blue_channel,blue_medians)
# post_group_by2(blue_channel,blue_medians)
# blue_medians.sort()
# del blue_medians[2]
# blue_medians.append(255)
# blue_medians.insert(0,0)
# blue_means = []
# for i in range(4):
# 	bottom = blue_medians[i]
# 	top = blue_medians[i+1]
# 	blue_means.append(np.mean([i for i in blue_channel if i> bottom and i <= top]))
# print(blue_medians)
# print(blue_means)
# print(blue_medians)
