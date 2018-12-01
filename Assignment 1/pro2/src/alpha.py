from PIL import Image
import numpy as np
import cv2
from scipy.misc import imread, imsave


img = imread("redapple.jpg")

red_regions = [img[:,:,0]]
green_regions = [img[:,:,1]]
blue_regions = [img[:,:,2]]
need_red_num = 8
need_green_num = 8
need_blue_num = 4

# test = np.array([[1,3,3],[3,2,1],[4,4,4]])
# print(test[test[:,2].argsort()])
# print(round(2.5))
# print(test[2:])
regions = [img]

while len(regions) < 256:
	new_regions = []
	for region in regions:
		red_gap = np.max(region[0])-np.min(region[0])
		green_gap = np.max(region[1])-np.min(region[1])
		blue_gap = np.max(region[2])-np.min(region[2])
		max_gap = max(red_gap,green_gap,blue_gap)
		new_order = []
		if max_gap == red_gap:
			new_order = region[region[:,:,0].argsort()]
		elif max_gap == green_gap:
			new_order = region[region[:,:,1].argsort()]
		else:
			new_order = region[region[:,:,2].argsort()]
		middle = round((new_order.shape[0]*new_order.shape[1])/2)
		print(region.shape)
		# print(new_order)
		new_regions.append(new_order[:middle,:])
		new_regions.append(new_order[middle:,:])
		# if len(new_order)%2 == 1:
		# 	new_regions.append(new_order[:middle,:])
		# 	new_regions.append(new_order[middle-1:,:])
		# else:
		# 	new_regions.append(new_order[:middle,:])
		# 	new_regions.append(new_order[middle:,:])
	regions = new_regions
	# print(len(regions))

# red_medians = []
# # red_regions = [[1,1,1,1,2,3,4,5,6,7,8]]
# while len(red_regions) < need_red_num:
# 	new_regions = []
# 	for region in red_regions:
# 		# new_order = region[region[:,0].argsort()]
# 		new_order = np.sort(region)
# 		median = np.median(region)
# 		red_medians.append(median)
# 		print(red_medians)
# 		middle = round(len(new_order)/2)
# 		# middle = region.index(median)
# 		print(middle)
# 		if len(new_order)%2 == 1:
# 			new_regions.append(new_order[:middle])
# 			new_regions.append(new_order[middle-1:])
# 		else:
# 			new_regions.append(new_order[:middle])
# 			new_regions.append(new_order[middle:])
# 		print(region)
# 		# new_regions.append([i for i in region if i <= median])
# 		# new_regions.append([i for i in region if i >= median])
# 		# new_region1 = []
# 		# new_region2 = []
# 		# for i in region:
# 		# 	if i <= median:
# 		# 		new_region1.append(i)
# 		# 	else:
# 		# 		new_region2.append(i)
# 		# new_regions.append(new_region1)
# 		# new_regions.append(new_region2)
# 	red_regions = new_regions

# print(red_regions)
# red_medians = np.sort(red_medians)
# print(red_medians[1])

# red_color = []
# for region in red_regions:
# 	red_color.append(int(np.mean(region)))

# red_color = np.sort(red_color)

# while len(green_regions) < need_green_num:
# 	new_regions = []
# 	for region in green_regions:
# 		# new_order = region[region[:,0].argsort()]
# 		new_order = np.sort(region)
# 		middle = round(len(new_order)/2)
# 		new_regions.append(new_order[:middle])
# 		new_regions.append(new_order[middle:])
# 	green_regions = new_regions

# green_color = []

# for region in green_regions:
# 	green_color.append(int(np.mean(region)))

# green_color = np.sort(green_color)


# while len(blue_regions) < need_blue_num:
# 	new_regions = []
# 	for region in blue_regions:
# 		# new_order = region[region[:,0].argsort()]
# 		new_order = np.sort(region)
# 		middle = round(len(new_order)/2)
# 		new_regions.append(new_order[:middle])
# 		new_regions.append(new_order[middle:])
# 	blue_regions = new_regions

# blue_color = []

# for region in blue_regions:
# 	blue_color.append(int(np.mean(region)))

# blue_color = np.sort(blue_color)

# palette = []
# print(red_color)
# print(green_color)
# print(blue_color)
# for i,x in enumerate(red_color):
# 	for j,y in enumerate(green_color):
# 		for k,z in enumerate(blue_color):
# 			palette.append((x,y,z))
			
# # print(palette)

# # width = img.shape[1]
# # height = img.shape[0]
# # dest1 = img
# # count = 0
# # for x in range(height):
# # 	for y in range(width):
# # 		distances = []
# # 		for c in palette:
# # 			distances.append(np.sqrt(np.sum((img[x,y]-c)**2)))
# # 		dest1[x, y] = palette[np.argmin(distances)]
# # 		count += 1
# # 		if(count%1000==0):
# # 			print(count)

# # imsave("mc1.jpg",dest1)
