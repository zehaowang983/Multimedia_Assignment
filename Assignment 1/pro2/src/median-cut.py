from PIL import Image
from cube import ColorSpace
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.misc import imread, imsave


# 画RGB值的二维直方图
image = cv2.imread("redapple.jpg")
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
# loop over the image channels
for (chan, color) in zip(chans, colors):
	# create a histogram for the current channel and
	# concatenate the resulting histograms for each
	# channel
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	features.extend(hist)
 
	# plot the histogram
	plt.plot(hist, color = color)
	plt.xlim([0, 256])
    
# plt.show() 

def Colors(image):
    # 获得每个像素点的rgb三元组
    colors = image.getcolors(image.size[0] * image.size[1])
    # print(colors[:10])
    return [c[1] for c in colors]

def median_cut(image, num_colors):
    colors = Colors(image)
    print(colors[:10])
    # 建立RGB立方体，每个像素点的RGB值为一个三维坐标
    cubes = [ColorSpace(*colors)]
    count = 1
    while len(cubes) < num_colors:
        # 所有子颜色空间最大的边长
        global_max_size = 0

        for index, cube in enumerate(cubes):
            # 每个通道的极差
            size = cube.size
            # 此颜色空间RGB的最大极差
            max_size = max(size)

            if max_size > global_max_size:
                global_max_size = max_size
                max_cube = index
        # 第1，4，7层按照R划分
        if (len(cubes)>=1 and len(cubes)<=2) or (len(cubes)>=8 and len(cubes)<=16) or (len(cubes)>=64 and len(cubes)<=128):
            max_color = 0
        # 第2，5，8层按照G划分
        elif (len(cubes)>=2 and len(cubes)<=4) or (len(cubes)>=16 and len(cubes)<=32) or (len(cubes)>=128 and len(cubes)<=256):
            max_color = 1
        # 第3，6层按照B划分
        else:
            max_color = 2
        split_cube = cubes[max_cube]
        cube_low, cube_high = split_cube.split(max_color)
        cubes = cubes[:max_cube] + [cube_low, cube_high] + cubes[max_cube + 1:]

    return [c.average for c in cubes]

apple_im = Image.open("../redapple.jpg")
print(apple_im.size)
palette = median_cut(apple_im,256)
print(len(palette))

# img = Image.new("RGB",(100,256))
# for i in range(0,100):
#     for j in range(0,256):
#         img.putpixel((i,j),palette[j])
        
# img.save("../palette.jpg")

apple_im = np.array(apple_im)
height = apple_im.shape[0]
width = apple_im.shape[1]

# print(apple_im.shape)
# dest1 = np.empty((height,width,3))
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

imsave("quantize_redapple.bmp",dest1)