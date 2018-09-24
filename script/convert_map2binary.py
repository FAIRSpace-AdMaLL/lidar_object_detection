#!/usr/bin/env python  

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import os.path

label_png = '/home/kevin/data/ncfm/map/ncfm_annotation1/label.png'
output_dir = './'

lbl = np.asarray(PIL.Image.open(label_png))

print lbl.dtype

(height, width) = lbl.shape
label = np.zeros((height, width))

text_file = open(os.path.join(output_dir, "map.txt"), "w")

for i in range(height):
    for j in range(width):
	if lbl[i, j] == 0 or lbl[i, j] == 1 or lbl[i, j] == 3 or lbl[i, j] == 4 or lbl[i, j] == 7:
		label[i, j] = 0  # background objects
	elif lbl[i, j] == 5:
		label[i, j] = 1 # pedestrains
	elif lbl[i, j] == 2 or lbl[i, j] == 6:
		label[i, j] = 2  # trucks
        text_file.write("%i " % label[i, j])

print label
implot = plt.imshow(label)

text_file.close()

plt.show()

cv2.imwrite(os.path.join(output_dir, "label_mask.jpeg"), label)

