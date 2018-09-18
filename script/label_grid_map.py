#!/usr/bin/env python  

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/kevin/data/ncfm/map/label.jpg')

print img.dtype

(height, width, _) = img.shape
label = np.zeros((height, width))

text_file = open("map.txt", "w")

for i in range(height):
    for j in range(width):
        if img[i, j][1] > 50:
            label[i, j] = 2  # truck
        elif img[i, j][2] > 50:
            label[i, j] = 1  # people

        text_file.write("%i " % label[i, j])

print label
implot = plt.imshow(label)

text_file.close()

plt.show()

cv2.imwrite("label_mask.jpeg", label)

