# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

# 读取图像
plt.subplot(221)
image = plt.imread("lenna.png") 
plt.imshow(image)
print("---image lenna----")
print(image)

# 灰度化
image_gray = rgb2gray(image)  
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(image_gray, cmap='gray')
print("---image gray----")
print(image_gray)

# 二值化
img_binary = np.where(image_gray >= 0.5, 1, 0) 
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223) 
plt.imshow(img_binary, cmap='gray')
plt.show()