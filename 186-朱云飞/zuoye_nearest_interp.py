# -*- coding: utf-8 -*-

import cv2
import numpy as np

def function(img):
    heigth,width,channles = img.shape
    emptyImage = np.zeros((400,400,channles),img.dtype)
    sh = 400/heigth
    sw = 400/width
    for i in range(400):
        for j in range(400):
            x = int(i/sh + 0.5)   #int(),转为整型，使用向下取整。
            y = int(j/sw + 0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom=function(img)
print(img.shape)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()