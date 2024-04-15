#!/usr/bin/python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
sobel边缘检测
'''
# 读取灰度图图像
# img = cv2.imread("lenna.png", 0)
img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE) 
# cv2.imshow('img',img)

# 计算x、y方向上的梯度（即水平边缘检测）
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

# 转换为绝对值并转换为8位无符号整数
sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)

# 组合Sobel X和Y方向的边缘检测结果   
combined_sobel = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

# 显示结果 
cv2.imshow('Original Image', img)  
cv2.imshow('Sobel X', sobelx_abs)
cv2.imshow('Sobel Y', sobely_abs)
cv2.imshow('result', combined_sobel)

# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
直方图均衡化
'''

# 应用直方图均衡化
equalized_img = cv2.equalizeHist(img)

# 直方图
hist = cv2.calcHist([equalized_img],[0],None,[256],[0,256])

plt.figure()
plt.hist(equalized_img.ravel(), 256)
plt.show()

# 显示原始图像和均衡化后的图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(equalized_img, cmap='gray')
plt.title('Histogram Equalized'), plt.xticks([]), plt.yticks([])

plt.show()

'''
双线性插值
'''

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()