"""
author: wiken
Date:2019/6/3
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

name = os.listdir("./pic")
print(name)
plt.subplot()
for i in name:
    if "." in i:
        img = cv.imread("./pic/1559623977257.png")
        h, w, c = img.shape
        c1 = img[:,:,0]
        c2 = img[:, :, 1]
        c3 = img[:, :, 2]
        # print(c1)
        plt.subplot()
        # print(img.shape[:2])
        plt.hist(c1.ravel(), 256)
        plt.hist(c2.ravel(), 256)
        plt.hist(c3.ravel(), 256)
        plt.title(f"test{i}", fontsize=8)
        plt.xticks([]), plt.yticks([])
        plt.show()



def get_num(img):
    h, w = img.shape
    dic = {}
    for w_ in range(w):
        for h_ in range(h):
            x = img[h_, w_]
            if str(x) not in dic:
                dic[str(img[h_, w_])] = 1
            else:
                dic[str(x)] += 1



#
#
# plt.hist(c1.ravel(), 256)
# plt.hist(c2.ravel(), 256)
# plt.hist(c3.ravel(), 256)
# plt.title("test", fontsize=8)
# plt.xticks([]), plt.yticks([])
# plt.show()

# images = [img, 0, th1, img, 0, th2, blur, 0, th3]
# titles = ['Original', 'Histogram', 'Global(v=100)',
#           'Original', 'Histogram', "Otsu's",
#           'Gaussian filtered Image', 'Histogram', "Otsu's"]
# img = sum_img.ravel()
# for i in range(1):
#     # 绘制原图
#
#     # 绘制直方图plt.hist，ravel函数将数组降成一维
#     plt.subplot(3, 3, i * 3 + 2)
#     plt.hist(img, 256)
#     plt.title(titles[i * 3 + 1], fontsize=8)
#     plt.xticks([]), plt.yticks([])
# plt.show()