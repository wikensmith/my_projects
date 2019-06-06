"""
author: wiken
Date:2019/6/3
"""

import cv2 as cv
import time
from matplotlib import pyplot as plt
import numpy as np
import os
name_lst = os.listdir("./pic")
for i in name_lst:
    if "." in i:
        # print(i)
        n = f'./pic/{i}'
        img = cv.imread(n)
        img2 = img.copy()
        h, w, c = img.shape
        # img = cv.GaussianBlur(img,(3,3),0)
        c1 = img[1, 1, 0]
        c2 = img[1, 1, 1]
        c3 = img[1, 1, 2]


        for c_ in range(c):
            if c_ == 0:
                cx = c1
            elif c_ == 1:
                cx = c2
            elif c_ == 2:
                cx = c3
            # print(cx, "ss")
            for h_ in range(h):
                for w_ in range(w):

                    if cx-20 < img2[h_, w_, c_] < cx+20:
                        img2[h_, w_, c_] = 0
                    if img2[h_, w_, c_] > 190:
                        img2[h_, w_, c_] = 0
        # gaus = cv.GaussianBlur(img2,(3,3),0)
        blured = cv.medianBlur(img2, 5)
        gaus = cv.GaussianBlur(blured,(3,3), 3, 0)
        cv.imshow("sss", img2)
        gray = cv.cvtColor(gaus, cv.COLOR_BGR2GRAY)
        r, b = cv.threshold(gray, 0, 255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        name = f'./pic/dealed/{str(int(time.time()*1000))}.png'
        cv.imwrite(name, b)
        print("over")
        # cv.imshow("ss", b)
        # cv.waitKey(0)








