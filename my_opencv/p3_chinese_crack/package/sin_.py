from math import sin, cos
import numpy as np
import cv2 as cv


img_n = np.zeros((40, 100), np.uint8)
print(img_n.shape)
temp_lst = []
for w_ in range(100):
    h_ = 15*cos(0.7*w_)+25
    temp_lst.append((int(h_), int(w_)))

for h_, w_ in temp_lst:
    try:
        img_n[h_-1, w_] = 0
        img_n[h_, w_] = 0
        img_n[h_+1, w_] = 0
    except:
        pass
cv.imshow("ss", img_n)
cv.waitKey(0)


