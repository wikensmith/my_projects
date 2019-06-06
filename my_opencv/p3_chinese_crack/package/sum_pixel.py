import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


name = './pic/1559623976958.png'
img = cv.imread(name)
# cv.imshow("img",img[:,:,2])
# cv.waitKey(0)

img2 = img.copy()
gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
h, w = gray.shape
print(gray.shape)
c = gray[1, 1]
print(c)
for h_ in range(h):
    for w_ in range(w):
        if c - 10 < gray[h_, w_] < c + 10:
            gray[h_, w_] = 255
        if gray[h_, w_] > 190:
            gray[h_, w_] = 255
# blured = cv.medianBlur(gray, 3)
# cv.imwrite("./pic/test.png", gray)
# sum_img = np.zeros(256, np.uint8)
sum_img = [0 for i in range(256)]
# print(sum_img)
for h_ in range(h):
    for w_ in range(w):
        sum_img[gray[h_, w_]] += 1

print("37, 35, 30",sum_img.index(37), sum_img.index(35), sum_img.index(30), sum_img.index(29), sum_img.index(28))

print("#######",sum_img)
sum_img.sort(reverse=True)
print(sum_img)
sum_img = sum_img[1:]
sum_all = 0
for i in sum_img:
    sum_all += i
print(sum_all)
sum_temp = 0
for i in sum_img:
    sum_temp += i
    if sum_temp/sum_all > 0.8:
        print("here is "+str(i))
        break


for h_ in range(h):
    for w_ in range(w):
        if gray[h_, w_] < 35 or gray[h_, w_] > 50:
            gray[h_, w_] = 255


cv.imshow("sdfsdf", gray)
cv.waitKey(0)
# plt.hist(sum_img, 256)
# plt.show()


# cv.imshow("img",gray)
# cv.waitKey(0)
