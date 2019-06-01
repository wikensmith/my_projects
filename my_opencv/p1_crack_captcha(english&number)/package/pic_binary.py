"""
author: wiken
Date:2019/5/31
"""
from generate_captcha import gen_captcha_text_and_image
import numpy as np
import cv2 as cv
from queue import Queue
import time


class PicBinary:
    def __init__(self, img):
        self.img = img
        self.q = Queue()
        self.visited = set()
        self.binaryed_img = ""

    def binary(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        # gaussican = cv.GaussianBlur(gray, (3,3), 3,0)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        blured = cv.medianBlur(binary, 5)
        blured2 = cv.medianBlur(binary, 5)
        # kernel = np.ones((3,3), np.uint8)
        # opening = cv.morphologyEx(blured, cv.MORPH_OPEN, kernel)
        # cv.imshow("img", self.img)
        # cv.imshow("bin", opening)
        # cv.imshow("blured", blured)
        # cv.imshow("blured2", blured2)
        # print(blured2.shape, "shape")
        # name = f'./pic/dealed/{str(int(time.time()*1000))}_{text}.png'
        # cv.imwrite(name, blured2)
        # cv.waitKey(0)
        return blured2

    def delete_noisy(self, img):
        h, w = img.shape
        # 当外包12 个格式的时候，多了
        func = [(1, -1), (1, 0), (1, 1), (1, 2),
                (0, -1), (0, 2),
                (-1, -1), (-1, 2),
                (-2, -1), (-2, 0), (-2, 1), (-2, 2)]
        # func  = [(1, -1),(1, 0),(1, 1),(0, -1),(0, 1),(-1, -1),(-1, 0),(-1, 1)]

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                sum_round = 0

                try:
                    for m, n in func:
                        if img[i + m, j + n] == 0:
                            sum_round += 1
                    # print(sum_round)
                    if sum_round >= 11:
                        img[i, j] = 0
                        img[i, j + 1] = 0
                        img[i - 1, j] = 0
                        img[i - 1, j + 1] = 0
                        # for m, n in func:
                        #     img[i + m, j + n] = 0
                except:
                    pass
        return img

    def get_begining(self, x):
        h, w = self.binaryed_img.shape
        for j in range(x, w):
            for i in range(h):
                if self.binaryed_img[i, j] == 255 and self.binaryed_img[i, j + 1] == 255 and self.binaryed_img[i, j + 2] == 255:
                    if self.binaryed_img[i, j] not in self.visited:
                        self.visited.add((i, j))
                        self.q.put((i, j))
                    break
            if self.visited:
                break

    def traverse(self):
        """遍历点
        :return:
        """
        h, w = self.binaryed_img.shape
        print(h, w,"sss")
        func = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        print(self.q.empty())
        # print(self.q.get())
        while not self.q.empty():
            i, j = self.q.get()
            print(i, j , "in while")
            for m, n in func:
                print(i+m, j+n, "here")
                try:
                    if self.binaryed_img[i + m, j + n] == 255 and \
                            self.binaryed_img not in self.visited:
                        print(i, j, "for in ")
                        # self.visited.add((i+m, j+n))
                        # self.q.put(i + m, j + n)
                except Exception as e:
                    print(e)
                    pass



    def cut(self):
        pass

    def main(self):
        image = self.binary()
        self.binaryed_img = self.delete_noisy(image)
        # cv.imshow("ss", self.binaryed_img)
        # cv.waitKey(0)
        x = 0
        for i in range(4):
            self.get_begining(x)
            self.traverse()
        cv.imshow("img", self.visited)
        cv.waitKey(0)


# cv.imshow("sso", self.img)
# cv.imshow("img", img)
# cv.waitKey(0)


if __name__ == '__main__':
    # text, img = gen_captcha_text_and_image()
    img = cv.imread("./pic/agnu.png")
    # cv.imshow("imgo", img)
    # cv.waitKey(0)
    PicBinary(img).main()
