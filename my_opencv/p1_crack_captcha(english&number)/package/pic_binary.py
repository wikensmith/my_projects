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
    def __init__(self):
        self.q = Queue()
        self.visited = set()
        self.binaryed_img = ""
        self.i = 0
        self.blured = ""
        self.text = ""

    def binary(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        blured = cv.medianBlur(binary, 5)
        return blured

    def delete_noisy(self, img):
        h, w = img.shape
        # 当外包12 个格式的时候，多了
        func = [(1, -1), (1, 0), (1, 1), (1, 2),
                (0, -1), (0, 2),
                (-1, -1), (-1, 2),
                (-2, -1), (-2, 0), (-2, 1), (-2, 2)]
        # func  = [(1, -1),(1, 0),(1, 1),(0, -1),(0, 1),(-1, -1),(-1, 0),(-1, 1)]
        func = [(1, -1), (1, 0), (1, 1), (1, 2), (1, 3),
                (0, -1), (0, 3),
                (-1, -1), (-1, 3),
                (-2, -1), (-2, 3),
                (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3)]

        # 当2*2 噪点时
        # for i in range(1, h - 1):
        #     for j in range(1, w - 1):
        #         sum_round = 0
        #
        #         try:
        #             for m, n in func:
        #                 if img[i + m, j + n] == 0:
        #                     sum_round += 1
        #             # print(sum_round)
        #             if sum_round >= 11:
        #                 img[i, j] = 0
        #                 img[i, j + 1] = 0
        #                 img[i - 1, j] = 0
        #                 img[i - 1, j + 1] = 0
        #                 # for m, n in func:
        #                 #     img[i + m, j + n] = 0
        #
        #         except:
        #             pass
        # 3*3 噪点时
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                sum_round = 0

                try:
                    for m, n in func:
                        if img[i + m, j + n] == 0:
                            sum_round += 1
                    # print(sum_round)
                    if sum_round >= 15:
                        img[i, j] = 0
                        img[i, j + 1] = 0
                        img[i, j + 2] = 0
                        img[i - 1, j] = 0
                        img[i - 1, j + 1] = 0
                        img[i - 1, j + 2] = 0
                        img[i - 2, j] = 0
                        img[i - 2, j + 1] = 0
                        img[i - 2, j + 2] = 0

                        # for m, n in func:
                        #     img[i + m, j + n] = 0

                except:
                    pass
        return img

    def get_begining(self, x):
        h, w = self.binaryed_img.shape
        x += 3
        for j in range(x, w):
            for i in range(h):
                try:
                    if self.binaryed_img[i, j] == 255 and self.binaryed_img[i, j + 1] == 255 and self.binaryed_img[i, j + 2] == 255:
                        if self.binaryed_img[i, j] not in self.visited:
                            self.visited.add((i, j))
                            self.q.put((i, j))
                        break
                except:
                    pass
            if self.visited:
                break

    def traverse(self):
        """遍历点
        :return:
        """
        h, w = self.binaryed_img.shape
        func = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while not self.q.empty():
            i, j = self.q.get()
            for m, n in func:
                try:
                    if self.binaryed_img[i + m, j + n] == 255 and (i + m, j + n) not in self.visited and i+m >= 0 and j + n >=0:
                        # print(i+m, j+n, "in trasver")
                        self.visited.add((i+m, j+n))
                        self.q.put((i + m, j + n))
                except Exception as e:
                    # print(e)
                    pass

    def cut(self):
        img = np.zeros((60, 80), np.uint8)
        x_max = 0
        x_min = 200
        for i, j in self.visited:
            if j > x_max:
                x_max = j
            if j < x_min:
                x_min= j
        pixel_sum = 0
        for i, j in self.visited:
            if j- x_min + 10 > 79 or i > 59:
                continue
            img[i, j- x_min + 10] = 255
            pixel_sum += 1
        print(pixel_sum, "pixel_sum")
        if pixel_sum < 100:
            return "", "", x_max + 3
        width = x_max - x_min
        begin_index = self.i
        self.i += 1
        # save_name = f"./pic/dealed/{self.text[begin_index: self.i]}_{str(int(time.time()*1000))}.png"
        if width > 60:
            return "", "", "over"
        if width > 35:
            self.i += 1
            # save_name = f"./pic/dealed2/{self.text[begin_index: self.i]}_{str(int(time.time()*1000))}.png"
        elif width < 0:
            return "", "", "over"
        # cv.imwrite(save_name, img)
        name = self.text[begin_index: self.i]
        return x_max, name, img

    def main(self):
        while True:
            self.visited = set()
            self.text, self.img = gen_captcha_text_and_image()
            image = self.binary()
            self.binaryed_img = self.delete_noisy(image)
            x = 0
            for i in range(4):
                self.visited = set()
                self.get_begining(x)
                self.traverse()
                x, name, img = self.cut()
                print(name)
                yield name, img
                if not x:
                    break
                if img == "over":
                    break
                elif img == "noisy":
                    pass
            self.i = 0
            self.visited = set()


if __name__ == '__main__':
    ac = PicBinary().main()
    for i in range(40):
        print(i, "ssssssssssssssssssssssssssssssssssssssssssss")
        name, img = ac.__next__()
        if not name:
            continue
        save_name = f"./pic/dealed/{name}_{str(int(time.time()*1000))}.png"
        cv.imwrite(save_name, img)
        # cv.imshow("img", img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

