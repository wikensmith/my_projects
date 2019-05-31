"""
author: wiken
Date:2019/5/31
"""
import cv2 as cv
from generate_captcha import gen_captcha_text_and_image


if __name__ == '__main__':
    text, img = gen_captcha_text_and_image()
    cv.imshow("img", img)
    cv.waitKey(0)
    print(text)

