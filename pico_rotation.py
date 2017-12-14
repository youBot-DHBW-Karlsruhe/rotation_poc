import cv2
import numpy as np


def get_rotation():
    im = cv2.imread('power1.png')
    show_image(im)

    im = im[140:, :]
    show_image(im)

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    show_image(thresh)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im, contours, -1, (0, 255, 255), 3)

    cnt = contours[0]
    M = cv2.moments(cnt)
    print(M)

    show_image(im)


def show_image(image):
    cv2.imshow('Stream', image)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    get_rotation()
