import cv2
import numpy as np


def get_rotation(image_scene, image_object):

    show_image(image_scene)

    contours_scene = find_contours(prepare_image(image_scene))
    contour_object = find_contours(prepare_image(image_object))[0]

    for index in range(len(contours_scene)):
        contour_scene = contours_scene[index]
        if len(contour_scene) > 0:
            matching = cv2.matchShapes(contour_scene, contour_object, 1, 0.0)
            if matching < 0.1:
                print(matching)
                cv2.drawContours(image_scene, contours_scene, index, (0, 255, 255), 3)
                show_image(image_scene)

    # cnt = contours[0]
    # M = cv2.moments(cnt)
    # print(M)


def prepare_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    show_image(thresh)
    return thresh

def find_contours(image):
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours.sort(key=len, reverse=True)
    return contours

def show_image(image):
    cv2.imshow('Stream', image)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    image_scene = cv2.imread('power1.png')[140:, :]
    image_object = cv2.imread('power_origin1.png')

    get_rotation(image_scene, image_object)
