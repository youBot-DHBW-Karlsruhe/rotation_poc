import cv2
import numpy as np


def get_object_rotation_in_scene(image_object, image_scene):
    contours_scene = find_contours(prepare_image(image_scene))
    contour_object = find_contours(prepare_image(image_object))[0]

    angle_object = get_contour_angle_on_image(contour_object, image_object, False)

    for index in range(len(contours_scene)):
        contour_scene = contours_scene[index]
        if len(contour_scene) > 0:
            matching = cv2.matchShapes(contour_scene, contour_object, 1, 0.0)
            if matching < 0.3:
                print("Matching:", matching)
                cv2.drawContours(image_scene, contours_scene, index, (0, 255, 255), 3)
                angle_in_scene = get_contour_angle_on_image(contour_scene, image_scene, True)
                print("Angle difference", angle_in_scene - angle_object)
                show_image(image_scene)


def get_contour_angle_on_image(contour, image, show_line=False):
    x1, y1, x2, y2 = get_contour_line_on_image(contour, image)
    angle_rad = np.arctan((y2 - y1) / (x2 - x1))
    angle_deg = angle_rad * 180 / np.pi + 90

    if show_line:
        draw_line(image, x1, y1, x2, y2)

    return angle_deg


def draw_line(image, x1, y1, x2, y2):
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    show_image(image)


def get_contour_line_on_image(contour, image):
    rows, cols = image.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    # First point
    x1 = 0
    y1 = int((-x * vy / vx) + y)
    # Second point
    x2 = cols - 1
    y2 = int(((cols - x) * vy / vx) + y)

    return x1, y1, x2, y2


def prepare_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    image_scene = cv2.imread('power1.png')
    image_object = cv2.imread('power_origin2.png')

    get_object_rotation_in_scene(image_object, image_scene)
