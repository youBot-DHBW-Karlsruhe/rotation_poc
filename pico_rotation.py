import cv2
import numpy as np


def get_object_rotation_in_scene(image_object, image_scene, show=False):
    contours_scene = find_contours(prepare_image(image_scene, show), show)
    contour_object = find_contours(prepare_image(image_object, show), show)[0]

    angle_object = get_contour_angle_on_image(contour_object, image_object, False)

    for index in range(len(contours_scene)):
        contour_scene = contours_scene[index]
        if len(contour_scene) > 200:
            matching = cv2.matchShapes(contour_scene, contour_object, 1, 0.0)
            if matching < 0.2:
                print("Contour Length:", len(contour_scene))
                print("Matching:", matching)
                cv2.drawContours(image_scene, contours_scene, index, (0, 255, 255), 3)
                angle_in_scene = get_contour_angle_on_image(contour_scene, image_scene, True)
                print("Angle difference", angle_in_scene - angle_object)


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


def prepare_image(image, show=False):
    # image_gray = cv2.morphologyEx(image_gray, 1, None)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
    image_gray = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
    if show:
        show_image(image_gray)
    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show:
        show_image(thresh)
    return image_gray


def find_contours(image, show=False):
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours.sort(key=len, reverse=True)
    if show:
        image_show = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(image_show, contours, -1, (0, 255, 255), 3)
        show_image(image_show)
    return contours


def show_image(image):
    cv2.imshow('Stream', image)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    image_scene_pico = cv2.imread('power3.png')
    image_power_bank = cv2.imread('power_origin2.png')
    show_image(image_scene_pico)

    get_object_rotation_in_scene(image_power_bank, image_scene_pico, show=True)
