import cv2
import urllib.request
import numpy as np


def get_image(colored):
    req = urllib.request.urlopen('http://192.168.0.27:8888/out.jpg')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

    # load the input image and convert it to grayscale
    image = cv2.imdecode(arr, -1)  # 'load it as it is'
    if colored:
        return image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image_gray


if __name__ == "__main__":
    while True:
        cv2.imshow('Stream', get_image(True))

        if cv2.waitKey(1) & 0xFF == 27:
            break
