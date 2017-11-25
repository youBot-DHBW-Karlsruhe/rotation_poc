import cv2
import urllib.request
import numpy as np

while True:
    req = urllib.request.urlopen('http://192.168.0.27:8888/out.jpg')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

    # load the input image and convert it to grayscale
    image = cv2.imdecode(arr, -1)  # 'load it as it is'
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load the detector Haar cascade, then detect cat faces
    # in the input image
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    cat_faces = detector.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=7, minSize=(15, 15))

    # loop over the cat faces and draw a rectangle surrounding each
    for (i, (x, y, w, h)) in enumerate(cat_faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow('Stream', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done
cv2.destroyAllWindows()
