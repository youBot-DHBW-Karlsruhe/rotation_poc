import cv2
import webcam

while True:
    image = webcam.get_image()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load the detector Haar cascade, then detect cat faces
    # in the input image
    detector = cv2.CascadeClassifier("haarcascade_duplo3.xml")
    cat_faces = detector.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=0, minSize=(10, 20))

    # loop over the cat faces and draw a rectangle surrounding each
    for (i, (x, y, w, h)) in enumerate(cat_faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow('Stream', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done
cv2.destroyAllWindows()