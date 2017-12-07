import cv2
import webcam

# img = cv2.imread('duplo42.png', 0)
img = webcam.get_image()

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
# detector = cv2.xfeatures2d.SURF_create(3000)
detector = cv2.xfeatures2d.SIFT_create()

# Find keypoints and descriptors directly
kp, des = detector.detectAndCompute(img, None)

img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
print(len(kp))

while True:
    cv2.imshow('Object Detection', img2)

    if cv2.waitKey(1) & 0xFF == 27:
        break
