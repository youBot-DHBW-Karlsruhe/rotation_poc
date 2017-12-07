import numpy as np
import cv2
import webcam

MIN_MATCH_COUNT = 1

img_object = cv2.imread('duplo42_3.png', 0)
# img_scene = cv2.imread('duplo42.png', 0)
img_scene = webcam.get_image(False)

# Initiate detector
detector = cv2.ORB_create()
detector = cv2.xfeatures2d.SURF_create(1000)
# detector = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors
kp_object, des_object = detector.detectAndCompute(img_object, None)
print(len(kp_object))
kp_scene, des_scene = detector.detectAndCompute(img_scene, None)
print(len(kp_scene))

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_object, des_scene, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp_object[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img_object.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img_scene = cv2.polylines(img_scene, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img_object = cv2.drawKeypoints(img_object, kp_object, None, (255, 0, 0), 4)
img_scene = cv2.drawKeypoints(img_scene, kp_scene, None, (255, 0, 0), 4)

img3 = cv2.drawMatches(img_object, kp_object, img_scene, kp_scene, good, None, **draw_params)

while True:
    cv2.imshow('Object Detection', img3)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break
