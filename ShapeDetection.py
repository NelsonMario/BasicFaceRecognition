import cv2
import numpy as np
import matplotlib.pyplot as plt

car = cv2.imread('asset/ShapeDetection/car.jpg')
scene = cv2.imread('asset\ShapeDetection/scene.png')

surf = cv2.xfeatures2d.SURF_create()
kp_car, descriptor_car = surf.detectAndCompute(car, None)
kp_scene, descriptor_scene = surf.detectAndCompute(scene, None)

descriptor_car = descriptor_car.astype('f')
descriptor_scene = descriptor_scene.astype('f')

FLANN_INDEX_TREE = 0
CHECKS = 500
algo = dict(algorithm = FLANN_INDEX_TREE)
search = dict(checks = CHECKS)
flann = cv2.FlannBasedMatcher(algo, search)

matches = flann.knnMatch(descriptor_car, descriptor_scene, 2)

match_mask = []
for i in range(0, len(matches)):
    match_mask.append([0, 0])

total_match = 0
for idx, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        match_mask[idx] = [1, 0]
        total_match += 1

img_res = cv2.drawMatchesKnn(car, kp_car, scene, kp_scene, matches, None, matchColor=[0, 0, 255], singlePointColor=[255, 0, 0], matchesMask=match_mask)

plt.imshow(img_res)
plt.show()