import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('asset\ImageProcessing\pink_guy.jpg')


canny_img = cv2.Canny(img, 125, 225)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, 5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, 5)
laplace_img = cv2.Laplacian(img, cv2.CV_64F)
laplace_uint8 = np.uint8(np.absolute(laplace_img))

edge_result = [canny_img, sobel_x, sobel_y, laplace_img, laplace_uint8]
edge_title = ['Canny', 'Sobel X', 'Sobel Y', 'Laplace', 'Laplace uint8']

for idx, (img, title) in enumerate(zip(edge_result, edge_title)):
    plt.subplot(2, 3, idx + 1)
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
plt.show()