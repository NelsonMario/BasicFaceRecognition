import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('asset\ImageProcessing\pink_guy.jpg', 0)

img = cv2.equalizeHist(img)

cv2.imshow('Image Example', img)
cv2.waitKey(0)

_, result_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
_, result_binary_inv = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
_, result_trunc = cv2.threshold(img, 128, 255, cv2.THRESH_TRUNC)
_, result_tozero = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO)
_, result_tozero_inv = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO_INV)
_, result_otsu = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)

result_image = [result_binary, result_binary_inv, result_trunc, result_tozero, result_tozero_inv, result_otsu]
result_title = ['Binary', 'Binary Inv', 'TRUNC', 'TOZERO', 'TOZERO Inv', 'Otsu']

for idx, (img, title) in enumerate(zip(result_image, result_title)):
    plt.subplot(3, 2, idx + 1)
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

plt.show()

mean_filter = cv2.blur(img, (11, 11))
gaussian_filter = cv2.GaussianBlur(img, (11, 11), 5.0)
median_filter = cv2.medianBlur(img, 11)
bilateral_filter = cv2.bilateralFilter(img, 11, 125, 125)

filter_result = [mean_filter, gaussian_filter, median_filter, bilateral_filter]
filter_title = ['Mean', 'Gaussian', 'Median', 'Bilateral']

for idx, (img, title) in enumerate(zip(filter_result, filter_title)):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
plt.show()