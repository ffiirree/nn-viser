import cv2
import torch

image  = cv2.imread('static/images/arctic_fox.jpeg')
cuda_image = cv2.cuda_GpuMat()
cuda_image.upload(image)
bilateral = cv2.bilateralFilter(cuda_image, 5, 25, 25)

cv2.imshow('original', image)
# cv2.imshow('bilateral', bilateral)
cv2.waitKey(0)
