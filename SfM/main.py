import cv2

img = cv2.imread("C:/Users/Great_Boy_Li/Desktop/0a5f387b9c33a8b13c6f1457b3eda7d.jpg")

resize_img = cv2.resize(img, (0, 0), fx=2, fy=2)

cv2.imwrite("C:/Users/Great_Boy_Li/Desktop/test.jpg", resize_img)

