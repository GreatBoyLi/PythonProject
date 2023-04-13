import cv2

img = cv2.imread("C:/Users/Great_Boy_Li/Desktop/123.jpg")

resize_img = cv2.resize(img, (0, 0), fx=4, fy=4)

cv2.imwrite("C:/Users/Great_Boy_Li/Desktop/1234.jpg", resize_img)


# test from windows
