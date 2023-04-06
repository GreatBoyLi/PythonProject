from skimage import data, feature, transform, draw, util, color
import numpy as np
from matplotlib import pyplot as plt

coin = util.img_as_ubyte(data.coins())
coin_circle = color.gray2rgb(coin)
coin_canny = feature.canny(coin, sigma=5)
coin_canny_circle = coin_canny

hough_radii = np.arange(20, 35, 2)
hough_res = transform.hough_circle(coin_canny, hough_radii)
accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=10)

for x, y, radius in zip(cx, cy, radii):
    rr, cc = draw.circle_perimeter(y, x, radius)
    coin_circle[rr, cc] = (220, 20, 20)

fig, axes = plt.subplots(2, 2, figsize=(10, 4))
axe = axes.ravel()

axe[0].imshow(coin, cmap='gray')
axe[1].imshow(coin_canny, cmap='gray')
axe[2].imshow(coin_circle, cmap='gray')
axe[3].imshow(coin_canny_circle, cmap='gray')
plt.show()