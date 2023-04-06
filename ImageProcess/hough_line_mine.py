from skimage import io, feature, transform
from matplotlib import pyplot as plt
import numpy as np

img = io.imread('/Users/lee/Desktop/liwenpeng.jpeg', as_gray=True)

fig, axes = plt.subplots(2, 2)
axe = axes.ravel()

img_canny = feature.canny(img, 3)

angles = np.linspace(0, np.pi, 360)
hspace, theta, distances = transform.hough_line(img_canny, angles)
angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(distances).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          distances[-1] + d_step, distances[0] - d_step]
axe[2].imshow(np.log(hspace + 1), cmap="gray", extent=bounds,aspect=1 / 3)
for _, angle, dist in zip(*transform.hough_line_peaks(hspace, theta, distances, num_peaks=2)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    axe[3].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

axe[0].imshow(img, cmap="gray")
axe[1].imshow(img_canny, cmap="gray")
axe[3].imshow(img_canny, cmap="gray")


plt.show()