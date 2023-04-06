import numpy as np
from matplotlib import pyplot as plt
from skimage import draw, transform

image = np.zeros((200, 200))
inx = np.arange(25, 175)
image[inx, inx] = 255
image[draw.line(45, 25, 25, 175)] = 255
image[draw.line(25, 135, 175, 155)] = 255

tested_angles = np.linspace(0, np.pi, 360, endpoint=False)
h, theta, d = transform.hough_line(image, theta=tested_angles)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Input Image")
# ax[0].set_axis_off()

angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]
ax[1].imshow(np.log(1 + h), cmap="gray", extent=bounds, aspect=1 / 1.5)
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')

ax[2].imshow(image, cmap='gray')
ax[2].set_title('Detected lines')
ax[2].set_ylim(image.shape[0], 0)

b = transform.hough_line_peaks(h, theta, d)
for _, angle, dist in zip(*b):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    print(x0, y0)
    print(angle)
    print('***************')
    ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))


plt.tight_layout()
plt.show()

a = np.unique(h)
