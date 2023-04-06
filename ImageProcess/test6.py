import numpy as np
from skimage import io, feature, util, filters, color, morphology, data
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from PIL import Image

img2 = io.imread('/Users/lee/Desktop/liwenpeng.jpeg')

# img = ndi.rotate(img, 15)
# img = util.random_noise(img, mode="speckle", mean=0.1)

img = color.rgb2gray(img2)

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x][y] > 20:
            pass

a = data.horse()
b = a != True
print(a[b])

img1 = util.invert(img)
# img3 = morphology.convex_hull_image(img1)


edg = feature.canny(img, sigma=2)

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].imshow(edg, cmap="gray")
# ax[1].imshow(img3, cmap=plt.cm.gray)

# ax.imshow(img)
plt.tight_layout()
plt.show()
