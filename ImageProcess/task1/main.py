import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from skimage import io
from load_image import loadImage
from unload_image import unLoadImage


carrier = io.imread('./image/carrier/carrier1.pgm', as_gray=True)
encrypt1 = io.imread('./image/encrypt/encrypt1.pgm', as_gray=True)
encrypt2 = io.imread('./image/encrypt/encrypt2.pgm', as_gray=True)
encrypt3 = io.imread('./image/encrypt/encrypt3.pgm', as_gray=True)

loadImg = loadImage(carrier, encrypt1, encrypt2, encrypt3)
encryptedImg = loadImg.load_image()

fig, axes = plt.subplots(2, 2)
axe = axes.ravel()

#axes[0].imshow(carrier, cmap='gray')
# axes[0].imshow(testImg, cmap='gray')


unloadImg = unLoadImage(encryptedImg)
protoImg, unencryptImg1, unencryptImg12, unencryptImg3 = unloadImg.unload_image()

axe[0].imshow(protoImg, cmap='gray')
axe[1].imshow(unencryptImg1, cmap='gray')
axe[2].imshow(unencryptImg12, cmap='gray')
axe[3].imshow(unencryptImg3, cmap='gray')

plt.show()
