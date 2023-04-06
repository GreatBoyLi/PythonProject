import numpy as np
from skimage import io
from matplotlib import pyplot as plt

img = io.imread('/Users/lee/Desktop/liwenpeng.jpeg', as_gray=True)

a = 130
b = 140
x, y = img.shape
x, y = int(x / 2), int(y / 2)

fftImg = np.fft.fft2(img)
fftImgForShow = 20 * np.log(np.abs(fftImg))
ifftImg = np.fft.ifft2(fftImg)
ifftImgForShow = np.abs(ifftImg)

fftImg[x - a: x + b, :] = 0
fftImg[:, y - a: y + b] = 0
fftImgForShow1 = 20 * np.log(np.abs(fftImg))
ifftImg1 = np.fft.ifft2(fftImg)
ifftImgForShow1 = np.abs(ifftImg1)



fig, axes = plt.subplots(2, 2)
axe = axes.ravel()

axe[0].imshow(fftImgForShow)
axe[1].imshow(fftImgForShow1)

axe[2].imshow(ifftImgForShow, cmap='gray')
# ifftImgForShow1[ifftImgForShow1 > 1] = 1
axe[3].imshow(ifftImgForShow1, cmap='gray')

print(ifftImgForShow.max())
print(ifftImgForShow1.max())
plt.show()










# fftimg = np.fft.fft2(img)
# a = 20*np.log(np.abs(fftimg))
# fshiftimg = np.fft.fftshift(fftimg)
# # fshiftimg = fftimg
# b = 20*np.log(np.abs(fshiftimg))
# 
# rows, cols = img.shape
# crow, ccol = int(rows / 2), int(cols / 2)
# fshiftimg[crow - 30: crow + 30, ccol - 30: ccol + 30] = 0
# ishiftimg = np.fft.ifftshift(fshiftimg)
# 
# ifftimg1 = np.fft.ifft2(ishiftimg)
# ifftimg = np.abs(ifftimg1)
# 
# fig, axes = plt.subplots(2, 2)
# 
# axe = axes.ravel()
# 
# axe[0].imshow(img, cmap='gray')
# axe[1].imshow(ifftimg, cmap='gray')
# axe[2].imshow(a)
# axe[3].imshow(b)
# 
# plt.show()