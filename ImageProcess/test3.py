from skimage import io, data, color
import numpy as np

img = io.imread('/Users/lee/Desktop/liwenpeng.jpeg', as_gray=True)
img1 = data.astronaut()

rows, cols = img.shape

lable1 = np.zeros((rows, cols))

for x in range(rows):
    for y in range(cols):
        if img[x, y] < 0.4:
            lable1[x, y] = 0
        elif img[x, y] < 0.75:
            lable1[x, y] = 1
        else:
            lable1[x, y] = 2

end = color.label2rgb(lable1)

io.imshow(end)
io.show()