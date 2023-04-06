from PIL import Image
from PIL import ImageFilter
import numpy as np
image = Image.open("/Users/lee/Desktop/liwenpeng.jpeg")
# image = image.convert("LA")
# image.show()

print(image.getpixel((100, 100)))

for x in range(0, image.size[0]):
    for y in range(0, 100):
        position = (x, y)
        pixel = (1000, 256, 226)
        image.putpixel(position, pixel)

# image.show()
x = [1, 2, 3]
y = [4, 5, 6]
print(y - x)


import