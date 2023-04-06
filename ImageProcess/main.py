from PIL import Image
import sys

def roll(im : Image, xDelta, yDelta) -> Image:
    xsize, ysize = im.size

    xDelta = xDelta % xsize
    if xDelta == 0:
        return im
    part1 = im.crop((0, 0, xDelta, ysize))
    part2 = im.crop((xDelta, 0, xsize, ysize))
    im.paste(part1, (xsize - xDelta, 0, xsize, ysize))
    im.paste(part2, (0, 0, xsize - xDelta, ysize))

    yDelta = yDelta % ysize
    if yDelta == 0:
        return im
    part3 = im.crop((0, 0, xsize, yDelta))
    part4 = im.crop((0, yDelta, xsize, ysize))
    im.paste(part3, (0, ysize - yDelta, xsize, ysize))
    im.paste(part4, (0, 0, xsize, ysize - yDelta))
    return im


def merge(im1, im2):
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im3 = Image.new("RGBA", (w, h))
    im3.paste(im1, (0, 0, im1.size[0], im1.size[1]))
    im3.paste(im2, (im1.size[0], 0, im1.size[0] + im2.size[0], im2.size[1]))
    return im3


im = Image.open("/Users/lee/Desktop/liwenpeng.jpeg")

print(im.format, im.size, im.mode)

im1 = im.crop((100,100,200,300))

im1 = im1.transpose(Image.Transpose.ROTATE_180)

im.paste(im1, (100,100,200,300))

im = roll(im, 100, 250)
# im1.show()
im.save("/Users/lee/Desktop/liwenpeng1.jpeg")

im3 = Image.open("/Users/lee/Desktop/liwenpeng.jpeg")
im2 = merge(im, im3)
im2.show()
#
# r, g, b = im.split()
# print(im.getpixel((5,4)))
# print(g)
# im = Image.merge(("RGB"), (b, g, r))
# im.show()

