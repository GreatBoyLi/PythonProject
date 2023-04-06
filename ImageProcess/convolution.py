from PIL import Image
from PIL import ImageFilter
import math


# 将图像向四周扩展，extend表示扩展的像素个数
def extendImage(oriImage: Image, extend: int = 1) -> Image:
    oriX, oriY = oriImage.size
    newX = oriX + 2 * extend
    newY = oriY + 2 * extend
    newImage = Image.new(oriImage.mode, (newX, newY))
    left = oriImage.crop((0, 0, extend, oriY))
    right = oriImage.crop((oriX - extend, 0, oriX, oriY))

    newImage.paste(left, (0, extend))
    newImage.paste(right, (oriX + extend, extend))
    newImage.paste(oriImage, (extend, extend))

    upper = newImage.crop((0, extend, newX, extend + extend))
    newImage.paste(upper, (0, 0))
    lower = newImage.crop((0, newY - 2 * extend, newX, newY - extend))
    newImage.paste(lower, (0, newY - extend))

    return newImage


# 平滑处理
def smoothImage(oriImage: Image, size: int = 3) -> Image:
    oriX, oriY = oriImage.size
    newImage = Image.new(oriImage.mode, (oriX, oriY))
    kernel = []
    for i in range(0, size):
        kernel.append([])
        for j in range(0, size):
            kernel[i].append(1 / (size * size))

    marginal = int((size - 1) / 2)
    for x in range(marginal, oriX - marginal):
        for y in range(marginal, oriY - marginal):
            sum = []
            if isinstance(oriImage.getpixel((0, 0)), int):
                length = 1
            else:
                length = len(oriImage.getpixel((0, 0)))
            for num in range(0, length):
                sum.append(0)
            for k in range(-marginal, marginal + 1):
                for l in range(-marginal, marginal + 1):
                    for pos in range(0, length):
                        a = kernel[marginal + k][marginal + l]
                        if isinstance(oriImage.getpixel((x - k, y - l)), int):
                            b = oriImage.getpixel((x - k, y - l))
                        else:
                            b = (oriImage.getpixel((x - k, y - l)))[pos]
                        sum[pos] += int(a * b)
            for num in range(0, len(sum)):
                sum[num] = sum[num]
            # print(sum)
            newImage.putpixel((x, y), tuple(sum))
    return newImage.crop((marginal, marginal, oriX - marginal, oriY - marginal))


# 扩展和平滑一起
def extendAndSmoothImage(oriImage: Image, size: int) -> Image:
    extendIma = extendImage(oriImage, int((size - 1) / 2))
    return smoothImage(extendIma, size)


# 获得轮廓线
def getContour(oriImage: Image, smoothImage: Image) -> Image:
    origX, origY = oriImage.size
    newImage = Image.new(oriImage.mode, (origX, origY))
    for x in range(0, origX):
        for y in range(0, origY):
            origColor = oriImage.getpixel((x, y))
            smoothColor = smoothImage.getpixel((x, y))
            if isinstance(oriImage.getpixel((x, y)), int):
                newImage.putpixel((x, y), (origColor - smoothColor))
            else:
                sum = []
                for i in range(0, len(oriImage.getpixel((x, y)))):
                    a = oriImage.getpixel((x, y))[i]
                    b = smoothImage.getpixel((x, y))[i]
                    sum.append(a - b)
                newImage.putpixel((x, y), tuple(sum))

    return newImage


def getX_Deri(origImage: Image, size: int) -> Image:
    extendImage1 = extendImage(origImage, int((size - 1) / 2))
    kernel = (-1, 0, 1)
    oriX, oriY = extendImage1.size

    newImage = Image.new(origImage.mode, (oriX, oriY))

    marginal = int((size - 1) / 2)
    for x in range(marginal, oriX - marginal):
        for y in range(marginal, oriY - marginal):
            sum = []
            if isinstance(extendImage1.getpixel((0, 0)), int):
                length = 1
            else:
                length = len(extendImage1.getpixel((0, 0)))
            for num in range(0, length):
                sum.append(0)
            for k in range(-marginal, marginal + 1):
                for pos in range(0, length):
                    a = kernel[marginal + k]
                    if isinstance(extendImage1.getpixel((x - k, y)), int):
                        b = extendImage1.getpixel((x - k, y))
                    else:
                        b = (extendImage1.getpixel((x - k, y)))[pos]
                    sum[pos] += int(a * b)
            # for num in range(0, len(sum)):
            #     sum[num] = sum[num]
            # print(sum)
            newImage.putpixel((x, y), tuple(sum))
    return newImage.crop((marginal, marginal, oriX - marginal, oriY - marginal))


def getY_Deri(origImage: Image, size: int) -> Image:
    extendImage1 = extendImage(origImage, int((size - 1) / 2))
    kernel = (-1, 0, 1)
    oriX, oriY = extendImage1.size

    newImage = Image.new(origImage.mode, (oriX, oriY))

    marginal = int((size - 1) / 2)
    for x in range(marginal, oriX - marginal):
        for y in range(marginal, oriY - marginal):
            sum = []
            if isinstance(extendImage1.getpixel((0, 0)), int):
                length = 1
            else:
                length = len(extendImage1.getpixel((0, 0)))
            for num in range(0, length):
                sum.append(0)
            for k in range(-marginal, marginal + 1):
                for pos in range(0, length):
                    a = kernel[marginal + k]
                    if isinstance(extendImage1.getpixel((x, y - k)), int):
                        b = extendImage1.getpixel((x, y - k))
                    else:
                        b = (extendImage1.getpixel((x, y - k)))[pos]
                    sum[pos] += int(a * b)
            # for num in range(0, len(sum)):
            #     sum[num] = sum[num]
            # print(sum)
            newImage.putpixel((x, y), tuple(sum))
    return newImage.crop((marginal, marginal, oriX - marginal, oriY - marginal))


def getGradient(origImage: Image, size: int) -> Image:
    extendImage1 = extendImage(origImage, int((size - 1) / 2))
    kernelX = (-1, 0, 1)
    kernelY = (-1, 0, 1)
    oriX, oriY = extendImage1.size

    newImage = Image.new(origImage.mode, (oriX, oriY))

    marginal = int((size - 1) / 2)
    for x in range(marginal, oriX - marginal):
        for y in range(marginal, oriY - marginal):
            sumX = []
            sumY = []
            sumGrad = []
            if isinstance(extendImage1.getpixel((0, 0)), int):
                length = 1
            else:
                length = len(extendImage1.getpixel((0, 0)))
            for num in range(0, length):
                sumX.append(0)
                sumY.append(0)
                sumGrad.append(0)
            for k in range(-marginal, marginal + 1):
                for pos in range(0, length):
                    a = kernelY[marginal + k]
                    if isinstance(extendImage1.getpixel((x, y - k)), int):
                        b = extendImage1.getpixel((x, y - k))
                    else:
                        b = (extendImage1.getpixel((x, y - k)))[pos]
                    sumY[pos] += int(a * b)

                    c = kernelX[marginal + k]
                    if isinstance(extendImage1.getpixel((x, y - k)), int):
                        d = extendImage1.getpixel((x - k, y))
                    else:
                        d = (extendImage1.getpixel((x - k, y)))[pos]
                    sumX[pos] += int(c * d)

            for i in range(0, length):
                sumGrad[i] = int(math.sqrt(sumX[i] ** 2 + sumY[i] ** 2))
            # for num in range(0, len(sum)):
            #     sum[num] = sum[num]
            # print(sum)
            newImage.putpixel((x, y), tuple(sumGrad))
    return newImage.crop((marginal, marginal, oriX - marginal, oriY - marginal))


if __name__ == '__main__':
    oriImage = Image.open("C:/Users/Great_Boy_Li/Desktop/portray.jpg")
    grayImage = oriImage.convert("L")
    grayImage.save("C:/Users/Great_Boy_Li/Desktop/portray1.jpg")

    # smoothImage = extendAndSmoothImage(grayImage, 3)
    # print("平滑成功！")
    # smoothImage.show()

    # xDeriImage = getX_Deri(grayImage, 3)
    # xDeriImage.show()
    #
    # yDeriImage = getY_Deri(grayImage, 3)
    # yDeriImage.show()

    gradImage = getGradient(grayImage, 3)
    gradImage.show()

    # contourImage = getContour(grayImage, smoothImage)
    # print("获得轮廓线成功！")
    # contourImage.show()

    # contourImage = getContour(oriImage, guaImage)
    # contourImage.show()

    gradImage.save("C:/Users/Great_Boy_Li/Desktop/outline.jpg")
