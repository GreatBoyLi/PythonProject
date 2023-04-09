import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)
float_tolerance = 1e-7


def computeKeyPointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """
    Compute SIFT key points and descriptors for an input iamge
    :param image:
    :param sigma:
    :param num_intervals: 一个金字塔要输出几个维度
    :param assumed_blur:
    :param image_border_width:
    :return:
    """
    # 转化为float类型
    image = image.astype('float32')
    #
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)


#########################
# Image pyramid related #
#########################

def generateBaseImage(image, sigma, assumed_blur):
    """Generate base image from input image by up sampling by 2 in both directions and blurring
    """
    logger.debug('Generating base image...')
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    # the image blur is now sigma instead of assumed_blur
    result = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
    return result


def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape(OpenCV default)
    """
    result = int(round(np.log(min(image_shape)) / np.log(2) - 1))
    return result


def generateGaussianKernels(sigma, num_intervals):
    """
    Generate list of gaussian kernels at which to blur the input image. Default valeus of sigma, intervals,
    and octaves follow section 3 of Lowe's paper.
    :param sigma:
    :param num_intervals:
    :return:
    """
    logger.debug("Generate scales...")
    num_image_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels = np.zeros(num_image_per_octave)
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_image_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)

    return gaussian_kernels


def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """
    Generate scale-space pyramid of Gaussian images
    :param image:
    :param num_octaves:
    :param gaussian_kernels:
    :return:
    """
    logger.debug('Generate Gaussian images...')
    gaussian_images = []

    for octave_index in range(num_octaves):
        # first image in octave already has the correct blur
        gaussian_images_in_octave = [image]
        for gaussian_kernel in gaussian_kernels:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                           interpolation=cv2.INTER_NEAREST)

    return np.array(gaussian_images, object)


def generateDoGImages(gaussian_images):
    """
    Generate Difference-of-Gaussians image pyramid
    :param gaussian_images:
    :return:
    """
    logger.debug('Generate Difference-of-Gaussian images...')
    dog_images = []
    for gaussian_images_in_octave in gaussian_images:
        dog_image_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            # ordinary subtraction will not work because the images are unsigned integers
            dog_image_in_octave.append(cv2.subtract(second_image, first_image))
        dog_images.append(dog_image_in_octave)

    return np.array(dog_images, object)


def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width,
                          contrast_threshold=0.04):
    """
    Find pixel positions of all scale-space extrema in the image pyramid
    :param gaussian_images:
    :param dog_images:
    :param num_intervals: 一个金字塔要输出几个维度
    :param sigma:
    :param image_border_width:
    :param contrast_threshold:
    :return:
    """
    logger.debug('Finding scale-space extrema...')
    # from OpenCV implementation
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(
                zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            # first, second, third 是第一层，第二层，第三层
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i - 1: i + 2][j - 1: j + 2],
                                         second_image[i - 1: i + 2][j - 1: j + 2],
                                         third_image[i - 1: i + 2][j - 1: j + 2], threshold):
                        # 确定下了极值点 second[i, j]
                        # 写下极值点亚像素的位置，泰勒
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1,
                                                                              octave_index, num_intervals,
                                                                              dog_images_in_octave, sigma,
                                                                              contrast_threshold, image_border_width)


def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """
    Return true if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbor, False otherwise
    :param first_subimage:
    :param second_subimage:
    :param third_subimage:
    :param threshold:
    :return:
    """
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_subimage) and \
                np.all(center_pixel_value >= second_subimage) and \
                np.all(center_pixel_value >= third_subimage)
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_subimage) and \
                np.all(center_pixel_value <= second_subimage) and \
                np.all(center_pixel_value <= third_subimage)
    return False


def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma,
                                    contrast_threshold, image_border_width, eigenvalue_ratio=10,
                                    num_attempts_until_convergence=5):
    """
    Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extreme's neighbors
    :param i:
    :param j:
    :param image_index: 每层金字塔要输出维度的那张dog
    :param octave_index: 第几层金字塔
    :param num_intervals: 每层金字塔要输出几个维度
    :param dog_images_in_octave: 某层金字塔dog的数组
    :param sigma:
    :param contrast_threshold:
    :param image_border_width:
    :param eigenvalue_ratio:
    :param num_attempts_until_convergence:
    :return:
    """
    logger.debug('Localizing scale-space extrema...')
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from unit8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's threshold
        # 计算极大值的DoG图片
        first_image, second_image, third_image = dog_images_in_octave[image_index - 1: image_index + 2]
        pixel_cube = np.stack([first_image[i - 1:i + 2, j - 1:j + 2], second_image[i - 1:i + 2, j - 1:j + 2],
                              third_image[i - 1:j + 2, j - 1:j + 2]]).astype('float32') / 255
        # 求关键点梯度
        gradient = computeGradientAtCenterPixel(pixel_cube)
        # Hessian矩阵
        hessian = computeHessianAtCenterPixel(pixel_cube)

    return 1


# 特别不清楚这块是怎么算的，需要再研究
def computeGradientAtCenterPixel(pixel_array):
    """
    Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    :param pixel_array:
    :return:
    """
    # 求梯度
    # With step size h, the central difference formula of order O(h^2) for f`(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f`(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to the first array axis, and s(scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])


def computeHessianAtCenterPixel(pixel_array):
    """
    Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference of order O(h^2), where h is the step size
    :param pixel_array:
    :return:
    """
    # With the Sep size h， the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h^2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With the step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x -1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to the first array axis, and s(scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0 ,2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


    # test
    #test