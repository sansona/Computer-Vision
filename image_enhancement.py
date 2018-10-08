import os
import cv2

import matplotlib.pyplot as plt
from numpy import *
from scipy.ndimage import filters, measurements, morphology
from PIL import Image, ImageFilter
from pylab import *
'''
Collection ofimage enhancementfunctions. Primarily use cv2 & PIL with some numpy
sprinkled in. Most functions are in the form editted_image = function(im), where im
is either the filename of the image or the numpy pixel array
'''

#------------------------------------------------------------------------------


def GetImList(path, extension):
    """
    Returns list of filenames for all images w/ extension (.jpg, .png...) in directory
    """
    return [os.path.join(path, f)
            for f in os.listdir(path) if f.endswith(extension)]

#------------------------------------------------------------------------------


def PrintAsArray(im):
    im = array(Image.open(im))
    print(im)

#------------------------------------------------------------------------------


def LoadImage(im, grayscale=False):
    '''
    allows loading of both image files and image array data into functions
    using same syntax. Returns numpy array for all given input
    '''
    if isinstance(im, ndarray):
        if grayscale == False:
            return im
        else:
            im = Image.fromarray(im).convert('L')
            im = array(im)
            return im
    else:
        if grayscale == False:
            im = array(Image.open(im))
            return im
        else:
            im = array(Image.open(im).convert('L'))
            return im

#------------------------------------------------------------------------------


def ImResize(im, baseWidth=300, save_im=False):
    im = LoadImage(im)
    im = Image.fromarray(im)
    wpercent = (baseWidth / float(im.size[0]))
    hsize = int(float(im.size[1] * float(wpercent)))
    resized = im.resize((baseWidth, hsize), Image.ANTIALIAS)

    if save_im == True:
        result = Image.fromarray(resized).convert('RGB')
        result.save('resized.png')

    return resized

#------------------------------------------------------------------------------


def GammaToLinear(im, save_im=False):
    '''
    source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    '''
    im = LoadImage(im)
    invGamma = 1.0 / 2.2
    table = array([((i / 255.0)**invGamma) * 255
                   for i in arange(0, 256)]).astype('uint8')

    linear = cv2.LUT(im, table)

    if save_im == True:
        linear_im = Image.fromarray(linear).convert('RGB')
        linear_im.save('linear.png')

    return linear

#------------------------------------------------------------------------------


def LinearToGamma(im, save_im=False):
    im = LoadImage(im)
    Gamma = 2.2
    table = array([((i / 255.0)**Gamma) * 255
                   for i in arange(0, 256)]).astype('uint8')

    gamma = cv2.LUT(im, table)

    if save_im == True:
        gamma_im = Image.fromarray(gamma).convert('RGB')
        # since usually final step, naming accordingly
        gamma_im.save('enhanced.png')

    return gamma

#------------------------------------------------------------------------------


def Histeq(im, nbr_bins=256, save_im=False):
    """
    Histogram equalization of grayscale image. Remaps image to new range via cdf
    """
    im = LoadImage(im, grayscale=True)
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalization

    im2 = interp(im.flatten(), bins[:-1], cdf)

    reshaped = im2.reshape(im.shape)
    if save_im == True:
        result = Image.fromarray(reshaped).convert('RGB')
        result.save('hist.png')

    return reshaped, cdf

#------------------------------------------------------------------------------


def ContourPlot(im):
    im = LoadImage(im, grayscale=True)
    #im = array(Image.open(im).convert('L'))
    contour(im, origin='image')
    axis('equal')
    axis('off')
    figure()
    hist(im.flatten(), 128)
    show()

#------------------------------------------------------------------------------


def Invert(im, save_im=False):
    im = LoadImage(im, grayscale=True)
    inv_arr = 255 - im

    if save_im == True:
        inv_im = Image.fromarray(inv_arr).convert('RGB')
        inv_im.save('inverted.png')

    return inv_arr

#------------------------------------------------------------------------------


def GaussianBlur(im, alpha=2, color=False, save_im=False):
    if color == True:
        im = LoadImage(im)
        im2 = zeros(im.shape)
        for i in range(3):
            # applies filter for each channel
            im2[:, :, i] = filters.gaussian_filter(im[:, :, i], alpha)
        im2 = array(im2, 'uint8')
    else:
        im = LoadImage(im, grayscale=True)
        im2 = filters.gaussian_filter(im, alpha)

    if save_im == True:
        result = Image.fromarray(array(im2, 'uint8')).convert('RGB')
        result.save('gaussian_blur_{}.png'.format(alpha))

    return im2

#------------------------------------------------------------------------------


def BinaryThreshold(im, save_im=False):
    '''
    useful for background separation in images w/ simple, distinguishable background
    '''
    im = LoadImage(im, grayscale=True)
    ret, thresh = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

    if save_im == True:
        thresh_im = Image.fromarray(array(thresh, 'uint8')).convert('RGB')
        thresh_im.save('thresh.png')

    return thresh

#------------------------------------------------------------------------------


def AdaptiveThreshold(im, method='Gaussian', save_im=False):
    '''
    useful for images w/ more complicated backgrounds
    '''
    im = LoadImage(im, grayscale=True)
    if method != 'Gaussian':
        thresh = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    else:
        thresh = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    if save_im == True:
        thresh_im = Image.fromarray(array(thresh, 'uint8')).convert('RGB')
        thresh_im.save('adaptive_thresh.png')

    return thresh

#------------------------------------------------------------------------------


def OtsuThreshold(im, save_im=False):
    # use this for bimodal images
    im = LoadImage(im, grayscale=True)
    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if save_im == True:
        thresh_im = Image.fromarray(array(thresh, 'uint8')).convert('RGB')
        thresh_im.save('otsu_thresh.png')

    return thresh

#------------------------------------------------------------------------------


def FloodFill(im, n=127, threshold_method='adaptive', save_im=False):
    '''
    floodfill function. Returns floodfill, inverted floodfill, and foreground mask.
    Levers : n, threshold_method. Try different combinations to find one that works
    '''
    im = LoadImage(im, grayscale=True)

    if threshold_method == 'binary':
        th, im = cv2.threshold(im, n, 255, cv2.THRESH_BINARY_INV)
    elif threshold_method == 'adaptive':
        im = cv2.adaptiveThreshold(im, n, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    elif threshold_method == 'otsu':
        ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    im_floodfill = im.copy()

    h, w = im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # combine the two images to get the foreground
    fill_image = im | im_floodfill_inv

    im_floodfill = Image.fromarray(array(im_floodfill, 'uint8')).convert('RGB')
    im_floodfill_inv = Image.fromarray(
        array(im_floodfill_inv, 'uint8')).convert('RGB')

    # this is a bit inefficient
    fill_image_arr = array(fill_image, 'uint8')
    fill_image = Image.fromarray(fill_image_arr).convert('RGB')
    fill_image_arr = array(fill_image)

    if save_im == True:
        # im_floodfill.save('im_floodfill.png')
        # im_floodfill_inv.save('im_floodfill_inv.png')
        # fill_image.save('fill.png')
        fill_image.save('fill.png')

    return fill_image_arr

#------------------------------------------------------------------------------


def BlackToTransparent(im):
    '''
    converts black pixels to transparent for use in mask.

    only takes in image file as arg since need RGBA. Returns image file since
    needs to be .png
    '''
    im = Image.open(im).convert('RGBA')

    im_data = im.getdata()

    new_pixels = []
    for item in im_data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_pixels.append((0, 0, 0, 0))
        else:
            new_pixels.append(item)

    im.putdata(new_pixels)
    im.save('transparent_mask.png', 'PNG')

#------------------------------------------------------------------------------


def AlphaBlend(foreground_im, mask, save_im=False):
    '''
    overlays mask & image. Best use is transparent binary mask
    '''
    fore = LoadImage(foreground_im)
    mask = LoadImage(mask)
    assert fore.shape == mask.shape

    blended = cv2.addWeighted(fore, 1.0, mask, 0.1, 0)

    if save_im == True:
        cv2.imwrite('blended.png', blended)

    return blended

#------------------------------------------------------------------------------


def ContrastEnhance(im, brightness=32, contrast=20, save_im=False):
    # suggest using CLAHE() instead in most cases
    im = LoadImage(im)
    contrast = cv2.addWeighted(
        im,
        1. +
        contrast /
        127.,
        im,
        0,
        brightness -
        contrast)

    if save_im == True:
        cv2.imwrite('contrast_enhanced.png', contrast)

    return contrast

#------------------------------------------------------------------------------


def CLAHE(im, save_im=False):
    '''
    Contrast-limited adaptive histogram equalization function
    '''
    # not sure why turns out weird when use LoadImage(). Needs image file as
    # input
    im = cv2.imread(im, 1)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if save_im == True:
        cv2.imwrite('CLAHE.png', contrast)

    return contrast

#------------------------------------------------------------------------------


def UnsharpMask(im, alpha=2, save_im=False):
    '''
    uses blurred image as mask to sharpen image
    '''
    im2 = GaussianBlur(im, alpha)  # mask
    im = LoadImage(im, grayscale=True)

    unsharp_im = cv2.subtract(im, im2)

    if save_im == True:
        result = Image.fromarray(array(unsharp_im, 'uint8'))
        result = result.convert('RGB')
        result.save('unsharp_mask_{}.png'.format(alpha))

    return unsharp_im

#------------------------------------------------------------------------------
