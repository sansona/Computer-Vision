#!/usr/bin/python3
import os

from image_enhancement import *

# Example pipeline for contrast enhancement. Main lever is threshold &
# threshold method

im_extensions = ['.jpg', '.jpeg', '.png', '.tif']
path = os.getcwd()
filelist = [f for f in os.listdir(path) if f.endswith(tuple(im_extensions))]
print(filelist)
for f in filelist:

    lin = GammaToLinear(f)
    blur = GaussianBlur(lin)
    # try binary, adaptive, and otsu threholding
    fill = FloodFill(blur, threshold_method='adaptive')
    AlphaBlend(lin, fill, save_im=True)
    contrast = CLAHE('blended.png')
    LinearToGamma(contrast, f, save_im=True)

redundant_im = ['blended.png', 'fill.png', 'linear.png']
for image in redundant_im:
    if os.path.isfile(image):
        os.remove(image)
