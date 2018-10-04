#!/usr/bin/python3
import os

from image_enhancement_functions import (GammaToLinear, GaussianBlur,
                                         FloodFill, AlphaBlend, CLAHE, LinearToGamma)

# Example pipeline for contrast enhancement. Main lever is threshold &
# threshold method
image = './test_images/sudoku.jpg'
lin = GammaToLinear(image)
blur = GaussianBlur(lin)
# try binary, adaptive, and otsu threholding
fill = FloodFill(blur, threshold_method='adaptive')
AlphaBlend(lin, fill, save_im=True)
contrast = CLAHE('blended.png')
LinearToGamma(contrast, save_im=True)

redundant_im = ['blended.png', 'fill.png', 'linear.png']
for image in redundant_im:
    if os.path.isfile(image):
        os.remove(image)
