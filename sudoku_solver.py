#!/usr/bin/python3

from image_enhancement_functions import *
from feature_detection_functions import *

image = LoadImage('./test_images/sudoku2.jpg')

# preprocessing
blurred = GaussianBlur(image, alpha=3)
adaptive_thresh = AdaptiveThreshold(blurred)
inv = Invert(adaptive_thresh)

# use largest approximated contour to find and crop to outer perimeter
cont_im, cont, hier = FindContours(inv)
list_corners = MaxApproxContour(cont, hier)
rect, corners = DrawRectangle(image, list_corners)
crop_rect = CropImToRectangle(inv, corners)

# divides image into 9x9 grid, overlays grid on cropped image
inv_crop = Invert(crop_rect)
OCROnTiles(inv_crop)
#DrawGridOverImg(inv_crop, save_im=True)
