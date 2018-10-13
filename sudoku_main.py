#!/usr/bin/python3

from image_enhancement import *
from feature_detection import *

image = LoadImage('./test_images/sudoku_news.png')

# preprocessing - use generic image processing functions I made
blurred = GaussianBlur(image, alpha=3)
adaptive_thresh = AdaptiveThreshold(blurred)
inv = Invert(adaptive_thresh)

# use largest approximated contour to find and crop to outer grid perimeter
cont_im, cont, hier = FindContours(inv)
list_corners = MaxApproxContour(cont, hier)
rect, corners = DrawRectangle(image, list_corners, save_im=True)
crop_rect = CropImToRectangle(inv, corners)

# divides image into 9x9 grid, runs OCR, solves grid from detected OCR values
inv_crop = Invert(crop_rect)
flat = OCROnTiles(inv_crop)
SolveOCRGrid(flat)
