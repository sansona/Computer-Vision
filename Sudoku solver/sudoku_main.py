#!/usr/bin/python3

from image_enhancement import *
from feature_detection import *

#------------------------------------------------------------------------------
# main issue comes w/ OCR unable to recognize digits successfully for specific
# puzzle. If this is the case, I recommend retraining the model with new
# training data using the functions in svm.py.
#------------------------------------------------------------------------------


image = LoadImage('path/to/sudoku_image')

# preprocessing - use generic image processing functions I made
blurred = GaussianBlur(image, alpha=3)
adaptive_thresh = AdaptiveThreshold(blurred)
inv = Invert(adaptive_thresh)

# use largest approximated contour to find and crop to outer grid perimeter
cont_im, cont, hier = FindContours(inv)
list_corners = MaxApproxContour(cont, hier)
rect, corners = DrawRectangle(image, list_corners)
crop_rect = CropImToRectangle(inv, corners)

# divides image into nxn grid, runs OCR, solves grid from detected OCR values, # overlays solved puzzle onto cropped image
inv_crop = Invert(crop_rect)
flat, start_board = OCROnTiles(inv_crop)
SolveOCRGrid(flat)
DrawSolvedGrid(inv_crop, flat, start_board)
