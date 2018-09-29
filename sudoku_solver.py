
from image_enhancement_functions import *
from feature_detection_functions import *

image = ###
blurred = GaussianBlur(image)
otsu = OtsuThreshold(blurred)
inverted = Invert(otsu)
contours = FindContours(inverted)
blur2 = GaussianBlur(contours, alpha=3)
BinaryThreshold(blur2, save_im=True)

#next step: work on removing noise from thresholded image