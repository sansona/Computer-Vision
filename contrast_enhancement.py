import os 

from image_enhancement_functions import (GammaToLinear, GaussianBlur, 
FloodFill, AlphaBlend, CLAHE, LinearToGamma)

#Example pipeline for contrast enhancement. Main lever is threshold & threshold method
image = 'tea2.jpeg'
lin = GammaToLinear(image)
blur = GaussianBlur(lin)
#try binary, adaptive, and otsu threholding
fill = FloodFill(blur, threshold_method='adaptive') 
AlphaBlend(lin, fill, save_im=True)
contrast = CLAHE('blended.tif')
LinearToGamma(contrast, save_im=True)

redundant_im = ['blended.tif', 'fill.tif', 'linear.tif']
for image in redundant_im:
	if os.path.isfile(image):
		os.remove(image)
