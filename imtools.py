from PIL import Image
from numpy import *
import os

def get_imlist(path, extension):
	"""
	Returns list of filenames for all images w/ extension (.jpg, .tif...) in directory
	"""
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(extension)]

def imresize(im, baseWidth=300):
	"""
	Resize image array
	Usage: im = Image.open(img)
	"""
	wpercent  = (baseWidth/float(im.size[0]))
	hsize = int(float(im.size[1]*float(wpercent)))
	resized = im.resize((baseWidth, hsize), Image.ANTIALIAS)
	resized.save('resized.jpg')

	#result = Image.fromarray(resized)
	#result = result.convert('RGB')
	#result.save('test2.jpg')
	return resized


def histeq(im, nbr_bins=256):
	"""
	Histogram equalization of grayscale image. Remaps image to new range via cdf
	Usage: im = array(Image.open(img).convert('L'))
	"""
	imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
	cdf = imhist.cumsum()
	cdf = 255*cdf/cdf[-1] #normalization

	im2 = interp(im.flatten(), bins[:-1], cdf)

	reshaped = im2.reshape(im.shape)
	#result = Image.fromarray(reshaped)
	#result = result.convert('RGB')
	#result.save('test.jpeg')

	return reshaped, cdf

def compute_average(imlist):
	"""
	Returns average of list of images - need to be same size
	"""
	averageim = array(Image.open(imlist[0]), 'f')

	for imname in imlist[1:]:
		try:
			averageim =+ array(Image.open(imname))
		except:
			print (imname + '...skipped')
	averageim = averageim/len(imlist)

	result = Image.fromarray(array(averageim, 'uint8'))
	result = result.convert('RGB')
	result.save('test.jpeg')

	return array(averageim, 'uint8')
