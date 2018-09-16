import os
import cv2

import matplotlib.pyplot as plt
from numpy import *
from scipy.ndimage import filters, measurements, morphology
from PIL import Image, ImageFilter
from pylab import *
'''
Collection of computer vision functions. Primarily use cv2 & PIL with some numpy
sprinkled in.
'''
def GetImList(path, extension):
	"""
	Returns list of filenames for all images w/ extension (.jpg, .tif...) in directory
	"""
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(extension)]

#--------------------------------------------------------------------------
def ImResize(im, baseWidth=300):
	"""
	Resize image array
	Usage: im = Image.open(img)
	"""
	wpercent  = (baseWidth/float(im.size[0]))
	hsize = int(float(im.size[1]*float(wpercent)))
	resized = im.resize((baseWidth, hsize), Image.ANTIALIAS)
	resized.save('resized.jpg')

	#result = Image.fromarray(resized).convert('RGB')
	#result.save('resized.jpg')
	return resized

#--------------------------------------------------------------------------
def Histeq(im, nbr_bins=256):
	"""
	Histogram equalization of grayscale image. Remaps image to new range via cdf
	"""
	im = array(Image.open(im).convert('L'))
	imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
	cdf = imhist.cumsum()
	cdf = 255*cdf/cdf[-1] #normalization

	im2 = interp(im.flatten(), bins[:-1], cdf)

	reshaped = im2.reshape(im.shape)
	result = Image.fromarray(reshaped).convert('RGB')
	result.save('hist.jpeg')

	return reshaped, cdf

#--------------------------------------------------------------------------
def ContourPlot(im):
	im = array(Image.open(im).convert('L'))
	contour(im, origin='image')
	axis('equal')
	axis('off')
	figure()
	hist(im.flatten(), 128)
	show()

#--------------------------------------------------------------------------
def ComputeAverage(imlist):
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

	#result = Image.fromarray(array(averageim, 'uint8')).convert('RGB')
	#result.save('averaged.jpeg')

	return array(averageim, 'uint8')

#--------------------------------------------------------------------------
def pca(X):
	""" 
	Usage: X, matrix with training data stored as flattened arrays in rows
	return: projection matrix (with important dimensions first), variance
	and mean.
	"""
	#get dimensions
	num_data,dim = X.shape

	#center data
	mean_X = X.mean(axis=0)
	X = X - mean_X

	if dim>num_data:
		#PCA - compact trick used
		M = dot(X,X.T) #covariance matrix
		e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
		tmp = dot(X.T,EV).T #this is the compact trick
		V = tmp[::-1] #reverse since last eigenvectors are the ones we want
		S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
		for i in range(V.shape[1]):
			V[:,i] /= S
	else:
		#PCA - SVD used
		U,S,V = linalg.svd(X)
		V = V[:num_data] # only makes sense to return the first num_data

	#return the projection matrix, the variance and the mean
	return V,S,mean_X

#--------------------------------------------------------------------------
def Denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
	"""
	Rudin-Osher-Fatemi (ROF) denoising model

	Input: noisy input image (grayscale), initial guess for U, weight of
	the TV-regularizing term, steplength, tolerance for stop criterion.

	Output: denoised and detextured image, texture residual. 
	"""

	m,n = im.shape #size of noisy image

	# nitialize
	U = U_init
	Px = im #x-component to the dual field
	Py = im #y-component of the dual field
	error = 1

	while (error > tolerance):
		Uold = U

		#gradient of primal variable
		GradUx = roll(U,-1,axis=1)-U #x-component of U's gradient
		GradUy = roll(U,-1,axis=0)-U #y-component of U's gradient

		#update the dual varible
		PxNew = Px + (tau/tv_weight)*GradUx
		PyNew = Py + (tau/tv_weight)*GradUy
		NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

		Px = PxNew/NormNew #update of x-component (dual)
		Py = PyNew/NormNew #update of y-component (dual)

		#update the primal variable
		RxPx = roll(Px,1,axis=1) #right x-translation of x-component
		RyPy = roll(Py,1,axis=0) #right y-translation of y-component

		DivP = (Px-RxPx)+(Py-RyPy) #divergence of the dual field.
		U = im + tv_weight*DivP #update of the primal variable

		#update of error
		error = linalg.norm(U-Uold)/sqrt(n*m);

	return U,im-U #denoised image and texture residual

#--------------------------------------------------------------------------
def GaussianBlur(im, alpha=5, color=False):
	'''
	Simple function to apply blur to image.
	Usage: GaussianBlur(im)
	'''
	if color==True:
		im = array(Image.open(im))
		im2 = zeros(im.shape)
		for i in range(3):
			#applies filter for each channel
			im2[:,:,i] = filters.gaussian_filter(im[:,:,i],alpha)
		im2 = array(im2, 'uint8')
	else:
		im = array(Image.open(im).convert('L'))
		im2 = filters.gaussian_filter(im, alpha)

	#result = Image.fromarray(array(im2, 'uint8')).convert('RGB')
	#result.save('gaussian_blur_{}.jpg'.format(alpha))

	return im2

#--------------------------------------------------------------------------
def Threshold(im):
	'''
	simple background separation
	'''
	im = GaussianBlur(im, 2)

	gaus_thres = cv2.adaptiveThreshold(im, 255, 
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	result = Image.fromarray(array(gaus_thres, 'uint8')).convert('RGB')
	result.save('thresholded.jpg')

	return gaus_thres

#--------------------------------------------------------------------------
def UnsharpMask(im, alpha=2):
	'''
	uses blurred image as mask to sharpen image
	'''
	im2 = GaussianBlur(im, alpha) #mask
	im = array(Image.open(im).convert('L'))

	unsharp_im = cv2.subtract(im, im2)

	#result = Image.fromarray(array(unsharp_im, 'uint8'))
	#result = result.convert('RGB')
	#result.save('unsharp_mask_{}.jpg'.format(alpha))

	return unsharp_im

#--------------------------------------------------------------------------
def FindOutline(im):
	'''
	returns outline of grayscale image w/ high background contrast
	'''
	im = Image.open(im).convert('L')
	im2 = im.filter(ImageFilter.FIND_EDGES)
	im2.save('outline.jpg')	

#--------------------------------------------------------------------------
def FindOutline_grad(im, filt='Laplacian'):
	'''
	uses gradient filter to detect object outlines. Includes option for 
	Sobel filters
	'''
	im = cv2.imread(im, 0)

	if filt == 'sobelx':
		sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
		x_im = Image.fromarray(sobelx).convert('RGB')
		x_im.save('sobelx.jpg')
	if filt == 'sobely':
		sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)
		y_im = Image.fromarray(sobely).convert('RGB')
		y_im.save('sobely.jpg')
	else:
		lap = cv2.Laplacian(im, cv2.CV_64F)
		lap_im = Image.fromarray(lap).convert('RGB')
		lap_im.save('lap.jpg')

#--------------------------------------------------------------------------
def HoughLineDetection(im, threshold=100):
	'''
	lines detection using probabilistic Hough transform
	'''
	im = cv2.imread(im)
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150)
	minLineLength = 100
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges, 1 , pi/180, 
		threshold, minLineLength, maxLineGap)

	a,b,c = lines.shape
	for i in range(a):
		cv2.line(im, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], 
			lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

	#cv2.imwrite('edges.jpg', edges)
	cv2.imwrite('houghlines.jpg',im)

#--------------------------------------------------------------------------
def CountObjects(im, iterations=1):
	'''
	counts objects in image, gets distribution of object sizes
	'''
	#thresholds image, makes binary
	im = array(Image.open(im).convert('L'))
	im = 1*(im<128)
	im = morphology.binary_opening(im, iterations=iterations)

	labels, num_obj = measurements.label(im)
	print('Number of objects: %s' %num_obj)
	labelled_im = Image.fromarray(labels).convert('RGB')
	labelled_im.save('labelled_im.jpg')

	#get distribution of obj sizes
	obj_sizes = {}
	for obj in labels.ravel():
		if obj in obj_sizes:
			obj_sizes[obj]+=1
		else:
			obj_sizes[obj] = 1
	del obj_sizes[0]

	return labels, num_obj, obj_sizes

#--------------------------------------------------------------------------
def LabelCenterOfMass(im):
	'''
	WIP. Trying to get to return multiple centers 
	'''
	im = array(Image.open(im).convert('L'))
	center_mass = measurements.center_of_mass(im)
	
