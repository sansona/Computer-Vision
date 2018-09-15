import os
import cv2
from numpy import *
from scipy.ndimage import filters
from PIL import Image
from pylab import *


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

	#result = Image.fromarray(resized)
	#result = result.convert('RGB')
	#result.save('resized.jpg')
	return resized

#--------------------------------------------------------------------------
def Histeq(im, nbr_bins=256):
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
	#result.save('hist.jpeg')

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

	#result = Image.fromarray(array(averageim, 'uint8'))
	#result = result.convert('RGB')
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

	#result = Image.fromarray(array(im2, 'uint8'))
	#result = result.convert('RGB')
	#result.save('gaussian_blur_{}.jpg'.format(alpha))

	return im2

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

