#!/usr/bin/python3
import os
import cv2

import matplotlib.pyplot as plt
from numpy import *
from scipy.ndimage import filters, measurements
from skimage import morphology
from PIL import Image, ImageFilter
from pylab import *

from image_enhancement_functions import LoadImage

#--------------------------------------------------------------------------
def FindContours(im, save_im=False):
	'''
	used for detecting overall grid pattern
	'''
	im = LoadImage(im, grayscale=True)
	ret, thresh = cv2.threshold(im, 127, 255, 0)
	im, contours, hi = cv2.findContours(thresh, cv2.RETR_TREE, 
		cv2.CHAIN_APPROX_SIMPLE)

	contour_im = cv2.drawContours(im, contours, -1, 
		(0,255,0), 12)

	if save_im==True:
		cv2.imwrite('contours.tif', contour_im)

	return contour_im

#--------------------------------------------------------------------------
def FilterSmallObjects(im):
	im = LoadImage(im)
	filtered = morphology.remove_small_objects(im, 50000)
	cv2.imwrite('filtered.tif', filtered)

#--------------------------------------------------------------------------
def DetectBlob(im):
	im = cv2.imread(im, cv2.IMREAD_GRAYSCALE)

	param = cv2.SimpleBlobDetector_Params()
	param.filterByArea=True   
	param.minArea=20
	param.filterByCircularity = False
	param.filterByColor = False
	param.filterByConvexity = False
	param.filterByInertia = False

	detector = cv2.SimpleBlobDetector_create(param)

	keypoints = detector.detect(im)

	blob_im = cv2.drawKeypoints(im, keypoints, array([]),
		(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imwrite('blobs.tif', blob_im)

#--------------------------------------------------------------------------
def FindOutline(im):
	'''
	returns outline of grayscale image w/ high background contrast
	'''
	im = LoadImage(im, grayscale=True)
	im = Image.fromarray(im)
	im2 = im.filter(ImageFilter.FIND_EDGES)
	im2.save('outline.tif')	

#--------------------------------------------------------------------------
def FindOutline_grad(im, filt='Laplacian'):
	'''
	uses gradient filter to detect object outlines. Includes option for 
	Sobel filters
	'''
	im = LoadImage(im, grayscale=True)

	if filt == 'sobelx':
		sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
		x_im = Image.fromarray(sobelx).convert('RGB')
		x_im.save('sobelx.tif')
		return sobelx
	if filt == 'sobely':
		sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)
		y_im = Image.fromarray(sobely).convert('RGB')
		y_im.save('sobely.tif')
		return sobely
	else:
		lap = cv2.Laplacian(im, cv2.CV_64F)
		lap_im = Image.fromarray(lap).convert('RGB')
		lap_im.save('lap.tif')
		return lap

#--------------------------------------------------------------------------
def HoughLineDetection(im):
	im = LoadImage(im)
	gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, otsu_im = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	#inverting & eroding seems to make edge detection work better in most 
	#	test cases
	otsu_im = 255 - otsu_im

	cv2.imwrite('otsu.tif', otsu_im)
	kernel = ones((4,4), 'uint8')
	eroded = cv2.erode(otsu_im, kernel, iterations=1)

	edges = cv2.Canny(eroded, 150, 200, 3, 5)

	lines = cv2.HoughLines(edges, 1, pi/180, 500)
	#print(lines)
	for line in lines:
		rho, theta = line[0]
		a = cos(theta)
		b = sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

	cv2.imwrite('houghlines.tif',im)

#--------------------------------------------------------------------------
def PHoughLineDetection(im, threshold=100):
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

	cv2.imwrite('Phoughlines.tif', im)

#--------------------------------------------------------------------------
def LabelCenterOfMass(im):
	'''
	WIP. Currently detects overall center. Want to have it detect multiple 
	'''
	im = cv2.imread(im, 0)
	ret, thresh = cv2.threshold(im, 127, 255, 0)
	image, contours, hierarchy = cv2.findContours(thresh, 1, 2)

	cnt = contours[0]
	m = cv2.moments(cnt)
	cx = int(m['m10']/m['m00'])
	cy = int(m['m01']/m['m00'])
	print(m)
	print(cx, cy)

#--------------------------------------------------------------------------
