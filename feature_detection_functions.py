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
	im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 
		cv2.CHAIN_APPROX_SIMPLE)

	contour_im = cv2.drawContours(im, contours, -1, 
		(0,255,0), 5)

	if save_im==True:
		cv2.imwrite('contours.tif', contour_im)

	return contour_im, contours, hierarchy

#--------------------------------------------------------------------------
def MaxApproxContour(all_contours, hierarchy):
	'''
	converts max absolute contour to approximate contour
	'''
	largest_contour = None 
	largest_area = 0
	min_area = 50
	
	#finds top left and bottom right corners through sum of coordinater
	for contour in all_contours:
		area = cv2.contourArea(contour)
		if area > largest_area: 
			largest_area = area
			largest_contour = contour

	#sets approximate contour
	epsilon = 0.1*cv2.arcLength(largest_contour, True)
	appr_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

	points = [] 
	for i in range(4):
		points.append((appr_contour[i][0][0], appr_contour[i][0][1]))

	#print(points)
	return points

#--------------------------------------------------------------------------
def DrawRectangle(im, corner_list, save_im=False):
	'''
	overlays rectangle on image given coordinates of corners
	'''
	top_left = None
	bottom_right = None
	minSum = 100000
	maxSum = 0
	for corner in corner_list:
		if corner[0] + corner[1] < minSum:
			minSum = corner[0] + corner[1]
			top_left = corner
		elif corner[0] + corner[1] > maxSum:
			maxSum = corner[0] + corner[1]
			bottom_right = corner
	rect = cv2.rectangle(im, top_left, bottom_right, (0,0,255), 8)

	print('Corner coordinates: (%s %s)' %(top_left, bottom_right))
	if save_im==True:
		cv2.imwrite('rectangle.tif', rect)

	return rect, [top_left, bottom_right]

#--------------------------------------------------------------------------
def CropToRectangle(im, corner_coords, save_im=False):
	'''
	crops image to largest contour detected via. corner coordinates
	'''
	x_min = corner_coords[0][0]
	y_min = corner_coords[0][1]
	x_max = corner_coords[1][0]
	y_max = corner_coords[1][1]

	cropped = im[y_min:y_max, x_min:x_max]
	if save_im==True:
		cv2.imwrite('cropped_rect.tif', cropped)

	return cropped

#--------------------------------------------------------------------------
def DrawGridOverImg(im, n=9, save_im=False):
	'''
	takes in cropped grid, overlays nxn grid
	'''
	im = LoadImage(im)
	rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
	width, height = im.shape[:2]
	width_inc = int(width/n)
	height_inc = int(height/n)

	x_col = []
	y_col = []
	#makes list of all endpoints of lines
	for i in range(1,n+1):
		x_col.append([(width_inc*i, 0), (width_inc*i, height)])
		y_col.append([(0, height_inc*i), (width, height_inc*i)])

	x_arr = asarray(x_col)
	y_arr = asarray(y_col)
	lines = vstack((x_arr, y_arr)) #list of all lines

	polylines = cv2.polylines(rgb_im, lines, False, (0,0,255), 5)
	if save_im==True:
		cv2.imwrite('polylines.tif', polylines)

	return polylines

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
def PHoughLineDetection(im, threshold=50):
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
