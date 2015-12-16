import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from PIL import Image
from pylab import *
from scipy.ndimage import filters
from os import walk
import string

import matplotlib.pyplot as plt
from matplotlib.pyplot import *

from pytesseract import * 

def get_harris_points(harrisim,min_dist=10,threshold=0.1):
	# find top corner candidates above a threshold
	corner_threshold = harrisim.max() * threshold
	harrisim_t = (harrisim > corner_threshold) * 1

	# get coordinates of candidates
	coords = np.array(harrisim_t.nonzero()).T # ...and their values
	candidate_values = [harrisim[c[0],c[1]] for c in coords] # sort candidates
	index = np.argsort(candidate_values)

	# store allowed point locations in array 
	allowed_locations = np.zeros(harrisim.shape) 
	allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
	
	# select the best points taking min_distance into account
	filtered_coords = [] 
	for i in index:
		if allowed_locations[coords[i,0],coords[i,1]] == 1:
			filtered_coords.append(coords[i]) 
			allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
			(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0 
	return filtered_coords

def compute_harris_response(im,sigma=3):
	# derivatives
	imx = np.zeros(im.shape)
	filters.gaussian_filter(im, (sigma,sigma), (0,1), imx) 
	imy = np.zeros(im.shape)
	filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
	#Compute components of the Harris matrix
	Wxx = filters.gaussian_filter(imx*imx,sigma) 
	Wxy = filters.gaussian_filter(imx*imy,sigma) 
	Wyy = filters.gaussian_filter(imy*imy,sigma)
	#Determinant and trace
	Wdet = Wxx*Wyy - Wxy**2
	Wtr = Wxx + Wyy
	return Wdet / Wtr

def plot_harris_points(image,filtered_coords):
	figure()
	gray()
	imshow(image)
	plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords] ,'*') 
	axis('off')	
	show()

def orderPoints(pts):
    rect = np.zeros((4,2), dtype = 'float32')
    # print rect 
    s = np.sum(pts, axis = 1)
    # print s
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # print diff
    
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]
    # print pts
    # print rect
    return rect
    #print 'finished order points'

def fourPointsTransform(image, pts):
    #Order points
    # print pts
    rect = orderPoints(pts)
    (tl, tr, br, bl) = rect
    #Compute distances
    #Compute the width of the new image
    widthA = np.sqrt(((br[0]-bl[0])**2)+((br[1]-br[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA),int(widthB))
    #print maxWidth
    #Compute the height of the new image 
    heightA = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    heightB = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    maxHeight = max(int(heightA), int(heightB))
    #print maxHeight
    #Create a new array with the new coordinates
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    #Compute the perspective transfor matrix a apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image,M, (maxWidth, maxHeight))    
    return warped


def main(): 

	dirpath = '../Images/TestSetSegmentation/'
	dirpathOut = '../Images/TestSetSegmentationOut/'
	filePDDI = []

	for dirpath, dirname, filename in walk(dirpath):
		filePDDI.extend(filename)

	for elementFile in filePDDI[:]:
		filename = dirpath+elementFile
		im_in = np.array(Image.open(filename).convert('L'))

		# Threshold.
		# Set values equal to or above 220 to 0.
		# Set values below 220 to 255.
		th, im_th = cv2.threshold(im_in, 100, 255, cv2.THRESH_BINARY_INV);

		sobelxEnchanced = cv2.Sobel(im_th,cv2.CV_64F,1,0,ksize=3)
		sobelyEnchanced = cv2.Sobel(im_th,cv2.CV_64F,0,1,ksize=3)
		# print sobelxEnchanced
		# print sobelyEnchanced
		  
		# Combine the two images to get the foreground.
		im_out = abs(sobelxEnchanced * sobelyEnchanced)

		median = cv2.GaussianBlur(im_out,(15,15),0)
		edges = median
		maxEdges = np.max(edges)
		normEdges = abs(edges/(maxEdges * 1.00)) # normalization

		# print normEdges

		binarizationTreshold = 0.2
		out_th = 1*(normEdges>binarizationTreshold)

		harrisim = compute_harris_response(out_th) 
		filtered_coords = get_harris_points(harrisim, 4)

		points = []
		maxSum = 0
		minSum = 10000
		maxDiff = 0
		minDiff = 10000

		for element in filtered_coords:
			s = element[0]+element[1]
			d = element[1]-element[0]
			if s > maxSum:
				pointsSum = element
				maxSum = s
			if s < minSum:
				pointsMinSum = element
				minSum = s
			if d > maxDiff:
				pointsDiff = element
				maxDiff = d
			if d < minDiff:
				pointsMinDiff = element
				minDiff = d

		points.append(pointsMinSum)
		points.append(pointsSum)
		points.append(pointsDiff)
		points.append(pointsMinDiff)
		pointsNp = np.asarray(points)
		# plot_harris_points(im_in, points)


		listPoints = []
		for element in points:
			tempTuple = (element[1],element[0])
			listPoints.append(tempTuple)

		warped = fourPointsTransform(im_in, listPoints)
		small = cv2.resize(warped, (0,0), fx=3, fy=3)
		print image_to_string(Image.fromarray(small), lang='spa') #SingleWord

		# outFileName = dirpathOut+string.split(elementFile,'.')[0]+'Segmented.png'
		# cv2.imwrite(outFileName, small)
 
		# Display images.
		# cv2.imshow("X", sobelxEnchanced)
		# cv2.imshow("Y", sobelyEnchanced)
		# cv2.imshow("X+Y", out_th)
		# cv2.waitKey(0)

if __name__=='__main__':
	main()