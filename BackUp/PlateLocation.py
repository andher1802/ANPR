
import string

from PIL import Image
from pylab import *

import cv2
import numpy as np

from CVTools.toolsPerspective import fourPointsTransform
from scipy.ndimage import measurements, morphology

import matplotlib.pyplot as plt
import collections
# from svmutil import *




def main():
	# WorkFlow for plate detection and clipping
	# 1. Filter the image for edge detection 
	# 2. Do a Vertical Projection and show the line graph in a 2D graph
	# 3. Then apply a convolutiom with a rank filter for detection of candidates to plates
	# 4. Apply a band clipping
	# 5. Over the band, apply a horizontal projection and show the result in a 2D plot
	# 6. Detect the plate using the same approach as in 4
	# 7. Derivate the horizontal projection 
	# 8. compute the max and min of the derivative for getting the BW and WB transitions
	# 9. Apply the plate clipping

	### Reading of the image and transforming it into an array for mathematical operations
	image = np.array(Image.open('./images/Car3.jpg').convert('L'))

	### Filter image for getting rid of noise
	### http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
	blur = cv2.GaussianBlur(image,(31,31),0)
	median = cv2.medianBlur(image,9)
	bilateral = cv2.bilateralFilter(image,9,75,75)

	#Select image to filter
	image2Filter = blur

	### Filter using laplacian derivatives as http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
	### http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
	laplacian = uint8(cv2.Laplacian(image2Filter,cv2.CV_64F))
	sobelx = uint8(cv2.Sobel(image2Filter,cv2.CV_64F,1,0,ksize=3))
	sobely = uint8(cv2.Sobel(image2Filter,cv2.CV_64F,0,1,ksize=3))
	canny = uint8(cv2.Canny(image2Filter,10,50))

	###compute the new intensity function for image enchancement
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(sobely,kernel,iterations = 1)

	### Add a convolution between the laplacian and a blurGaussian filter
	blurEdge = cv2.GaussianBlur(erosion,(31,31),0) 

	#Normalization for edge function after blurring 
	maxBlurEdge = np.max(blurEdge)
	normBlurEdge = blurEdge/(maxBlurEdge * 1.00) # normalization

	#Computing intensity function 
	ImageToComputeEnchancement = np.array(image, dtype="int32")
	preEnchancementImage = np.array(enchancementFunction(normBlurEdge), dtype="float32")

	#Computing enchanced image
	kernel = np.ones((5,5),np.float32)/25
	blurredImage = cv2.filter2D(image,-1,kernel)
	row = []
	for i in xrange(ImageToComputeEnchancement.shape[0]): 
		column = []
		for j in xrange(ImageToComputeEnchancement.shape[1]):
			jPrima = 0
			jPrima = (preEnchancementImage[i][j]*(ImageToComputeEnchancement[i][j] - blurredImage[i][j])) + blurredImage[i][j]
			column.append(jPrima)
			# print ImageToComputeEnchancement[i][j],  blurredImage[i][j], preEnchancementImage[i][j], jPrima
		row.append(column)
	enchancedImage = np.array(row, dtype="uint8")

	#Computing edges for enchanced image
	laplacianEnchanced = cv2.Laplacian(enchancedImage,cv2.CV_64F)
	sobelxEnchanced = cv2.Sobel(enchancedImage,cv2.CV_64F,1,0,ksize=3)
	sobelyEnchanced = cv2.Sobel(enchancedImage,cv2.CV_64F,0,1,ksize=3)
	cannyEnchanced = cv2.Canny(enchancedImage,10,50)

	
	#Image for being computed by edge density function
	imageForEdging = sobelyEnchanced
	edges = imageForEdging
	maxEdges = np.max(edges)
	normEdges = abs(edges/(maxEdges * 1.00)) # normalization
	
	#binarization of NormEdges
	binaryImage = 1*(normEdges>0.4)
	
	#Algorithm for removing background curves and noise
	imageToNoiseRemoving = binaryImage
	mMatrix = np.zeros((imageToNoiseRemoving.shape[0],imageToNoiseRemoving.shape[1]),np.uint32)
	nMatrix = np.zeros((imageToNoiseRemoving.shape[0],imageToNoiseRemoving.shape[1]),np.uint32)
	tLong = 35
	tShort = 20

	for iPrima in xrange(imageToNoiseRemoving.shape[0]-2-2-1):
		i = iPrima+2 
		for jPrima in xrange(imageToNoiseRemoving.shape[1]-2-2-1):
			j = jPrima+2
			if imageToNoiseRemoving[i][j] == 1:
				# print i, j, imageToNoiseRemoving.shape[0]-3, imageToNoiseRemoving.shape[1]-3, mMatrix.shape[0], mMatrix.shape[1] 
				if (imageToNoiseRemoving[i-1][j-1]+imageToNoiseRemoving[i-1][j]+imageToNoiseRemoving[i-1][j+1]+imageToNoiseRemoving[i][j-1]) > 0:
					mMatrix[i][j] = max(mMatrix[i-1][j-1],mMatrix[i-1][j],mMatrix[i-1][j+1],mMatrix[i][j-1])+1
				else:
					mMatrix[i][j] = max(mMatrix[i-2][j-1],mMatrix[i-2][j],mMatrix[i-2][j+1],mMatrix[i-1][j-2], mMatrix[i-1][j+2], mMatrix[i][j-2])+1

	for iPrima in reversed(xrange(imageToNoiseRemoving.shape[0]-2-2-1)):
		i = iPrima+2 
		for jPrima in reversed(xrange(imageToNoiseRemoving.shape[1]-2-2-1)):
			j = jPrima+2
			if imageToNoiseRemoving[i][j] == 1:
				# print i, j, imageToNoiseRemoving.shape[0]-3, imageToNoiseRemoving.shape[1]-3, mMatrix.shape[0], mMatrix.shape[1] 
				if (imageToNoiseRemoving[i+1][j-1]+imageToNoiseRemoving[i+1][j]+imageToNoiseRemoving[i+1][j+1]+imageToNoiseRemoving[i][j+1]) > 0:
					nMatrix[i][j] = max(nMatrix[i+1][j-1],nMatrix[i+1][j],nMatrix[i+1][j+1],nMatrix[i][j+1])+1
				else:
					nMatrix[i][j] = max(nMatrix[i+2][j-1],nMatrix[i+2][j],nMatrix[i+2][j+1],nMatrix[i+1][j-2],nMatrix[i+1][j+2],nMatrix[i][j+2])+1

	imageDeNoised = np.array(imageToNoiseRemoving)
	for i in xrange(imageDeNoised.shape[0]-1):  
		for j in xrange(imageDeNoised.shape[1]-1):
			if imageToNoiseRemoving[i][j] == 1:
				# print mMatrix[i][j]+nMatrix[i][j]
				if (mMatrix[i][j]+nMatrix[i][j] > tLong) or (mMatrix[i][j]+nMatrix[i][j] < tShort):
					imageDeNoised[i][j] = 0

	ImageShaped = np.array(imageDeNoised)

	### Sum all rows for a Vertical projection
	verticalSum = np.sum(ImageShaped, axis=1) # axis 1 for vertical axis 0 for horizontal
	maxVertical = np.max(verticalSum)
	normVerticalSum = verticalSum/(maxVertical * 1.00) # normalization


	## Validation of image size for band clipping
	c1 = 10

	## Getting the candidates for bands. For now we should select five candidates
	bandCandidates = []
	chunksNumber = 4
	modulusVerticalSum = verticalSum.shape[0] % chunksNumber
	verticalSumChunks = np.split(verticalSum[modulusVerticalSum:], chunksNumber)

	referenceSize = verticalSumChunks[0].shape[0]

	for i in range(chunksNumber):
		maxVerticalIndex = np.argmax(verticalSumChunks[i])
		lowLimit = i * referenceSize
		upLimit = (i+1) * referenceSize 
		if lowLimit > maxVerticalIndex+lowLimit-c1: 
			lowBound = lowLimit
		else: 
			lowBound = maxVerticalIndex+lowLimit-c1
		if upLimit < maxVerticalIndex+lowLimit+c1:
			upBound = upLimit
		else: 
			upBound = maxVerticalIndex+lowLimit+c1

		bandCandidates.append(image[modulusVerticalSum+lowBound:upBound,:])

	#Plotting all results
	#Plotting
	# plt.subplot(3, 3, 1)
	# plt.imshow(image, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 2)
	# plt.imshow(imageDeNoised, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 3)
	# plt.imshow(imageToNoiseRemoving, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 4)
	# plt.imshow(enchancedImage, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 5)
	# plt.imshow(canny, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 6)
	# plt.imshow(erosion, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 7)
	# plt.imshow(blurEdge, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 8)
	# plt.imshow(preEnchancementImage, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(3, 3, 9)
	# plt.imshow(enchancedImage, cmap = cm.Greys_r)
	# plt.xticks([]), plt.yticks([])
	plt.show()

	#Plot chunks
	for i in range(chunksNumber):	
		plt.subplot(chunksNumber, 2, i+1)
		plt.imshow(bandCandidates[i], cmap = cm.Greys_r)
		plt.xticks([]), plt.yticks([])
	plt.show()

if __name__ == '__main__':
	main()	