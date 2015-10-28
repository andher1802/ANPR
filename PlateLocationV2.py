import sys
import string
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import * 
import collections

sys.path.append('../Libraries/libsvm-3.20/python/')

from os import walk
from scipy.ndimage import measurements, morphology
from PIL import Image
from pylab import *
from svmutil import *

def enchancementFunction(image):
	constantA = 2 / (0.15)**2
	constantB = 2 / (0.8 - 0.15)**2
	row = []
	for i in image:
		column = []
		for j in i:
			if j >= 0 and j < 0.15: 
				fg = 3 / ((constantA*(j-0.15)**2)+1)
			elif j >= 0.15 and j < 1: 
				fg = 3 / ((constantB*(j-0.15)**2)+1)
			else: 
				fg = 1
			column.append(fg)
		row.append(column)
	return row

def enchanceImage(image):
	### Filter image for getting rid of noise
	### http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
	blur = cv2.GaussianBlur(image,(15,15),0)
	median = cv2.medianBlur(image,9)
	bilateral = cv2.bilateralFilter(image,9,75,75)
	#Select image to filter
	image2Filter = blur
	sobelyEnchanced = cv2.Sobel(image2Filter,cv2.CV_64F,0,1,ksize=3)
	###compute the new intensity function for image enchancement
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(sobelyEnchanced,kernel,iterations = 1)
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
	return enchancedImage

def denoising(binaryImage):
	#Algorithm for removing background curves and noise
	imageToNoiseRemoving = binaryImage
	mMatrix = np.zeros((imageToNoiseRemoving.shape[0],imageToNoiseRemoving.shape[1]),np.uint32)
	nMatrix = np.zeros((imageToNoiseRemoving.shape[0],imageToNoiseRemoving.shape[1]),np.uint32)
	tLong = 80
	tShort = 25
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
	return ImageShaped

def trainSVM():
	dirpath = '../Images/TrainningSet/'
	filePDDI = []
	
	for dirpath, dirname, filename in walk(dirpath):
		filePDDI.extend(filename)
	trainingSet = []
	trainingClasses = []
	
	for element in filePDDI[:]:
		filename = dirpath+element
		### Reading of the image and transforming it into an array for mathematical operations
		image = np.array(Image.open(filename).convert('L'))
		# image = np.array(Image.open(filename))
		stackedImage = np.hstack(image)
		# print stackedImage
		# cv2.imwrite('./output/imagesUINT8/'+element,image,)
		# cv2.imwrite('./output/TestSet/'+element,image,)
		label = element.split('_')
		# print label[0]
		if label[0] == 'Y':
			numberLabel = 1
		elif label[0] == 'N': 
			numberLabel = -1
		trainingClasses.append(numberLabel)
		trainingSet.append(stackedImage.tolist())

	prob = svm_problem(trainingClasses,trainingSet)
	m = svm_train(prob, '-t 0 -c 1')
	# m = svm_train(prob, '-t 0 -c 1')
	return m

def main():
	# dirpath = '/home/andres/Documents/PythonDevelopment/ComputerVisionCourse/output/TestSetGray/'
	dirpath = '../Images/TestSet/'
	filePDDI = []
	model = trainSVM()

	#Parameters for testing
	step = 5
	score = 0.5
	binarizationTreshold = 0.05

	for dirpath, dirname, filename in walk(dirpath):
		filePDDI.extend(filename)

	for element in filePDDI[:]:
		filename = dirpath+element
		### Reading of the image and transforming it into an array for mathematical operations
		image = np.array(Image.open(filename).convert('L'))

		### Filter image for getting rid of noise
		### http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
		# blur = cv2.GaussianBlur(image,(15,15),0)
		# median = cv2.medianBlur(image,9)
		# bilateral = cv2.bilateralFilter(image,9,75,75)
		# image2Filter = blur

		#Computing enchanced image for edging
		# enchancedImage = image2Filter
		enchancedImage = enchanceImage(image)

		#Computing edges for enchanced image
		# laplacianEnchanced = cv2.Laplacian(enchancedImage,cv2.CV_64F)
		# sobelxEnchanced = cv2.Sobel(enchancedImage,cv2.CV_64F,1,0,ksize=3)
		sobelyEnchanced = cv2.Sobel(enchancedImage,cv2.CV_64F,0,1,ksize=3)
		# cannyEnchanced = cv2.Canny(enchancedImage,10,50)

		#Image for being computed by edge density function
		imageForEdging = sobelyEnchanced
		edges = imageForEdging
		maxEdges = np.max(edges)
		normEdges = abs(edges/(maxEdges * 1.00)) # normalization

		#Dilation
		kernel = np.ones((10,1),np.uint8)
		dilation = cv2.dilate(normEdges,kernel,iterations = 2)

		###compute the new intensity function for image enchancement
		kernel = np.ones((5,10),np.uint8)
		erosion = cv2.erode(dilation,kernel,iterations = 4)

		###compute the new intensity function for image enchancement
		kernel = np.ones((15,2),np.uint8)
		erosion2 = cv2.erode(erosion,kernel,iterations = 2)

		#binarization of NormEdges
		kernel = np.ones((5,20),np.uint8)
		preBinaryImage = cv2.dilate(erosion2,kernel,iterations = 2)
		preBinaryImage = cv2.GaussianBlur(preBinaryImage,(31,31),0)
		# preBinaryImage = cv2.GaussianBlur(erosion2,(31,31),0)
		binaryImage = 1*(preBinaryImage>binarizationTreshold)

		chunksNumber = 4
		modulusImage = binaryImage.shape[0] % chunksNumber

		bandCandidates = []
		bandHorizontalProjection = []
		for element in xrange(chunksNumber):
			currentBand = np.vsplit(binaryImage[modulusImage:], chunksNumber)[element]
			HSum = np.sum(currentBand, axis=0) # axis 1 for vertical axis 0 for horizontal
			bandHorizontalProjection.append(HSum)
			bandCandidates.append(currentBand)
		
		sumBands = 0
		index = 0
		for element in bandHorizontalProjection:
			if np.sum(element) > sumBands:
				indexMax = index
				sumBands = np.sum(element)
				maxVertical = np.max(element)
				if maxVertical != 0:
					normVerticalSum = element/(maxVertical * 1.00) # normalization
				else: 
					normVerticalSum = element
			index += 1

		validIndex = []
		for index in xrange(len(normVerticalSum)):
			if normVerticalSum[index] != 0: 
				validIndex.append(index)

		potentialPlate = np.zeros((binaryImage.shape[0],binaryImage.shape[1]),np.uint32)
		for i in xrange(binaryImage.shape[0]-1):  
			for j in xrange(binaryImage.shape[1]-1):
				if binaryImage[i][j] != 0:
					potentialPlate[i][j] = image[i][j]

		selectecBand = np.vsplit(potentialPlate[modulusImage:], chunksNumber)[indexMax]

		tempStartV = binaryImage.shape[0]-selectecBand.shape[0]
		# print 'shapes',binaryImage.shape, selectecBand.shape, binaryImage[tempStartV:,:].shape
		VSum = np.sum(binaryImage[tempStartV:,:], axis=0) 
		HSum = np.sum(binaryImage[tempStartV:,:], axis=1)
		# print 'lengths H and V',len(HSum), len(VSum)

		tempSizeLeft = 0
		tempSizeRight = 0
		tempSizeTop = 0
		tempSizeBottom = 0

		potentialPlateSizeH = 0
		for element in HSum:
			if element != 0: 
				tempSizeLeft = potentialPlateSizeH
				break
			potentialPlateSizeH += 1
		potentialPlateSizeH = 0
		for element in reversed(HSum):
			if element != 0: 
				tempSizeRight = potentialPlateSizeH
				break
			potentialPlateSizeH += 1
		potentialPlateSizeV = 0
		for element in VSum:
			if element != 0: 
				tempSizeTop = potentialPlateSizeV
				break
			potentialPlateSizeV += 1
		potentialPlateSizeV = 0
		for element in reversed(VSum):
			if element != 0: 
				tempSizeBottom = potentialPlateSizeV
				break
			potentialPlateSizeV += 1

		SizeRight = selectecBand.shape[0]-tempSizeRight
		SizeBottom = selectecBand.shape[1]-tempSizeBottom
		preSelectecBand = selectecBand[tempSizeLeft:SizeRight,tempSizeTop:SizeBottom]
		# print 'start left, start top, end right, end bottom', tempSizeLeft, tempSizeTop, SizeRight, SizeBottom
		# print 'size horizontal, size vertical', SizeRight-tempSizeLeft, SizeBottom-tempSizeTop
		# print 'shape original, shape modified', selectecBand.shape, preSelectecBand.shape 

		selectecBand = preSelectecBand
		modulusV = selectecBand.shape[0] % 30 # shape is given by column, row order 
		modulusH = selectecBand.shape[1] - (selectecBand.shape[1] % 70)
		numberPlatesV = int(selectecBand[modulusV:,xrange(modulusH)].shape[0]/30)
		numberPlatesH = int(selectecBand[modulusV:,xrange(modulusH)].shape[1]/70)

		platerForSVM = []
		potentialPlate = np.zeros((30,70),np.uint32)
		scoreSVM = []
		indexSVMValid = 0		
		for i in xrange(0,selectecBand.shape[0]-30,step):  
			for j in xrange(0,selectecBand.shape[1]-70,10):
				testData = []
				testClass = []
				potentialPlate = selectecBand[i:i+30,j:j+70]
				stackedImage = np.hstack(potentialPlate)
				testData.append(stackedImage.tolist())
				testClass.append(1)
				p_label, p_acc, p_val = svm_predict(testClass,testData, model)
				if p_label[0]==1 and p_val[0][0]>score:
					platerForSVM.append(potentialPlate)
					scoreSVM.append(p_val[0][0])
					print p_val[0][0]

		platerForSVMMaxScore = []
		if len(scoreSVM)!=0:
			platerForSVMMaxScore = [platerForSVM[np.argmax(scoreSVM)]]
		print "---------------------------------------------"
		
		#Plotting all results
		# #Plotting
		# plt.subplot(2, 3, 1)
		# plt.imshow(image, cmap = cm.Greys_r)
		# plt.xticks([]), plt.yticks([])
		# plt.subplot(2, 3, 2)
		# plt.imshow(imageForEdging, cmap = cm.Greys_r)
		# plt.subplot(2, 3, 3)
		# plt.imshow(preBinaryImage, cmap = cm.Greys_r)
		# plt.subplot(2, 3, 4)
		# plt.imshow(binaryImage, cmap = cm.Greys_r)
		# plt.subplot(2, 3, 5)
		# plt.imshow(potentialPlate, cmap = cm.Greys_r)
		# plt.subplot(2, 3, 6)
		# plt.imshow(selectecBand, cmap = cm.Greys_r)
		# plt.show()

		plt.subplot(1, 1, 1)
		plt.imshow(selectecBand, cmap = cm.Greys_r)
		plt.show()

		#Plot Max horizontal projection
		# plt.plot(normVerticalSum)
		# plt.show()
		# Plotting

		i=0
		if len(platerForSVMMaxScore) != 0:
			for i in xrange(len(platerForSVMMaxScore)):	
				plt.subplot(int(len(platerForSVMMaxScore)/5)+int(len(platerForSVMMaxScore)%5), 5, i+1)
				plt.imshow(platerForSVMMaxScore[i], cmap = cm.Greys_r)
				plt.xticks([]), plt.yticks([])
			plt.show()

if __name__ == '__main__':
	main()	