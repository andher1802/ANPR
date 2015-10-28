import cv2
import numpy as np
import sys
from os import walk
from PIL import Image

sys.path.append('../Libraries/libsvm-3.20/python/')
from svmutil import *

def main():
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
		stackedImage = np.hstack(image)
		label = element.split('_')
		if label[0] == 'Y':
			numberLabel = 1
		elif label[0] == 'N': 
			numberLabel = -1
		trainingClasses.append(numberLabel)
		trainingSet.append(stackedImage.tolist())

	prob = svm_problem(trainingClasses,trainingSet)
	m = svm_train(prob, '-t 0 -c 9')

	dirpath = '../Images/TestSetClasiffier/'

	filePDDI = []
	for dirpath, dirname, filename in walk(dirpath):
		filePDDI.extend(filename)

        #Performing a test for the algorithm
	testSet = []
	testClass = []
	testLabel = []
	
	for element in filePDDI[:]:
		filename = dirpath+element
		image = np.array(Image.open(filename).convert('L'))
		stackedImage = np.hstack(image)
		testSet.append(stackedImage.tolist())
		testLabel.append(element)
		label = element.split('_')
		if label[0] == 'N': 
			testClass.append(-1)
		else:
			testClass.append(1)

	p_label, p_acc, p_val = svm_predict(testClass,testSet, m)
	
	resultsCompilation = []
	for index in xrange(len(testClass)):
		resultsCompilation.append([testLabel[index], testClass[index], int(p_label[index])])
		if testClass[index] != int(p_label[index]):
			print testLabel[index]

if __name__ == '__main__':
	main()
