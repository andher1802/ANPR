import cv2
import numpy as np

from PIL import Image
from os import walk
from pytesseract import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

def main():
    dirpath = '../Images/OCRSet/'
    # dirpath = '../Images/TestSetSegmentationOut/'
    filePDDI = []

    for dirpath, dirname, filename in walk(dirpath):
        filePDDI.extend(filename)

    listImages = []

    for element in filePDDI[:]:
        filename = dirpath+element
        im = np.array(Image.open(filename).convert('L'))
        im = cv2.resize(im, (0,0), fx=4, fy=4)
        maxEdges = np.max(im)
        normEdges = abs(im/(maxEdges * 1.00)) # normalization
        kernel = np.ones((1,1),np.uint8)
        binaryImage = cv2.erode(normEdges,kernel,iterations = 1)
        kernel = np.ones((1,1),np.uint8)
        binaryImage = cv2.dilate(normEdges,kernel,iterations = 1)
        # binaryImage = im
        binarizationTreshold = 0.55
        binaryImage = 1*(binaryImage>binarizationTreshold)
        # print binaryImage
        binaryImage = np.array(binaryImage, dtype=np.uint8)
        imRestored = Image.fromarray(binaryImage)

        print image_to_string(imRestored,config='-psm 10', lang='spa'), element #SingleWord
        # print image_to_string(imRestored,config='-psm 7'), element #SingleLine
        listImages.append(imRestored)
    
    i = 0
    for images in listImages:
        i += 1
        j = int(len(listImages)/2)
        plt.subplot(j+1, 2, i)
        plt.imshow(images, cmap = cm.Greys_r)
        plt.xticks([]), plt.yticks([])
       
    plt.show()


if __name__ == '__main__':
	main()