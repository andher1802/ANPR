import string
from PIL import Image
from pylab import *
import cv2
import numpy as np
from CVTools.toolsPerspective import fourPointsTransform
#from skimage.filter import threshold_adaptive

def turboScan():
    #edge detection
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    image = np.array(Image.open('./images/example_01.png').convert('L'))
 
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
 
    # show the original image and the edge detected image
    print "STEP 1: Edge Detection"
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    
    #use edge into an image to find the document
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
    # loop over the contours
    for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
    

    # show the contour (outline) of the piece of paper
    print "STEP 2: Find contours of paper"
    bck2RGB = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bck2RGB, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Contourned", bck2RGB)
    #apply perspective to find the top-down document
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = fourPointsTransform(bck2RGB, screenCnt.reshape(4,2))
 
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    #warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #warped = threshold_adaptive(warped, 250, offset = 10)
    #warped = warped.astype("uint8") * 255
 
    # show the original and scanned images
    print "STEP 3: Apply perspective transform"
    cv2.imshow("Scanned", warped)
    bwImage = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #bwImage = threshold_adaptive(warped, 250, offset=10)
    cv2.imshow("ScannedWB", bwImage)
    cv2.waitKey(0)

    print 'finished' 

def main():
    turboScan()

if __name__=='__main__':
    main()
