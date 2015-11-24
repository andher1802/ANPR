import numpy as np
import cv2

def orderPoints(pts):
    rect = np.zeros((4,2), dtype = 'float32')
    #print rect 
    s = np.sum(pts, axis = 1)
    #print s
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    #print diff
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]
    #print pts
    #print rect
    return rect
    #print 'finished order points'

def fourPointsTransform(image, pts):
    #Order points
    print pts
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
    return 0

if __name__=='__main__':
    main()
