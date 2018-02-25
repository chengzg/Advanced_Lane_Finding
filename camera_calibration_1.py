import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import sys 
import os
from defined_globals import *
#%matplotlib inline

def processImg(img):
    img_shape=img.shape
    if len(img_shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img

def displayImg(img, isGrayscale=False):
    if isGrayscale is True:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.show()

def readImg(imgPath):
    img = mpimg.imread(imgPath)

    return img    

def readFolderImgs(folderPath="."):
    jpgImgs = []
    folderPath.replace("//", "\\")
    folderPath = os.path.normpath(folderPath)
    for file in os.listdir(folderPath):
        if file.endswith(".jpg"):
            jpgImgs.append(os.path.join(folderPath, file))

    return jpgImgs
 
def getAllPoints(imgPaths):
    
    # 3D real obj points
    objPoints = []
    # 2D points in image
    imgPoints = []

    chessWidth = 9
    chessHeight = 6;
    imgDimension = (chessWidth, chessHeight)

    objPs = np.zeros((chessWidth*chessHeight, 3), np.float32)
    objPs[:, :2] = np.mgrid[0:chessWidth, 0:chessHeight].T.reshape(-1, 2)
    #print(objPs)

    try:    
        index = 0;    
        for path in imgPaths:
            originalImg = readImg(path)
            gray = processImg(originalImg)
            #displayImg(img, True)
            ret, corners = cv2.findChessboardCorners(gray, imgDimension, None)
            
            if (ret is True):
                
                print("find board corner for", index, path)

                if (False):  
                    cv2.drawChessboardCorners(originalImg, imgDimension, corners, ret)
                    plt.imshow(originalImg)
                    plt.show()


                imgPoints.append(corners)
                objPoints.append(objPs)
            else:
                print("fail to find board corner for img ", index, path)
            index += 1                
        return imgPoints, objPoints 
    except:
        print("Unexpected error:", sys.exc_info()[0])

def getUndistortedImg(inputImg, imgPaths=None):
    global mtx, dist;

    if mtx is None:
        if (imgPaths is None):
            imgPaths = readFolderImgs("./camera_cal")
        gray = processImg(inputImg)
        print(gray.shape[::-1])
        imgPoints, objPoints = getAllPoints(imgPaths);
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

    dstImg = cv2.undistort(inputImg, mtx, dist, None, mtx)    
    return dstImg

if __name__ == "__main__":
    try:
        imgPaths = readFolderImgs("./camera_cal");
        imgPath = imgPaths[13]
        #imgPath = "test_images/straight_lines.jpg"
        #imgPath = "images/image_582.jpg"
        originalImg = readImg(imgPath)
        displayImg(originalImg);
        gray = processImg(originalImg)
        dstImg = getUndistortedImg(originalImg)

        displayImg(dstImg)

        originalImg = readImg(imgPaths[8])
        displayImg(originalImg);
        gray = processImg(originalImg)
        dstImg = getUndistortedImg(originalImg)

        displayImg(dstImg)
    except:
        print("Unexpected error:", sys.exc_info()[0])