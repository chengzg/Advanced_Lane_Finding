import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import sys 
import os
from defined_globals import *
#%matplotlib inline

def processImg(img):
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
    global mtx, dist, newcameramtx;
    if newcameramtx is None:
        if (imgPaths is None):
            imgPaths = readFolderImgs("./camera_cal")
        gray = processImg(inputImg)
        imgPoints, objPoints = getAllPoints(imgPaths);
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
        h, w = inputImg.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dstImg = cv2.undistort(inputImg, mtx, dist, None, newcameramtx)
    return dstImg

if __name__ == "__main__":
    try:
        imgPaths = readFolderImgs("./camera_cal");
        originalImg = readImg(imgPaths[13])
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