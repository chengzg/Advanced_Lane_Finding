import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import sys 
import os
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
                
                print(index)
                imgPoints.append(corners)
                objPoints.append(objPs)
                print("===========, ", len(objPoints))
            index += 1                
        return imgPoints, objPoints 
    except:
        print("Unexpected error:", sys.exc_info()[0])

initialized = False
mtx = dist = newcameramtx = None
def getUndistortedImg(inputImg):
    global initialized;
    global mtx, dist, newcameramtx;
    if initialized is False:
        imgPoints, objPoints = getAllPoints(imgPaths);
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
        h, w = inputImg.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        initialized = True

    dstImg = cv2.undistort(inputImg, mtx, dist, None, newcameramtx)
    return dstImg

if __name__ == "main":
    imgPaths = readFolderImgs("./camera_cal");
    originalImg = readImg(imgPaths[5])
    displayImg(originalImg);
    gray = processImg(originalImg)
    dstImg = getUndistortedImg(originalImg)

    displayImg(dstImg)


    originalImg = readImg(imgPaths[8])
    displayImg(originalImg);
    gray = processImg(originalImg)
    dstImg = getUndistortedImg(originalImg)

    displayImg(dstImg)