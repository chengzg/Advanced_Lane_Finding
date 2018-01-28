import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import sys 
import os

def abs_sobel_thresh(img, orient="x", sobel_kernel=3, thresh=(0, 255)):    
    gray = img;
    if (len(img.shape) > 2):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelValue = None
    abs_sobelValue = None
    if orient=="x":
        sobelValue = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobelValue = np.absolute(sobelValue)
    else:
        sobelValue = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelValue = np.absolute(sobelValue)

    scaled_sobel = np.uint8(255 * abs_sobelValue / np.max(abs_sobelValue))

    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sobel_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0,255)):
    
    gray = img;
    if (len(img.shape) > 2):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel);
    abs_sobelValue = np.sqrt(sobelx * sobelx + sobely * sobely)    

    scaled_sobel = np.uint8(255 * abs_sobelValue / np.max(abs_sobelValue))

    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sobel_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    
    gray = img;
    if (len(img.shape) > 2):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelValue = None
    abs_sobelValue = None
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel);
    abs_sobely = np.absolute(sobely)

    sobelGradient = np.arctan2(abs_sobely, abs_sobelx)    
    binary_output =  np.zeros_like(sobelGradient)
    binary_output[(sobelGradient >= thresh[0]) & (sobelGradient <= thresh[1])] = 1

    return binary_output

def combined_threshold(img, kernel_size=3):
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply each of the thresholding functions

    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined
#imgPath = "test_images/straight_lines1.jpg"
imgPath = "test_images/signs_vehicles_xygrad.png"

img = mpimg.imread(imgPath)

#sxbinary = abs_sobel_thresh(img, "x", 30, 100)
#plt.imshow(sxbinary, cmap='gray')
#plt.show()

#sybinary = abs_sobel_thresh(img, "y", 50, 200)
#plt.imshow(sybinary, cmap="gray");
#plt.show()

#sbinary = abs_sobel_thresh(img, "xy", 9, 30, 100)
#plt.imshow(sbinary, cmap="gray");
#plt.show()

#sbinary = dir_threshold(img, 15, thresh=(0.7, 1.3))
#plt.imshow(sbinary, cmap="gray");
#plt.show()

sbinary = combined_threshold(img, 15)
plt.imshow(sbinary, cmap="gray");
plt.show()