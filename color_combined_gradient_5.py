import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from defined_globals import *
from PerspectiveTransform_2 import * 


def getRegionOfInterest(img):
    print(img.shape)
    global srcPnt1, srcPnt2, srcPnt3, srcPnt4
    roi_corners = np.array([[srcPnt1, srcPnt2, srcPnt3, srcPnt4]])

    mask =  np.zeros(img.shape, dtype=np.uint8)
    channel_count = 2 #img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(img, mask)

    #plt.imshow(masked_image)
    #plt.show()
    return masked_image

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(125, 255), sx_thresh=(50, 100), sy_thread=(80, 100), l_thresh=(50,255)):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]

    # Sobel x    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #l_channel = hls[:,:,1]
    #gray = l_channel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    print("scaled_sobel")
    #print(scaled_sobel)
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    print("sxbinary")
    #print(sxbinary)

    # sobel y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) 
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    # Threshold y gradient
    sybinary = np.zeros_like(scaled_sobel)
    sybinary[(scaled_sobel >= sy_thread[0]) & (scaled_sobel <= sy_thread[1])] = 1

 
    # Threshold color channel
    s_binary = np.zeros_like(s_channel, np.uint8)
    s_pick = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
    l_pick = (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])
    s_binary[ s_pick & l_pick] = 1
    #s_binary[ s_pick] = 1
    print("s_binary")
    #print(s_binary)

    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    colored_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    print("colored_binary")
    #print(colored_binary)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    # Only consider region of interest, same as perspective transform
    combined_binary = getRegionOfInterest(combined_binary)
    return colored_binary, combined_binary
    

if __name__ == "__main__":
    try:
        #start = 532
        #start = 559
        #start = 1250
        #start = 1029
        start = 1250
        number = 50
        for i in range(number):
            imagePath = "images/image_" + str(start + i) + ".jpg"
            image = mpimg.imread(imagePath)
            colored_result, combined_result = pipeline(image)

            # Plot the result
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
            f.tight_layout()

            ax1.imshow(image)
            ax1.set_title(str(start+i), fontsize=20)

            ax2.imshow(colored_result)
            ax2.set_title('Pipeline Colored Result', fontsize=20)

            global srcPnt1, srcPnt2, srcPnt3, srcPnt4
            x = [srcPnt1[0], srcPnt2[0], srcPnt3[0], srcPnt4[0]]
            y = [srcPnt1[1], srcPnt2[1], srcPnt3[1], srcPnt4[1]]    
            ax2.fill(x, y, edgecolor="r", fill=False)


            ax3.imshow(combined_result, cmap="gray")            
            ax3.set_title('Pipeline Combined Result', fontsize=20)

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
            plt.show();
    except:
        print("Unexpected error:", sys.exc_info()[0])