import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]

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

 
    # Threshold color channel
    s_binary = np.zeros_like(s_channel, np.uint8)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
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
    return colored_binary, combined_binary
    

if __name__ == "__main__":
    try:
        image = mpimg.imread('test_images/test1.jpg')
        colored_result, combined_result = pipeline(image)

        # Plot the result
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=20)

        ax2.imshow(colored_result)
        ax2.set_title('Pipeline Colored Result', fontsize=20)

        ax3.imshow(combined_result, cmap="gray")
        ax3.set_title('Pipeline Combined Result', fontsize=20)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
        plt.show();
    except:
        print("Unexpected error:", sys.exc_info()[0])