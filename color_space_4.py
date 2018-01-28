import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def hls_select(img, thresh=(0, 255), chan=2):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    channel = hls[:,:,2]
    if chan == 0:
        channel = hls[:,:,0]
    elif chan == 1:
        channel = hls[:,:,1]
    
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

image = mpimg.imread('test_images/test6.jpg')
s_binary = hls_select(image, thresh=(90, 255), chan=2)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(s_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
plt.show()