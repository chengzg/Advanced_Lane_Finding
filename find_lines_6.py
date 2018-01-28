from PerspectiveTransform_2 import *
from color_combined_gradient_5 import *

# find the histogram
def get_histogram(img):    
    histogram = np.sum(img[img.shape[0]//2:, :], axis = 0)
    if (False):
        plt.plot(histogram)
        plt.show()
    
    return histogram

def get_left_right_x_base_from_histogram(histogram, left_margin=150, right_margin=1200-150):

    #left_margin = 150
    #right_margin = 1200 - 150
    midpoint = np.int(histogram.shape[0]/2)

    leftx_base = np.argmax(histogram[left_margin:midpoint]) + left_margin
    rightx_base = np.argmax(histogram[midpoint:right_margin]) + midpoint

    return leftx_base, rightx_base


margin = 100
minpix = 50
def get_left_right_lane_fit(warped, leftx_base, rightx_base, nwindows=9, margin=100, minpix=50):
    
    #nwindows = 9
    window_height = np.int(warped.shape[0]/nwindows)

    leftx_current = leftx_base
    rightx_current = rightx_base
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img = np.dstack((warped, warped, warped)) * 255
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
        (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
        (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)


    return left_lane_inds, right_lane_inds



def get_left_right_lane_pixel_polynomial_fit(leftx, lefty, rightx, righty):

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def compute_pixel_curvature(y_eval, left_fit, right_fit):
    #y_eval = image.shape[0]
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    return left_curverad, right_curverad


# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension
def get_left_right_lane_world_polynomial_fit(leftx, lefty, rightx, righty):

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    return left_fit_cr, right_fit_cr



#compute the curvature
def compute_world_curvature(y_eval, left_fit_cr, right_fit_cr):

    # Calculate the new radii of curvature
    # y_eval = image.shape[0]
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

    return left_curverad, right_curverad


# Create an image to draw on and an image to show the selection window
def show_selection_windows(warped, left_fit, right_fit):
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img = np.dstack((warped, warped, warped))*255    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    left_fitx = left_fit[0]* ploty ** 2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty ** 2 + right_fit[1]*ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


# Visualization
def visualize_polynomial_fit(warped, left_fit, right_fit):
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img = np.dstack((warped, warped, warped))*255
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]* ploty ** 2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty ** 2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def mapto_original_image(image, warped, left_fit, right_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]* ploty ** 2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty ** 2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Minv = get_minv();
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

def map_detected_region_to_image(image):
    colored_result, combined_result = pipeline(image)
    warped = warp(combined_result)

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = get_histogram(warped);
    leftx_base, rightx_base = get_left_right_x_base_from_histogram(histogram)
    left_lane_inds, right_lane_inds= get_left_right_lane_fit(warped, leftx_base, rightx_base)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = get_left_right_lane_pixel_polynomial_fit(leftx, lefty, rightx, righty)
    left_fit_cr_pixel, right_fit_cr_pixel = compute_pixel_curvature(image.shape[0], left_fit, right_fit)
    left_fit_cr_world, right_fit_cr_world = get_left_right_lane_world_polynomial_fit(leftx, lefty, rightx, righty)
    left_fit_cr_world, right_fit_cr_world = compute_world_curvature(image.shape[0], left_fit_cr_world, right_fit_cr_world)

    #show_selection_windows(warped, left_fit, right_fit)    
    #visualize_polynomial_fit(warped, left_fit, right_fit)
    
    result = mapto_original_image(image, warped, left_fit, right_fit)
    return result

import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
def testVideo():
    # Import everything needed to edit/save/watch video clips
    inputVideo = 'project_video.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    outputVideo = 'project_video_output.mp4'
    clip1 = VideoFileClip(inputVideo)
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(map_detected_region_to_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(outputVideo, audio=False)

if __name__ == "__main__":

    #image = mpimg.imread('test_images/test6.jpg')
    #image = mpimg.imread('test_images/straight_lines2.jpg')
    #image = mpimg.imread('test_images/signs_vehicles_xygrad.jpg')
    #result = map_detected_region_to_image(image)
    #plt.imshow(result)
    #plt.show()
    try:
        testVideo()
        print("--------------------------------")
    except:
        print("Unexpected error:", sys.exc_info()[0])