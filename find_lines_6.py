from PerspectiveTransform_2 import *
from color_combined_gradient_5 import *

# find the histogram
def get_histogram(img):    
    histogram = np.sum(img[img.shape[0]//2:, :], axis = 0)
    global dispaly    
    if (display):
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
def show_selection_windows(warped, left_fit, right_fit, left_lane_inds, right_lane_inds):
    
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
start = 535
increment = 0
def visualize_polynomial_fit(warped, left_fit, right_fit, left_lane_inds, right_lane_inds):

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
    global increment, start
    imagePath = "images_2/image_" + str(start + increment) + ".jpg"
    increment += 1
    plt.savefig(imagePath)
    plt.show()
    

def update_region_of_interested(left_up, right_up, right_down, left_down, minv):
    
    src = np.float32([left_up, right_up, right_down, left_down])
    ones = np.ones(shape=(len(src), 1))
    src_ones = np.hstack([src, ones])
    #print(src_ones.shape)    
    #print(minv.shape)
    #dest = cv2.transform(src, minv)
    dest = minv.dot(src_ones.T).T
    #print(src.shape)
    #print(dest.shape)
    global srcPnt1, srcPnt2, srcPnt3, srcPnt4
    srcPnt1 = dest_left_up = dest[0][0:2]/dest[0][2]
    srcPnt1[0] -= 15
    srcPnt2 = dest_right_up = dest[1][0:2]/dest[1][2]
    srcPnt2[0] += 15
    srcPnt3 = dest_right_down = dest[2][0:2]/dest[2][2]
    srcPnt3[0] += 50
    srcPnt4 = dest_left_down = dest[3][0:2]/dest[3][2]
    srcPnt4[0] -= 50
    
    #print(srcPnt1, srcPnt2, srcPnt3, srcPnt4)


def show_region_of_interests(image):
    global srcPnt1, srcPnt2, srcPnt3, srcPnt4
    x = [srcPnt1[0], srcPnt2[0], srcPnt3[0], srcPnt4[0]]
    y = [srcPnt1[1], srcPnt2[1], srcPnt3[1], srcPnt4[1]]    
    plt.fill(x, y, edgecolor="r", fill=False)

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

    left_up = pts_left[0][0]
    right_up = pts_right[0][-1]
    right_low = pts_right[0][0]
    left_low = pts_left[0][-1]
    #print(left_up, right_up, right_low, left_low)

    

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    global display
    if (display):
        plt.imshow(color_warp)
        plt.show()

    Minv = get_minv();
    update_region_of_interested(left_up, right_up, right_low, left_low, Minv)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    if (display):
        plt.imshow(newwarp)
        plt.show()

    # Combine the result with the original image.
    print(image.shape)
    print(newwarp.shape)
    result = cv2.addWeighted(image[:,:,:3], 1, newwarp, 0.3, 0)

    return result


class Line():
    def __init__(self):
        self.detected = False
        self.curvature = 0
        self.line_fit = None
        self.frame_index = 0
        # the current frame average x pixel value
        self.x_value = None


left_line_history = []
right_line_history = []
current_frame_index = 0

# Check whether the current detected left/right line is correct one
def pass_sanity_check(left_line, right_line):
    global left_line_history, right_line_history

    if (len(left_line_history) == 0):
        return True
        
    left_curvature_sum = 0;
    left_x_sum = 0 
    for line in left_line_history:
        left_curvature_sum += line.curvature
        left_x_sum += line.x_value
    left_curvature_average = left_curvature_sum / len(left_line_history)
    left_line_x_average = left_x_sum / len(left_line_history)

    right_curvature_sum = 0;
    right_x_sum = 0
    for line in right_line_history:
        right_curvature_sum += line.curvature
        right_x_sum += line.x_value
    right_curvature_average = right_curvature_sum / len(right_line_history)
    right_line_x_average = right_x_sum / len(right_line_history)
                                                            
    con1 = abs(left_curvature_average - left_line.curvature) >  left_curvature_average
    con2 = abs(right_curvature_average - right_line.curvature) > right_curvature_average
    con3 = abs(left_line_x_average - left_line.x_value) > left_line_x_average
    con4 = abs(right_line_x_average - right_line.x_value) > right_line_x_average
    print(con1, con2, con3, con4)                                      
    if (con1 or con2 or con3 or con4):
        return True;        #return False
    return True;

def exportimage(image):
    print(image.shape)
    global current_frame_index;
    current_frame_index += 1
    imgPath = "images_1/image_" + str(current_frame_index) + ".jpg"
    mpimg.imsave(imgPath, image)
    newImage = image.copy()
    return newImage

def map_detected_region_to_image(image):
    print(image.shape)
    global left_line_history, right_line_history, current_frame_index
    current_frame_index += 1

    colored_result, combined_result = pipeline(image)
    global display;
    if (display):
        plt.imshow(image)
        show_region_of_interests(image)
        plt.show()

    if (display):
        plt.imshow(combined_result)
        
        plt.show()
    warped = warp(combined_result)

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    needToRestartSlidingWindowFinding = False
    if len(left_line_history) == 0 or (current_frame_index - left_line_history[-1].frame_index) > 5:
        needToRestartSlidingWindowFinding = True;
    
    left_line = Line();
    right_line = Line();

    needToRestartSlidingWindowFinding = True;

    if (needToRestartSlidingWindowFinding is True):
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

#        out_img = np.dstack((warped, warped, warped))*255
#        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#        plt.imshow(out_img)
#        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
#        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#        plt.plot(left_fitx, ploty, color='yellow')
#        plt.plot(right_fitx, ploty, color='yellow')
#        plt.xlim(0, 1280)
#        plt.ylim(720, 0)
#        plt.show()


        # fill the left Line data
        left_line.curvature = left_fit_cr_world
        left_line.frame_index = current_frame_index
        left_line.line_fit = left_fit
        left_line.x_value = np.sum(leftx)/len(leftx)

        # fill the right Line data
        right_line.curvature = right_fit_cr_world
        right_line.frame_index = current_frame_index
        right_line.line_fit = right_fit
        right_line.x_value = np.sum(rightx)/len(rightx)


    else:
        # use the previous found correct data to speed computation
        margin_offset = 100       
        # old left_fit, right_fit 
        left_fit = left_line_history[-1].line_fit
        right_fit = right_line_history[-1].line_fit

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin_offset)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin_offset))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin_offset)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin_offset)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_fit, right_fit = get_left_right_lane_pixel_polynomial_fit(leftx, lefty, rightx, righty)
        left_fit_cr_pixel, right_fit_cr_pixel = compute_pixel_curvature(image.shape[0], left_fit, right_fit)
        left_fit_cr_world, right_fit_cr_world = get_left_right_lane_world_polynomial_fit(leftx, lefty, rightx, righty)
        left_fit_cr_world, right_fit_cr_world = compute_world_curvature(image.shape[0], left_fit_cr_world, right_fit_cr_world)



        # fill the left Line data
        left_line.curvature = left_fit_cr_world
        left_line.frame_index = current_frame_index
        left_line.line_fit = left_fit
        left_line.x_value = sum(leftx)/len(leftx)

        # fill the right Line data
        right_line.curvature = right_fit_cr_world
        right_line.frame_index = current_frame_index
        right_line.line_fit = right_fit
        right_line.x_value = sum(rightx)/len(rightx)

    # check whether it passes the sanity check
    if (pass_sanity_check(left_line, right_line)):
        left_line_history.append(left_line)
        right_line_history.append(right_line)
        if (len(left_line_history) > 10):
            left_line_history.pop(0)
            right_line_history.pop(0)        
    else:
        # use the last correct frame data
        left_fit = left_line_history[-1].line_fit
        right_fit = right_line_history[-1].line_fit

    global display;
    if (display):
        show_selection_windows(warped, left_fit, right_fit, left_lane_inds, right_lane_inds)    
        visualize_polynomial_fit(warped, left_fit, right_fit, left_lane_inds, right_lane_inds)

    
    result = mapto_original_image(image, warped, left_fit, right_fit)
    
    if (display):
        plt.imshow(result)
        show_region_of_interests(result)
        plt.show()
    
    export_image = False
    if (export_image):
        imgPath = "images_1/image_" + str(current_frame_index) + ".jpg"
        mpimg.imsave(imgPath, result)        
    return result


import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
def testVideo():
    inputVideo = 'project_video.mp4'
    #inputVideo = 'project_video_output_1.mp4'
    outputVideo = 'project_video_output.mp4'

    clip1 = VideoFileClip(inputVideo)
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    output_clip = clip1.fl_image(map_detected_region_to_image) #NOTE: this function expects color images!!
    #output_clip = clip1.fl_image(exportimage)
    output_clip.write_videofile(outputVideo, audio=False)

display = False
if __name__ == "__main__":

    #image = mpimg.imread('test_images/test6.jpg')
    #image = mpimg.imread('test_images/straight_lines2.jpg')
    #image = mpimg.imread('test_images/signs_vehicles_xygrad.jpg')
    #result = map_detected_region_to_image(image)
    #plt.imshow(result)
    #plt.show()
    try:
        #image = mpimg.imread('test_images/test6.jpg')
        #image = mpimg.imread('test_images/straight_lines2.jpg')
        #image = mpimg.imread('test_images/signs_vehicles_xygrad.jpg')
        
        if (display):
            imageName = "images/image_"      
            global start
            #start = 579             
            #start = 1027
            #start = 1250
            start = 1
            for i in range(50):
                imagePath = imageName + str(start + i) + ".jpg"
                image = mpimg.imread(imagePath)        
                result = map_detected_region_to_image(image)
                print("image path is: ", imagePath)
                plt.imshow(result)
                plt.show()
        else:
            testVideo()
        #image = mpimg.imread('images/image_532.jpg')        
        #result = map_detected_region_to_image(image)
        #plt.imshow(result)
        #plt.show()

        
        print("--------------------------------")
    except:
        print("Unexpected error:", sys.exc_info()[0])