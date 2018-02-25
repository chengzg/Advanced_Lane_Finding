## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration_3_original.png "original distorted image"
[image2]: ./output_images/camera_calibration_3_undistorted.png "undistorted image"
[image3]: ./output_images/perspective_transform_image_test_original.png "original distorted test image"
[image4]: ./output_images/perspective_transform_image_test_undistorted.png "original undistorted test image"
[image5]: ./output_images/perspective_transform_image_test_transformed.png "perspective transform test image"
[image6]: ./output_images/pipeline_combined_image.png "perspective transform image"
[image7]: ./output_images/perspective_transform_image_582_original.png "original distorted actual image"
[image8]: ./output_images/perspective_transform_image_582_undistorted.png "original undistorted actual image"
[image9]: ./output_images/perspective_transform_image_582_transformed.png "perspective transform actual image"
[image10]: ./output_images/region_of_interested.png "region of interested"
[image11]: ./output_images/filtered_image.png "filter lane lines image"
[image12]: ./output_images/warped_lane_image.png "warped lane lines image"
[image13]: ./output_images/pixel_histogram.png "pixel histogram"
[image14]: ./output_images/fit_in_polynomial_image.png "fit in polynomial image"
[image15]: ./output_images/filled_polygon_image.png "fit the polyline rectangle image"
[image16]: ./output_images/transformed_back_polygon_image.png "transform polyline rectangle back to image space"
[image17]: ./output_images/map_detection_to_original_image.png "map detected region back to undistorted image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained Python file located in line "./camera_calibration_1.py".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objPs` is just a replicated array of coordinates, and `objPoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgPoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objPoints` and `imgPoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Provide an example of a perspective transfromation image
To demonstrate this step, here is the orignal straight line image i used

![alt text][image3]

Firstly i compute the undistorted image based on the calibration matrix computed above, i got the below image with manual defined 4 src points

![alt text][image4]

After apply the perspective transform, i got the below image:

![alt text][image5]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in `color_combined_gradient_5.py`.  Here's an example of my output for this step. It is one frame captured from the test project video.

![alt text][image6]

For the color threshold, i used 2 channels from the HLS space, the s_channel and the l_channel. Initially only the s_channel is used with the gradient to pick up the lines. But it does not work well when the image has shadows across the lanes. So the l channel is introduced to reduce the wrong detection of the pixels. I then tweak the thresholds, s_thresh and l_thresh to get better identification of the lanes.  Note that the connected red lines are the 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp() and computerM_Minv()`, which appears in the file `perspective_transform_2.py`.  The `warper()` function takes as inputs an image (`img`). Internally it will use the harded code srcWrapPnt1/2/3/4 points as source(`src`) and the destWrapPnt1/2/3/4 points as destination (`dst`) points.  Based on the chosen points, i will use the `cv2.getPerspectiveTransform(src, dest)` and `cv2.getPerspectiveTransform(dest, src)` to compute the transform and inverse transformtion matrix. And it will only be computed once.

The hard coded source and destination points are:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 579, 457      | 300, 0        | 
| 732, 457      | 900, 0        |
| 1207, 719     | 900, 719      |
| 195, 719      | 300, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]
![alt text][image8]
![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Below is the sample image. The red polyline are the region of interest that i will use it to detect the lanes

![alt text][image10]

Based on the previous described pipeline, i get the filter lane pixels as shown below:

![alt text][image11]

Then i use the perspective transformation matrix computed earlier to transfer the lanes to birds eye view: 

![alt text][image12]

From the above image, i compute the histogram to identify the lane pixles, as shown below:

![alt text][image13]

From the identified pixels, i fit my lane lines with a 2nd order polynomial as shown below:
Here i follow the course material by implement sliding windows and fit a 2nd order polynomial to find the lanes.
I did this in function `get_left_right_lane_fit` and `get_left_right_lane_pixel_polynomial_fit` in my code in `find_lines_6.py`

![alt text][image14]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Basically to compute the curvature, i follow what is described in the course material. Firstly i compute the pixel curvature as shown in function `compute_pixel_curvature` in find_lines_6.py. Then i compute the real world curvature in `compute_world_curvature`
Before that i get the world polynomial fit in function `get_left_right_lane_world_polynomial_fit` 
 
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `mapto_original_image` in my code in `find_lines_6.py`.  
Based on the polylines found, i fill the region bounded by the polylines with green colors, shown below:

![alt text][image15]

After that, the polygon is transfered back to perspetive view shown below:

![alt text][image16]

Finally it is added to the undistorted original image. 

![alt text][image17]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The current approach very much depends on how you correctly identify the lane pixels. Here i use both gradient and color threshold. And for color identification, i used the s_channel to help identify both yellow and white lanes. 
I also used the l_channel to reduce the noise introduced by the shadows. However, it does not work perfectly fine. It very much depends on the tweaked parameters to do correct identification. In some scenarios the parameter might 
need to be more carefully adjusted. 

Another problem for current approach is that the middle lane is dashed lines, near the car the lane marking is constantly missing. It can make the polynomial fit vary very much from its previous frame. The current approach is to
compare the newly computed curvature with the historical ones. If it varies too much, it will use the average of the last 10 frames. It works but not perfectly. One possible solution is that we can acutally manually add the missing
lane marking for each frame since we know where it should be with respect to the car size and car center. In this case we will have a better polynomial fit. I believe it will give a better result.  
