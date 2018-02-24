
# for camera calibration
mtx = dist = newcameramtx = None

# define the constants to do the transform
srcWrapPnt1  = [559,  457] #left_up
srcWrapPnt2  = [732,  457] #right_up
srcWrapPnt3  = [1191, 719] #right_down
srcWrapPnt4  = [138,  719] #right_up

destWrapPnt1 = [300,  0]
destWrapPnt2 = [900,  0]
destWrapPnt3 = [900,  719]
destWrapPnt4 = [300,  719]

#define the initial value to do region of interest
srcPnt1  = [559,  457]
srcPnt2  = [722,  457]
srcPnt3  = [1251, 719]
srcPnt4  = [188,  719]

#define the camera transformation matrix
M = None
Minv = None

#define the frame data
left_line_history = []
right_line_history = []
current_frame_index = 0

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension

margin = 100
minpix = 50

# Define the testing data constant
testing_start = 760
testing_increment = 0

