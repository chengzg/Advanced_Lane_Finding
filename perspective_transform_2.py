from defined_globals import *
from camera_calibration_1 import *
from matplotlib.lines import Line2D 


def computerM_Minv():
    global M, Minv
    # it is the sample image
    imgPath = "test_images/straight_lines1.jpg"
    img = readImg(imgPath)
    src = np.float32(
            [srcWrapPnt1,
            srcWrapPnt2,
            srcWrapPnt3,
            srcWrapPnt4
            ])
    dest = np.float32(
            [destWrapPnt1,
            destWrapPnt2,
            destWrapPnt3,
            destWrapPnt4 
            ])
        
    M = cv2.getPerspectiveTransform(src, dest)

    Minv = cv2.getPerspectiveTransform(dest, src)

def get_minv():
    if (Minv is None):
        computerM_Minv()
    return Minv

def warp(img):
    img = getUndistortedImg(img)

    global M
    img_size = (img.shape[1], img.shape[0])
    if (M is None):
        computerM_Minv()
        
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

if __name__ == "__main__":
    
    #imgPath = "test_images/straight_lines1.jpg"
    #imgPath = "test_images/test1.jpg"
    #imgPath = "test_images/test2.jpg"
    #imgPath = "test_images/test3.jpg"
    #imgPath = "test_images/test4.jpg"
    #imgPath = "test_images/test5.jpg"
    #imgPath = "test_images/test6.jpg"
    #imgPath = "images/image_533.jpg"
    imgPath = "images/image_582.jpg"
    originalImg = readImg(imgPath)
    #displayImg(originalImg);
    #gray = processImg(originalImg)
    #undistortedImg = getUndistortedImg(originalImg)

    display = False;
    if (display):
        plt.imshow(originalImg)
        plt.plot(579,  457, ".")
        plt.plot(702,  457, ".")
        plt.plot(1121, 719, ".")
        plt.plot(188,  719, ".")
        plt.show()

    warped = warp(originalImg)
    
    xs = [300, 300, 900, 900]
    ys = [0, 719, 719, 0]
    line = Line2D(xs, ys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_line(line)
    #line1 = plt.plot(300, 0, 300, 719, ls="-", label='line 1', color="r", linewidth=2)
    #line2 = plt.plot(900, 0, 900, 719, ls="-", label='line 2', color="r", linewidth=2)
    #plt.setp(line1)
    plt.imshow(warped)
    plt.show()