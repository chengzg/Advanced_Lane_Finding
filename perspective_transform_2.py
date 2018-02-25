from defined_globals import *
from camera_calibration_1 import *
from matplotlib.lines import Line2D 


def computerM_Minv():
    global M, Minv
    # it is the sample image
    imgPath = "test_images/straight_lines.jpg"
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
    global M
    img_size = (img.shape[1], img.shape[0])
    if (M is None):
        computerM_Minv()
        
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

if __name__ == "__main__":
    
    imgPath = "test_images/straight_lines.jpg"
    #imgPath = "test_images/test1.jpg"
    #imgPath = "test_images/test2.jpg"
    #imgPath = "test_images/test3.jpg"
    #imgPath = "test_images/test4.jpg"
    #imgPath = "test_images/test5.jpg"
    #imgPath = "test_images/test6.jpg"
    #imgPath = "images/image_533.jpg"
    #imgPath = "images/image_582.jpg"
    #imgPath = "images/image_1.jpg"
    #imgPath = "images/image_619.jpg"
    originalImg = readImg(imgPath)
    displayImg(originalImg);   
    undistortedImg = getUndistortedImg(originalImg)

    display = True;
    if (display):
        plt.imshow(originalImg)
        plt.plot(srcWrapPnt1[0],  srcWrapPnt1[1], ".")
        plt.plot(srcWrapPnt2[0],  srcWrapPnt2[1], ".")
        plt.plot(srcWrapPnt3[0],  srcWrapPnt3[1], ".")
        plt.plot(srcWrapPnt4[0],  srcWrapPnt4[1], ".")
        plt.show()

    
    warped = warp(originalImg)
    
    xs = [300, 300, 900, 900]
    ys = [0, 719, 719, 0]
    line = Line2D(xs, ys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_line(line)
    plt.imshow(warped)
    plt.show()
