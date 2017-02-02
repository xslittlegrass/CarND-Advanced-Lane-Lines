import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from scipy.signal import find_peaks_cwt, general_gaussian, fftconvolve
from scipy.ndimage.measurements import center_of_mass


def camera_cal_parameters(images,image_size):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size,None,None)

    return mtx, dist

def undistort(image,dist_mtx,dist_param):
    """undistort image using camera caliberation parameters"""
    undis_image = cv2.undistort(image, dist_mtx,dist_param, None, dist_mtx)
    return undis_image


def preprocess(image,mask_vertices,c_thresh,gray_thresh):
    """preprocessing: apply mask, binarization."""
    
    # mash image
    mask = np.uint8(np.zeros_like(image[:,:,0]))
    vertices = mask_vertices
    cv2.fillPoly(mask, vertices, (1))
    
    # binarize in C channel, mainly detect yellow lane lines
    c_channel = np.max(image,axis=2)-np.min(image,axis=2)
    _,c_binary = cv2.threshold(c_channel,c_thresh[0],c_thresh[1],cv2.THRESH_BINARY)

    # binarize in gray channel, mainly detect white lane lines
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    _, gray_binary = cv2.threshold(gray,gray_thresh[0],gray_thresh[1],cv2.THRESH_BINARY)
    
    # combine the results
    combined_binary_masked = cv2.bitwise_and(cv2.bitwise_or(c_binary,gray_binary),mask)
    
    return combined_binary_masked

def perspective_transform(img_binary,src,dst):
    """convert the perspective image to bird view"""
    img_size = img_binary.shape[::-1]
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img_binary, M, img_binary.shape[::-1], flags=cv2.INTER_LINEAR)
    return warped_img


def select_lane_lines(warped_lane,sliding_window = [80,120]):
    """select pixels that are lane lines"""

    def gaussian_smooth(data):
        window = general_gaussian(51, p=0.5, sig=20)
        filtered = fftconvolve(window, data)
        filtered = (np.average(data) / np.average(filtered)) * filtered
        filtered = np.roll(filtered, -25)
        return filtered[0:len(data)]

    
    window_width = sliding_window[0]
    window_height = sliding_window[1]
    
    # histogram of the lower half of the image
    histogram = np.sum(warped_lane[warped_lane.shape[0]//2:,:], axis=0)
    histogram_smoothed = gaussian_smooth(histogram)
    
    # find the peaks from the smoothed curve
    peakidx = find_peaks_cwt(histogram_smoothed, np.arange(90,100))
    
    # if more than two peaks found, report the error
    if len(peakidx) != 2:
        print('peakidx len is {}'.format(peakidx))
    
    # slidding window
    left_windows = []
    right_windows = []
    current_left_window = [peakidx[0]-window_width,peakidx[0]+window_width]
    current_right_window = [peakidx[1]-window_width,peakidx[1]+window_width]

    for i in range(6):
        # update current sliding window according to the center of mass in the current window 
        current_height_window = [720-(i+1)*window_height, 720-i*window_height]
        left_lane_windowed = warped_lane[current_height_window[0]:current_height_window[1], current_left_window[0]:current_left_window[1]]
        right_lane_windowed = warped_lane[current_height_window[0]:current_height_window[1], current_right_window[0]:current_right_window[1]]
        hist_left = np.sum(left_lane_windowed, axis=0)
        hist_right = np.sum(right_lane_windowed, axis=0)
        # if there is no pixels in the current window, skip the updating
        if not all(hist_left==0):
            center_left = int(center_of_mass(hist_left)[0]) + current_left_window[0]
            current_left_window = [center_left-window_width,center_left+window_width]
        if not all(hist_right==0):    
            center_right = int(center_of_mass(hist_right)[0]) + current_right_window[0]
            current_right_window = [center_right-window_width,center_right+window_width]
        # record the current window
        left_windows.append([current_height_window,current_left_window])
        right_windows.append([current_height_window,current_right_window])
    
    left_window_binary = np.zeros_like(warped_lane)
    right_window_binary = np.zeros_like(warped_lane)
    for wd in left_windows:
        xmin = wd[0][0]
        xmax = wd[0][1]
        ymin = wd[1][0]
        ymax = wd[1][1]
        left_window_binary[xmin:xmax,ymin:ymax] = 1
    
    for wd in right_windows:
        xmin = wd[0][0]
        xmax = wd[0][1]
        ymin = wd[1][0]
        ymax = wd[1][1]
        right_window_binary[xmin:xmax,ymin:ymax] = 1
    
    # select only the pixels in the sliding window
    left_lane_binary = np.zeros_like(warped_lane)
    left_lane_binary[(left_window_binary==1) & (warped_lane==1)]=1
    right_lane_binary = np.zeros_like(warped_lane)
    right_lane_binary[(right_window_binary==1) & (warped_lane==1)]=1
    
    return [left_lane_binary,right_lane_binary]

def fit_lane_line(lr_binary_images):
    """fit the left and right lane lines, return the quadratic coefficients"""
    left_lane_binary = lr_binary_images[0]
    right_lane_binary = lr_binary_images[1]
    
    # fit the left and right lane pixels 
    left_Y,left_X = np.where(left_lane_binary==1)
    left_fit = np.polyfit(left_Y, left_X, 2)
    left_fitx = left_fit[0]*left_Y**2 + left_fit[1]*left_Y + left_fit[2]

    right_Y,right_X = np.where(right_lane_binary==1)
    right_fit = np.polyfit(right_Y, right_X, 2)
    right_fitx = right_fit[0]*right_Y**2 + right_fit[1]*right_Y + right_fit[2]
    
    return [left_fit,right_fit]

def draw_lane(image,left_right_fits,src,dst):
    """highlight lane using left and right fitting parameters"""

    Minv = cv2.getPerspectiveTransform(dst,src) # the inverse transform
    
    left_fit = left_right_fits[0]
    right_fit = left_right_fits[1]
    
    # y values
    yval = np.linspace(0,700)

    
    # left and right x values
    left_fitx = left_fit[0]*yval**2 + left_fit[1]*yval + left_fit[2]
    right_fitx = right_fit[0]*yval**2 + right_fit[1]*yval + right_fit[2]
    
    # left, right lane lines according to the fitting
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yval]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yval])))])
    pts = np.hstack((pts_left, pts_right))

    # draw polygon
    color_warp = np.zeros_like(image).astype(np.uint8)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result





def cal_curvature(lr_binary_images):
    
    y_eval = 720
    
    left_lane_binary = lr_binary_images[0]
    right_lane_binary = lr_binary_images[1]
    
    lefty,leftx = np.where(left_lane_binary==1)
    righty,rightx = np.where(right_lane_binary==1)

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    radius = (left_curverad + right_curverad)/2.
    
    # fit polynomials in pixel space
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # left right lane position
    left_lane_pos  = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_lane_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    # diviate from center in meters
    deviation = ((right_lane_pos + left_lane_pos)/2. - 1280/2.)*xm_per_pix
    
    return [radius, deviation]

def print_info(image,radius,deviation):
    img = np.copy(image)
    cv2.putText(img, "curve radius: {} m".format(int(radius)), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,225,225), 2, cv2.LINE_AA)
    cv2.putText(img, "center deviation: {0:.2f} m".format(deviation), (50,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,225,225), 2, cv2.LINE_AA)
    return img


def lane_detection(img,dist_mtx,dist_param,mask_vertices, c_thresh, gray_thresh, src, dst,sliding_window):
    """combined workflow of detecting lane"""

    image = np.copy(img)
    undist_image = undistort(image,dist_mtx=dist_mtx,dist_param=dist_param)
    warped_lane = perspective_transform(preprocess(undist_image,mask_vertices,c_thresh,gray_thresh),src,dst)
    (il,ir)=select_lane_lines(warped_lane,sliding_window)
    image_highlight = draw_lane(undist_image,fit_lane_line((il,ir)),src,dst)
    radius, deviation = cal_curvature([il,ir])
    result = print_info(image_highlight,radius,deviation)

    return result
