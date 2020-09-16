import cv2
import numpy as np
import matplotlib.pyplot as plt

nx = 9
ny = 6
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

#calibration images
images = []
name = 'calibration'
c=0
for i in range(1,21):
    a = 'camera_cal/'+name+str(i)+'.jpg'
    img = cv2.imread(a)
    images.append(img)

for idx, fname in enumerate(images):
    gray = cv2.cvtColor(fname, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(fname, (nx,ny), corners, ret)

# to undistort image
def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

#changing into bird's eye view
def warp_perspective(test_image,M):
    img_size = (test_image.shape[1], test_image.shape[0])
    warped = cv2.warpPerspective(test_image, M, img_size)
    return warped


# calculating radius of curvature and position
def calc_radius(ploty, left_fitx, right_fitx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    mid_curvature = (left_curverad + right_curverad) / 2  # getting mean radius of curvature
    mid = 640
    left_lane = left_fitx[-1]
    right_lane = right_fitx[-1]
    pos = (mid - (left_lane + right_lane) / 2) * xm_per_pix  # calculating position

    return mid_curvature, pos

# finding the lane and drawing rectangular boxes
def find_path(binary_warped, undistort, Minv):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # print(out_img.shape)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = (binary_warped.shape[0] - (window + 1) * window_height)
        win_y_high = (binary_warped.shape[0] - window * window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 3)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 3)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    """plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)"""
    # call calculate radius of curvature function
    radius, pos = calc_radius(ploty, left_fitx, right_fitx)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistort.shape[1], undistort.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    # plt.imshow(result)

    # print radius of curvature and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result, 'Radius of curvature: ' + str(radius) + 'm', (20, 40), font, 1, (255, 255, 255), 2,
                         cv2.LINE_AA)

    if pos < 0:
        side = 'left'
        pos *= -1

    else:
        side = 'right'

    result = cv2.putText(result, 'Distance of car from center: ' + str(pos) + 'm', (20, 80), font, 1, (255, 255, 255),
                         2, cv2.LINE_AA)

    return result

# applying gradient threshold
def apply_threshold(img, s_thresh=(150, 255), sx_thresh=(30, 100)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel xy
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)  # Take the derivative in xy
    abs_sobelxy = np.absolute(sobelxy)
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # using the s channel for picking lines
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Using the l channel to remove shadows which it used to interpret as lane lines
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > 150)] = 1

    # Threshold xy gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > sx_thresh[0]) & (scaled_sobel < sx_thresh[1])] = 1

    color_binary = np.zeros_like(scaled_sobel)
    color_binary[((s_binary == 1) & (l_binary == 1)) | (sxbinary == 1)] = 1
    return color_binary

def process_image(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    src = np.float32([
        [0.1 * width, height],
        [width - 0.1 * width, height],
        [width / 2 - 0.1 * width / 2, height / 2 + 0.3 * height],
        [width / 2 + 0.1 * width / 2, height / 2 + 0.3 * height],
    ])
    dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    undist = cal_undistort(image, objpoints, imgpoints)
    binary = apply_threshold(undist, (150, 255), (30, 100))
    binary_warped = warp_perspective(binary, M)
    shade_path = find_path(binary_warped, undist, Minv)
    return shade_path