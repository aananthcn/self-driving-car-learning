# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import argparse


# Aananth: select a trapezoid of interest
# Concept: apply a trapezoid region-of-interest (ROI) mask to your edge-detected 
# image before running the Hough transform. This ensures cv2.HoughLinesP only 
# detects lines inside the triangle.
def select_region(image, masked_edges, apex_y):
    ysize = image.shape[0]
    xsize = image.shape[1]

    # Use a trapezoid so the top isn't pinched to a single point
    vertices = np.array([[
        (0, ysize),                        # Bottom-left
        (xsize * 0.45, apex_y),            # Top-left
        (xsize * 0.55, apex_y),            # Top-right
        (xsize, ysize)                     # Bottom-right
    ]], dtype=np.int32)

    mask = np.zeros_like(masked_edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(masked_edges, mask)


def draw_extrapolated_lines(line_image, lines, y_min, y_max):
    if lines is None:
        return

    left_x, left_y = [], []
    right_x, right_y = [], []

    for line in lines:
        x1, y1, x2, y2 = line
        if x2 == x1:  # Avoid division by zero on vertical lines
            continue
        
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter out near-horizontal noise (-0.5 < slope < 0.5)
        if slope < -0.5:  # Left lane (y increases downward in image coordinates)
            left_x.extend([x1, x2])
            left_y.extend([y1, y2])
        elif slope > 0.5:  # Right lane
            right_x.extend([x1, x2])
            right_y.extend([y1, y2])

    # Fit a 1st-degree polynomial (y = mx + b) and extrapolate for each side
    for x_pts, y_pts in [(left_x, left_y), (right_x, right_y)]:
        if len(x_pts) > 0:
            m, b = np.polyfit(x_pts, y_pts, 1)
            x_bottom = int((y_max - b) / m)
            x_top = int((y_min - b) / m)
            cv2.line(line_image, (x_bottom, int(y_max)), (x_top, int(y_min)), (255, 0, 0), 10)


def main(args):
	# Read in and grayscale the image
	image = mpimg.imread(args.image)
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

	debug = False

	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	# Define our parameters for Canny and apply
	low_threshold = 50
	high_threshold = 150
	full_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 1
	theta = np.pi/180
	threshold = 50
	min_line_length = 20
	max_line_gap = 50
	line_image = np.copy(image)*0 #creating a blank to draw lines on


	# Aananth: select the region of interest
	ysize = image.shape[0]
	apex_y = int(ysize * 0.66)  # Adjust this single value to raise/lower the top cutoff
	masked_edges = select_region(image, full_edges, apex_y)

	# Run Hough on edge detected image
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
								min_line_length, max_line_gap)

	ysize = image.shape[0]
	draw_extrapolated_lines(line_image, lines, y_min=apex_y, y_max=ysize)

	# Draw the lines on the edge image
	combo = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
	plt.imshow(combo)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Hough Transform Line Detection")
	parser.add_argument("image", nargs="?", default="../../res/exit-ramp.jpg", type=str, help="Path to the input image file")
	args = parser.parse_args()

	main(args)