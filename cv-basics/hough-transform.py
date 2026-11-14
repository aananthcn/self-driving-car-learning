# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import argparse


# Aananth: select the triangle of interest
# Concept: apply a triangular region-of-interest (ROI) mask to your edge-detected 
# image before running the Hough transform. This ensures cv2.HoughLinesP only 
# detects lines inside the triangle.
def select_region(image, masked_edges):
	ysize = image.shape[0]
	xsize = image.shape[1]

	left_bottom = [0, ysize]
	right_bottom = [xsize, ysize]
	apex = [xsize/2, 4*ysize/7] 

	# Define the polygon vertices for the triangle
	vertices = np.array([[left_bottom, apex, right_bottom]], dtype=np.int32)

	# Create a blank mask and fill the triangle with white (255)
	mask = np.zeros_like(masked_edges)
	cv2.fillPoly(mask, vertices, 255)

	# Keep only the edges inside the triangle
	masked_edges = cv2.bitwise_and(masked_edges, mask)

	return masked_edges


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
	min_line_length = 10
	max_line_gap = 1
	line_image = np.copy(image)*0 #creating a blank to draw lines on


	# Aananth: select the region of interest
	masked_edges = select_region(image, full_edges)

	# Run Hough on edge detected image
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
								min_line_length, max_line_gap)

	# Aananth: debug the error
	if debug:
		print("threshold: ", threshold)
		print("lines: ", lines)

	# Iterate over the output "lines" and draw lines on the blank
	if lines is not None:
		for line in lines:
			# Aananth: These lines are source of error
			# for x1,y1,x2,y2 in line:
			# 	cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
			x1, y1, x2, y2 = line
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

	# Create a "color" binary image to combine with line image
	# color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
	color_edges = np.dstack((full_edges, full_edges, full_edges))

	# Draw the lines on the edge image
	combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
	plt.imshow(combo)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Hough Transform Line Detection")
	parser.add_argument("image", nargs="?", default="../res/exit-ramp.jpg", type=str, help="Path to the input image file")
	args = parser.parse_args()

	main(args)