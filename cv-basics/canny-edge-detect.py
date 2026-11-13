import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2  #bringing in OpenCV libraries


# Read the image and convert to grayscale
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion


# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)


# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 90
high_threshold = 100
edges = cv2.Canny(gray, low_threshold, high_threshold)


# Display images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display Original Image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display Grayscale Image
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

# Display Canny Edges
axes[2].imshow(edges, cmap='Greys_r')
axes[2].set_title('Canny Edges')
axes[2].axis('off')

# Render everything at once
plt.tight_layout()
plt.show()