import argparse
import cv2
import numpy as np


class LaneTracker:
  def __init__(self, buffer_size=10):
    self.buffer_size = buffer_size
    self.left_fits = []
    self.right_fits = []

  def get_smoothed_fit(self, new_fit, fit_list):
    if new_fit is not None:
      fit_list.append(new_fit)
      if len(fit_list) > self.buffer_size:
        fit_list.pop(0)  # Remove oldest frame
    # Return average of stored fits, or fallback to latest if available
    if len(fit_list) > 0:
      return np.mean(fit_list, axis=0)
    return None


# Instantiate a global tracker outside process_frame
tracker = LaneTracker(buffer_size=12)


def select_region(image, masked_edges, apex_y):
    ysize = image.shape[0]
    xsize = image.shape[1]

    # Use a trapezoid region of interest
    vertices = np.array(
        [[
            (0, ysize),  # Bottom-left
            (xsize * 0.45, apex_y),  # Top-left
            (xsize * 0.55, apex_y),  # Top-right
            (xsize, ysize),  # Bottom-right
        ]],
        dtype=np.int32,
    )

    mask = np.zeros_like(masked_edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(masked_edges, mask)


def draw_extrapolated_lines(line_image, lines, y_min, y_max):
  global tracker
  if lines is not None:
    left_x, left_y = [], []
    right_x, right_y = [], []

    for line in lines:
      coords = line[0] if line.ndim == 2 else line
      x1, y1, x2, y2 = coords

      if x2 == x1:
        continue

      slope = (y2 - y1) / (x2 - x1)

      if slope < -0.5:  # Left lane
        left_x.extend([x1, x2])
        left_y.extend([y1, y2])
      elif slope > 0.5:  # Right lane
        right_x.extend([x1, x2])
        right_y.extend([y1, y2])

    # Calculate current frame coefficients
    current_left_fit = (
        np.polyfit(left_x, left_y, 1) if len(left_x) > 0 else None
    )
    current_right_fit = (
        np.polyfit(right_x, right_y, 1) if len(right_x) > 0 else None
    )
  else:
    current_left_fit = None
    current_right_fit = None

  # Smooth across past frames using the tracker
  smoothed_left = tracker.get_smoothed_fit(
      current_left_fit, tracker.left_fits
  )
  smoothed_right = tracker.get_smoothed_fit(
      current_right_fit, tracker.right_fits
  )

  # Draw smoothed left lane
  if smoothed_left is not None:
    m, b = smoothed_left
    x_bottom = int((y_max - b) / m)
    x_top = int((y_min - b) / m)
    cv2.line(line_image, (x_bottom, int(y_max)), (x_top, int(y_min)), (255, 0, 0), 10)

  # Draw smoothed right lane
  if smoothed_right is not None:
    m, b = smoothed_right
    x_bottom = int((y_max - b) / m)
    x_top = int((y_min - b) / m)
    cv2.line(line_image, (x_bottom, int(y_max)), (x_top, int(y_min)), (255, 0, 0), 10)



def process_frame(frame):
    """Applies the complete lane detection pipeline to a single video frame."""
    ysize, xsize = frame.shape[:2]

    # 1. Convert to Grayscale (OpenCV VideoCapture gives BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # 3. Canny Edge Detection
    low_threshold = 50
    high_threshold = 150
    full_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # 4. Region of Interest Masking
    apex_y = int(ysize * 0.66)
    masked_edges = select_region(frame, full_edges, apex_y)

    # 5. Hough Transform
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_length = 20
    max_line_gap = 50
    line_image = np.zeros_like(frame)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho,
        theta,
        threshold,
        np.array([]),
        min_line_length,
        max_line_gap,
    )

    # 6. Extrapolate and draw lines
    draw_extrapolated_lines(line_image, lines, y_min=apex_y, y_max=ysize)

    # 7. Blend lines with the original frame
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return combo


def main(args):
    # Initialize video capture from file (or 0 for webcam)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error opening video file: {args.video}")
        return

   	# Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate delay in milliseconds per frame (fallback to 30ms if FPS is invalid/0)
    delay = int(1000 / fps) if fps > 0 else 33

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("Processing video... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream reached.")
            break

        # Process the current frame
        processed_frame = process_frame(frame)

        # Write to output file if enabled
        if out:
            out.write(processed_frame)

        # Display the resulting frame live with proper timing
        cv2.imshow("Lane Detection", processed_frame)

        # Wait dynamically based on the video's FPS instead of a fixed 1ms
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break 

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Lane Detection using Hough Transforms"
    )
    parser.add_argument(
        "video", type=str, help="Path to the input video file (e.g., solidWhiteRight.mp4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save output video (e.g., output.mp4)",
    )
    args = parser.parse_args()
    main(args)