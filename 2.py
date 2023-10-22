import cv2
import numpy as np

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)  # Calculate the slope of the line

        if slope < -0.5:  # Left lane line
            left_lines.append((x1, y1))
            left_lines.append((x2, y2))
        elif slope > 0.5:  # Right lane line
            right_lines.append((x1, y1))
            right_lines.append((x2, y2))

    if len(left_lines) > 0:
        left_points = np.array(left_lines)
        left_lane = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)

        vx, vy, x, y = left_lane
        left_slope = vy / vx
        left_intercept = y - left_slope * x

        # Calculate the starting and ending points of the left lane line
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - left_intercept) / left_slope)
        x2 = int((y2 - left_intercept) / left_slope)

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if len(right_lines) > 0:
        right_points = np.array(right_lines)
        right_lane = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)

        vx, vy, x, y = right_lane
        right_slope = vy / vx
        right_intercept = y - right_slope * x

        # Calculate the starting and ending points of the right lane line
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - right_intercept) / right_slope)
        x2 = int((y2 - right_intercept) / right_slope)

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 6)

    return image

# Initialize video capture
cap = cv2.VideoCapture('input_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, roi_vertices)

    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw the detected lane lines on the original frame
    lane_lines = draw_lines(frame, lines)

    # Display the resulting frame
    cv2.imshow('Lane Detection', lane_lines)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()