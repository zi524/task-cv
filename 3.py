import cv2
import numpy as np

def order_points(pts):
    # Order the points in the top-left, top-right, bottom-right, and bottom-left order
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Set the destination points for the perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Load the image
image = cv2.imread('input_image.jpg')

# Resize the image for better display
image = cv2.resize(image, (800, 800))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge map
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Iterate through the contours and find the document contour
for contour in contours:
    # Approximate the contour with a polygon
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # If the approximated polygon has four points, assume it is the document contour
    if len(approx) == 4:
        document_contour = approx
        break

# Perform perspective transform to obtain the scanned image
scanned_image = four_point_transform(image, document_contour.reshape(4, 2))

# Display the original and scanned images
cv2.imshow("Original Image", image)
cv2.imshow("Scanned Image", scanned_image)

# Save the scanned image
cv2.imwrite("scanned_image.jpg", scanned_image)

# Wait for key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()