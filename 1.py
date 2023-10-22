import cv2
import numpy as np

def detect_shapes(frame, color):
    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for detection (blue color in this example)
    lower_color = np.array([100, 50, 50])
    upper_color = np.array([130, 255, 255])
    
    # Threshold the HSV image to get only the desired color
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through each contour
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        
        # Identify shapes based on number of vertices
        if len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Square"
        
        # Draw contours and shapes on the frame
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
        cv2.putText(frame, shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the video capture
    ret, frame = cap.read()
    
    if ret:
        # Detect shapes of a specific color (blue in this example)
        processed_frame = detect_shapes(frame, "blue")
        
        # Display the resulting frame
        cv2.imshow('Contours Detection', processed_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()