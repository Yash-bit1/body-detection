import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parameters for hand detection
lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Lower bound for skin color (HSV)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)  # Upper bound for skin color (HSV)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert frame to HSV for hand detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for skin detection in the frame
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    
    # Find contours for hand detection in the mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # Minimum contour area for detecting hands
            # Get the bounding box for the detected hand
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Hand', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with face and hand detections
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
