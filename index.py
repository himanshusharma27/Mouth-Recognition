import cv2
import numpy as np

# Load pre-trained face and mouth cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mouth.xml')

# Constants for dot size and mouth opening threshold
DOT_RADIUS = 2
MOUTH_THRESHOLD = 30  # 3 cm

# Function to apply the filter effect
def apply_filter(frame, mouth):
    x, y, w, h = mouth
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, "Mouth Opened!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# Function to detect and track mouth in the frame
def detect_mouth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        for (mx, my, mw, mh) in mouths:
            # Calculate the center of the mouth
            cx = int(mx + 0.5 * mw)
            cy = int(my + 0.5 * mh)

            # Apply the red dots on the edges of the mouth
            cv2.circle(frame, (x + mx, y + my), DOT_RADIUS, (0, 0, 255), -1)
            cv2.circle(frame, (x + mx + mw, y + my), DOT_RADIUS, (0, 0, 255), -1)

            # Check if the mouth is opened wider than the threshold
            if mh > MOUTH_THRESHOLD:
                frame = apply_filter(frame, (x + mx, y + my, mw, mh))

    return frame

# Main video capture loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = detect_mouth(frame)
    cv2.imshow('Mouth Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
