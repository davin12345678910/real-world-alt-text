import cv2

# Open the webcam with explicit permission request for Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam. Please grant camera access permission.")
    exit()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Display the webcam feed
        cv2.imshow('Webcam Live', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Streaming has stopped")

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()