#Program for face dector by using OPENCV in PYTHON and SAVING/CAPTURING the IMAGE
#Install OPENCV by using command line terminal: (pip install openCV-python)
#Download a file "haarcascade_frontalface_default.xml" from KAGGLE & save the file in same folder where source code is saved

import cv2  # Import OpenCV library
import time # Import time
a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")# Loading the Haar Cascade for face detection
b = cv2.VideoCapture(0) # Initialize video capture
if not b.isOpened(): # Checking if the camera is opened successfully or not
    print("ERROR: PLEASE CHECK  YOUR CAMERA. IT IS UNABLE TO ACCESSIBLE :<(") # Error if the camera is  not opened
    exit()  
last_detection_time = time.time()
time_limit = 40
while True:
    c_rec, d_image = b.read() # Read a frame from the video
    if not c_rec or d_image is None:  # Checking if the frame was captured properly
        print("Failed to capture image. Exiting.")
        break
    clean_image = d_image.copy()
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)  # Converting the frame to grayscale
    f = a.detectMultiScale(e, scaleFactor=1.3, minNeighbors=5)  # Detecting faces in the frame
    if len(f) > 0:  # If faces are detected, update the last detection time
        last_detection_time = time.time()
        for (x, y, w, h) in f:  # Draw a rectangle around detected faces
            cv2.rectangle(d_image, (x, y), (x + w, y + h), (100, 200, 300), 5) # Frame for the faces detected
    cv2.imshow('Face Detector', d_image) # Displaying the frame with detected faces
    key = cv2.waitKey(1) & 0xFF   # Wait for a key press
    if key == ord('s'):  # Saving the image when key "s" is pressed
        image_filename = f"captured_from_face_dector_{int(time.time())}.jpg"
        cv2.imwrite(image_filename, clean_image)  # Saving the image with clean frame
        print(f"Image saved as {image_filename}")
    if key == 27:   # Exit the loop when the 'Esc' key is pressed
        break
    if time.time() - last_detection_time > time_limit:  
        print("No face detected for 40 seconds. Stopping the program.")  # Stoping the program after 40 seconds if no face is dected
        break
# Main Program
b.release()  
cv2.destroyAllWindows()  