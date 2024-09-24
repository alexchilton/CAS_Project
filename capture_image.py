import cv2
import os
import time
from PIL import Image
#from rembg import remove
import requests

from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

url = "http://127.0.0.1:8000/predict/"

def load_and_preprocess_image(img_path, w=128, h=128):
    """
    Loading and pre-processing images for further use.
    Kwargs*: img_path: to be assigned, w: width of the target image, default value 128,
    h: height of the target image, default value 128.

    """

    img = image.load_img(img_path, target_size=(w, h))
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
    return img_array

# Initialize the webcam
cam = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cam.isOpened():
    raise Exception("Could not open video device")
else:
    print("Webcam opened successfully")

# Allow the camera to warm up
time.sleep(2)

# Capture and show images, save on key press
for i in range(5):
    ret, frame = cam.read()
    if not ret:
        print(f"Failed to capture frame {i}")
    else:
        print(f"Captured frame {i}")
        # Display the captured image
        cv2.imshow("Captured Image", frame)
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        # If 's' key is pressed, save the image
        if key == ord('s'):
            image_path = f"captured_image_{i}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Saving image to: {os.path.abspath(image_path)}")
            # Open the image file in binary mode
            with open(image_path, "rb") as image_file:
                # Send a POST request with the image file
                response = requests.post(url, files={"file": image_file})

            # Print the response from the server
            print(response.json())
    # Close the window if 'q' key is pressed
        elif key == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

