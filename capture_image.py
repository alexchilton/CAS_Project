import cv2
import os
import time
from PIL import Image
from rembg import remove

from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

MODEL_LOCATION = '/Users/alexchilton/Downloads/working/best_model.keras'

# Load the trained model
model = tf.keras.models.load_model(MODEL_LOCATION)

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
            #image_path = f"captured_image_{i}.jpg"
            #cv2.imwrite(image_path, frame)
            #print(f"Saving image to: {os.path.abspath(image_path)}")
            print("saving")
    # Close the window if 'q' key is pressed
        elif key == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

image_paths = [
    '/Users/alexchilton/DataspellProjects/CAS_Project2/captured_image_0.jpg',
    '/Users/alexchilton/DataspellProjects/CAS_Project2/captured_image_1.jpg',
    '/Users/alexchilton/DataspellProjects/CAS_Project2/captured_image_2.jpg',
    '/Users/alexchilton/DataspellProjects/CAS_Project2/captured_image_3.jpg',
    '/Users/alexchilton/DataspellProjects/CAS_Project2/captured_image_4.jpg'
]

# Directory to save the processed images
output_dir = '/Users/alexchilton/DataspellProjects/CAS_Project2/processed_images'
os.makedirs(output_dir, exist_ok=True)

# Process each image
# Process each image
for img_path in image_paths:
    # Load the image
    img = Image.open(img_path)

    # Remove the background
    img_no_bg = remove(img)

    # Resize the image to 128x128
    img_resized = img_no_bg.resize((128, 128))

    # Convert to RGB mode
    img_rgb = img_resized.convert("RGB")

    # Save the processed image
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    img_rgb.save(output_path, format='JPEG')
    print(f"Processed image saved to: {output_path}")
# List to store the preprocessed images
preprocessed_images = []

# Iterate over the image paths
for img_path in image_paths:
    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path)
    # Append the preprocessed image to the list
    preprocessed_images.append(img_array)

print("number of images is ", len(preprocessed_images))

# Print model summary to check input shape
model.summary()

# Print TensorFlow and Keras versions
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Ensure the preprocessed_images array has the correct shape and data type
preprocessed_images = np.vstack(preprocessed_images).astype('float32')

# Print the shape and data type for debugging
print("preprocessed_images shape:", preprocessed_images.shape)
print("preprocessed_images dtype:", preprocessed_images.dtype)

try:
    # Predict the classes of the preprocessed images
    predictions = model.predict(preprocessed_images)

    # Get the class with the highest probability for each image
    predicted_classes = np.argmax(predictions, axis=1)

    # Print the predicted classes
    print("Predicted classes:", predicted_classes)
except Exception as e:
    print("Error during prediction:", str(e))