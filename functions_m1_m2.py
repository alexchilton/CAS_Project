import numpy as np
import seaborn as sns
import pandas as pd
import os
import random
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import warnings

# tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image


# sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score


def get_true_label(img_path):
    """ 
    Given an image path, get the true label
    """

    return os.path.basename(os.path.dirname(img_path))

# Load and preprocess the images
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

def show_random_images(path = DIR_TRAIN, num_images=16):

    """ 
    Random selector of target image and display in a plt.figure. 
    path: path to train set directory, default DIR_TRAIN;
    num_images: images to be displayed by the random function, default=16.
    """

    # Define the path to your test directory
    vis_dir = path

    # Get the list of all subdirectories (classes)
    classes = os.listdir(vis_dir)

    # Initialize an empty list to store file paths
    image_paths = []

    # Loop through each class folder and collect a few image paths
    for cls in classes:
        class_dir = os.path.join(vis_dir, cls)
        images = os.listdir(class_dir)
        for img in images:
            image_paths.append(os.path.join(class_dir, img))

    # Randomly select 16 images from the test set
    random_images = random.sample(image_paths, num_images)

    # Print the paths of the randomly selected images (for debugging only)
    #for img_path in random_images:
    #    print(img_path)


    # Set up a 4x4 grid for plotting
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    # Loop through the grid and add an image to each subplot
    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(random_images[i])
        ax.imshow(img)
        ax.set_title(get_true_label(random_images[i]))
        ax.axis('off')  # Hide axes

    # Display the plot
    plt.show()

def show_number_in_class(path = DIR_TRAIN, strType = 'Training', strColor = 'skyblue'):
    """ 
    Count and plot number of pictures in each class for the directory selected. 
    Output displayed in barplot. 

    """

    # Get the list of all subdirectories (classes)
    classes = os.listdir(path)

    # Count the number of images in each class
    image_count = {cls: len(os.listdir(os.path.join(path, cls))) for cls in classes}

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(image_count.keys(), image_count.values(), color=strColor)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Class in the ' + strType + ' Data')
    plt.xticks(rotation=90, ha='right')  # Rotate class names for better readability
    plt.show()

def resize_and_save_image(input_path, output_path, size=(128, 128)):
    """ 
    Resize and save output images. 
    kwargs: 
    input_path: string, path of the input pictures; 
    output_path: string, path tp save output pictures;
    size: tuple, intended size of the output picture, default=(128, 128);

    """
    try:
        with Image.open(input_path) as img:
            # Handle images with Transparency in Palette mode
            if img.mode == 'P':
                img = img.convert('RGBA')
            # Convert image to RGB if it has an alpha channel or is Palette based
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            img = img.resize(size, Image.LANCZOS)
            img.save(output_path, format='JPEG')
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_directory(input_directory, output_directory):
    for root, dirs, files in os.walk(input_directory):
        # Create the corresponding directory structure in the output directory
        relative_path = os.path.relpath(root, input_directory)
        output_path = os.path.join(output_directory, relative_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Process each image file
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(output_path, file_name)
                resize_and_save_image(input_file_path, output_file_path)