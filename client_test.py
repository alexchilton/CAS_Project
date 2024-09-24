import requests

# URL of the FastAPI service
url = "http://127.0.0.1:8000/predict/"

# Path to the image file you want to test
#image_path = "/Users/alexchilton/Downloads/working/validation/apple/Image_23.jpg"
#image_path = "/Users/alexchilton/Downloads/working/validation/apple/Image_23.jpg"
#image_path = "/Users/alexchilton/Downloads/working/validation/apple/Image_23.jpg"
image_path = "/Users/alexchilton/DataspellProjects/CAS_Project2/processed_images/captured_image_1.jpg"

# Define class labels
class_labels = {
    0: "apple",
    1: "banana",
    2: "beetroot",
    3: "bell pepper",
    4: "cabbage",
    5: "carrot",
    6: "cauliflower",
    7: "chilli pepper",
    8: "corn",
    9: "cucumber",
    10: "eggplant",
    11: "garlic",
    12: "ginger",
    13: "grapes",
    14: "jalepeno",
    15: "kiwi",
    16: "lemon",
    17: "lettuce",
    18: "mango",
    19: "onion",
    20: "orange",
    21: "paprika",
    22: "pear",
    23: "peas",
    24: "pineapple",
    25: "pomegranate",
    26: "potato",
    27: "raddish",
    28: "soy beans",
    29: "spinach",
    30: "sweet potato",
    31: "tomato",
    32: "turnip",
    33: "watermelon"
}


# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Send a POST request with the image file
    response = requests.post(url, files={"file": image_file})

# Print the response from the server
print(response.json())

# Get the predicted class index from the response
predicted_class = response.json().get("predicted_class")

# Get the class label from the dictionary
predicted_label = class_labels.get(predicted_class, "Unknown")

# Print the predicted class and label
print(f"Predicted class: {predicted_class}")
print(f"Predicted label: {predicted_label}")
