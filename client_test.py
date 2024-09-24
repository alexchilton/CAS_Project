import requests

# URL of the FastAPI service
url = "http://127.0.0.1:8000/predict/"

# Path to the image file you want to test
image_path = "/Users/alexchilton/Downloads/working/validation/apple/Image_23.jpg"
image_path = "/Users/alexchilton/Downloads/working/validation/apple/Image_23.jpg"
image_path = "/Users/alexchilton/Downloads/working/validation/apple/Image_23.jpg"
image_path = "/Users/alexchilton/DataspellProjects/CAS_Project2/processed_images/captured_image_1.jpg"

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Send a POST request with the image file
    response = requests.post(url, files={"file": image_file})

# Print the response from the server
print(response.json())