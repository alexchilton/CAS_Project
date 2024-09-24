"""pip install fastapi uvicorn tensorflow pillow oython-multipart
then to run uvicorn fastapi_image_predictor_service:app --reload
"""


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the trained model
MODEL_LOCATION = '/Users/alexchilton/Downloads/working/best_model.keras'
#MODEL_LOCATION = '/Users/alexchilton/Downloads/working/resnet_model.keras'

model = load_model(MODEL_LOCATION)

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

def preprocess_image_resnet(img: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to the required format for the model.
    """
    img = img.resize(IMG_SIZE)  # Resize to the expected input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to the required format for the model.
    """
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Preprocess the image
        #img_array = preprocess_image_resnet(img)
        img_array = preprocess_image(img)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return JSONResponse(content={"predicted_class": int(predicted_class)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
