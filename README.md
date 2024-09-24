# CAS_Project

Context
This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:

Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango
Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant

Inspiration
The primary motivation behind creating this dataset was to develop an application capable of recognizing food items from photographs. The application aims to suggest various recipes that can be prepared using the identified ingredients.

The dataset unfortunately had some issues, the test was identical to the validation and these were also within the train data - sometimes more than once.

To clean the data up the pre_traintest_fileoperation notebook should be run, this finds duplicates via a hash comparison then deletes the duplicates. It then takes another 10 random images and puts them in the validation class structure.

The final_notebook is from me which firstly does some basic image and class count displays, followed by a random forest and a random forest stratified split for a non deep learning model for comparison effect.

The end of the notebook shows a basic cnn with data augmentation.

Lara then implemented a transfer learning example using resnet.

I started doing an image capture using a webcam which then converts the image to 128 x 128, removes the background and then tries to do an identification via the model. 

It currently has some issues! work in progress...

The plan after image identification is to send the images to openai to get a recipe suggestion. It would be better as a mobile app as it is 
fairly difficult to use via webcam! Hands get in the way and add misinformation
