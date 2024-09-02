import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Function to preprocess the input image
def preprocess_input_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image(img_path):
    input_image = preprocess_input_image(img_path)
    
    # Make prediction
    predictions = model.predict(input_image)
    
    # Get the predicted class index
    predicted_class_index = tf.argmax(predictions, axis=-1).numpy()[0]

    # Return the predicted class index
    return int(predicted_class_index)

def main():
    # Path to the image you want to test
    image_path = 'y.jpg'  # Replace with the actual image path

    # Predefined class labels (ensure this matches your model's training)
    class_labels = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    ]
    
    # Predict the class
    predicted_class_index = predict_image(image_path)
    
    # Map the class index to the class label
    predicted_label = class_labels[predicted_class_index]
    
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class label: {predicted_label}")

if __name__ == "__main__":
    main()
