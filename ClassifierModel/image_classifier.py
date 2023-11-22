
import pickle
from PIL import Image
import numpy as np

class ImageClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        with open(self.model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def classify_image(self, image_path):
        preprocessed_image = self.load_preprocess_image(image_path)
        predicted_class = self.model.predict([preprocessed_image])[0]
        return predicted_class

    def load_preprocess_image(self, image_path):
        # Preprocess the image (resize, normalization, etc.)
        # Resize the image to the target size
        image = Image.open(image_path)
        target_size=(300,300)
        img = img.resize(target_size, Image.ANTIALIAS)
        
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        
        preprocessed_image = img_array  # Preprocess the image using appropriate techniques
        return preprocessed_image
