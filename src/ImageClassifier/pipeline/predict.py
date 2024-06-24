import numpy as np
from ImageClassifier.constants import *
from ImageClassifier.utils.common import read_yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
        config = read_yaml(CONFIG_FILE_PATH)
        params = read_yaml(PARAMS_FILE_PATH)
        evaluation = config.evaluation
        self.path_of_model = evaluation.trained_model_path
        self.target_size = params.IMAGE_SIZE[:2]


    
    def predict(self):
        # load model
        model = load_model(Path(self.path_of_model))

        imagename = self.filename
        logging.info(f"Loading image from {self.filename}")
        test_image = image.load_img(imagename, target_size = self.target_size)
        test_image = image.img_to_array(test_image)
        
        test_image = np.expand_dims(test_image, axis = 0)
        logging.info("Making prediction")
        result = np.argmax(model.predict(test_image), axis=1)
        logging.info(f"Prediction result: {result}")
        print(result)

        if result[0] == 1:
            prediction = 'Healthy'
            return [{ "image" : prediction}]
        else:
            prediction = 'Coccidiosis'
            return [{ "image" : prediction}]