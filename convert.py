"""this is a test code """
import coremltools
import numpy as np
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array

MODEL_ARC_PATH = "finetuning.h5"

# JSONファイルからモデルのアーキテクチャを得る
model_arc_str = open(MODEL_ARC_PATH).read()
model = model_from_json(model_arc_str)

# モデル構成の確認
model.summary()

coreml_model = coremltools.converters.keras.convert('finetuning.h5')
coreml_model.save('vtuber.mlmodel')
