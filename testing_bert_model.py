
import numpy as np
import keras_nlp
from tensorflow import keras, argmax


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# model_name = 'pretrained-classifier-1-epoch.keras'
model_name = './models/bert-classifier.keras'

with keras.utils.custom_object_scope({'BertClassifier':keras_nlp.models.BertClassifier, 'loss': keras.losses.SparseCategoricalCrossentropy(from_logits=True)}):
    model = keras.models.load_model(model_name)

prediction = model.predict(['''the great achievement of the catholic church lay in harmonizing , civilizing the deepest impulses of ordinary , ignorant people .
if you think your teacher is tough , wait until you get a boss . he doesn ' t have tenure .'''])
print(argmax(prediction, axis=-1).numpy())
