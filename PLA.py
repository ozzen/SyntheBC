import numpy as np
from keras.models import load_model

model = load_model('Results/candidateBC.h5')
# model_weights = model.load_weights

# for layer in model.layers:
#     weights = layer.get_weights()
state = [[0.646, 0.001, 0.646, 0.001, 0.4658, 0.001, 0.714, 0.001, 0.6, -0.001]]
action = model.predict(state)
print(action)

# weights = model.layers[1].get_weights()[0]
# biases = model.layers[1].get_weights()[1]
# print(weights,biases)
# print(weights.size, len(weights))
# print(model.layers[1].get_config())
# print(np.amin(weights))
# print(np.amin(biases))
# print(model.summary())