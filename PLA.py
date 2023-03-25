import numpy as np
from keras.models import load_model

model = load_model('Results/candidateBaC_DG_isld.h5')
# model_weights = model.load_weights
#
# for layer in model.layers:
#     weights = layer.get_weights()
state = [[0.060949796, 1.0, -0.03706444, 1.058440119, 1.418440297, 0.143445991, 0.0, 1.0, 0.024624822, 0.017516932, 12.47, 0.667265678]]
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