import numpy as np
import pickle
from model import FactorizationMachine
from train import run_model

x = np.array([[1, 0, 0,  1, 0, 0, 0,  0.3, 0.3, 0.3, 0.0,  13,  0, 0, 0, 0],
              [1, 0, 0,  0, 1, 0, 0,  0.3, 0.3, 0.3, 0.0,  14,  1, 0, 0, 0],
              [1, 0, 0,  0, 0, 1, 0,  0.3, 0.3, 0.3, 0.0,  16,  0, 1, 0, 0],
              [0, 1, 0,  0, 0, 1, 0,  0.0, 0.0, 0.5, 0.5,  5,   0, 0, 0, 0],
              [0, 1, 0,  0, 0, 0, 1,  0.0, 0.0, 0.5, 0.5,  8,   0, 0, 1, 0],
              [0, 0, 1,  1, 0, 0, 0,  0.5, 0.0, 0.5, 0.0,  9,   0, 0, 0, 0],
              [0, 0, 1,  0, 0, 1, 0,  0.5, 0.0, 0.5, 0.0,  12,  1, 0, 0, 0]], dtype=np.float32)

y = np.array([5, 3, 1, 4, 5, 1, 5], dtype=np.float32)

model = FactorizationMachine(data_shape=16, latent_shape=10)

trained = run_model(model, x, y, epochs=200)

pickle.dump(trained, open("FM.model", "wb"))



