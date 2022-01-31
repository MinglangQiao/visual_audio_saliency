import numpy as np


## for vggnet input
MEAN_VALUE = np.array([123.68, 116.779, 103.939])   # RGB, refere to this: https://github.com/tensorflow/models/issues/517
# MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:, None, None]