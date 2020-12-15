import tensorflow as tf
import tensorflow.python.platform
from tensorflow.keras import layers, models

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print(tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
))
