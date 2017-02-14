#!/usr/bin/env python

import PIL.Image
import numpy as np
import tensorflow as tf
from time import clock

from dream import *

# use convert('RGB') to fix png (RGBA) problem
img0 = PIL.Image.open('pilatus800.jpg').convert('RGB')
img0 = np.float32(img0)

layer = 'mixed4d_3x3_bottleneck_pre_relu'

start = clock()

render_deepdream(tf.square(T(layer)), img0, layer + '_non.png')
render_deepdream(T(layer)[:, :, :, 139], img0, 'flower_non.png')
render_deepdream(tf.square(T('mixed4c')), img0, 'mixed4c_non.png')

render_lap_deepdream(tf.square(T(layer)), img0, layer + '_lap.png')
render_lap_deepdream(T(layer)[:, :, :, 139], img0, 'flower_lap.png')
render_lap_deepdream(tf.square(T('mixed4c')), img0, 'mixed4c_lap.png')

# Uncomment the following code only if you want to check all the layers.
'''
with open('models/mixed_layer', 'r') as f:
    for layer in f:
        layer = layer[:-1]
        render_deepdream(tf.square(T(layer)), img0, layer + '_non.png')
        render_lap_deepdream(tf.square(T(layer)), img0, layer + '_lap.png')

with open('models/single_layer', 'r') as f:
    for layer in f:
        layer = layer[:-1]
        render_deepdream(tf.square(T(layer)), img0, layer + '_non.png')
        render_lap_deepdream(tf.square(T(layer)), img0, layer + '_lap.png')
'''

end = clock()
print(end - start)
