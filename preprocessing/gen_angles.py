import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import scipy.ndimage.interpolation
from skimage.transform import resize

def normalize(arr):
    denominator = arr.max() - arr.min()
    if denominator == 0:
        return arr
    return (arr - arr.min()) / denominator


from matplotlib import pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file

arrs = []
for i in range(216):
    path = f"data/manifest-1612365584013/MIDRC-RICORD-1B/MIDRC-RICORD-1B-419639-000340/01-18-2005-CT CHEST HIGH RESOLUTION-06379/2.000000-SUPINE CHEST RECON 12-09859/1-{i+1:03}.dcm"
    ds = dcmread(path)
    arr = ds.pixel_array
    arrs.append(arr)

arr = np.array(arrs)

os.makedirs('out', exist_ok=True)

# Angles
angle = 90
axis = 0
axis_1 = 1
axis_2 = 0
rotated = scipy.ndimage.interpolation.rotate(arr, angle, axes=(axis_1, axis_2))

axis_1 = 2
axis_2 = 0
angle_delta = 60
angle = 0.0
idx = 0
while angle < 360:
    new = scipy.ndimage.interpolation.rotate(rotated, angle, axes=(axis_1, axis_2))
    max_im = np.max(new, axis=0)
    n = normalize(max_im)
    n = resize(n, (128, 128))
    n[n<0.3] = 0
    skimage.io.imsave(f'out/{idx}.png', ((1.0 - n) * 255).astype(np.uint8))

    angle += angle_delta
    idx += 1
