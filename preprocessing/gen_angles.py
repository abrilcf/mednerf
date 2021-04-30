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
#print(arr.shape)

#fig, axs = plt.subplots(3)

# Axis projections
os.makedirs('out', exist_ok=True)
'''
for axis in range(3):
    max_im = np.max(arr, axis=axis)
    n = normalize(max_im)
    n = resize(n, (128, 128))
    n[n<0.3] = 0
    skimage.io.imsave(f'out/{axis}.png', ((1.0 - n) * 255).astype(np.uint8))
    skimage.io.imsave(f'out/{axis}_normalize.png', ((1.0 - n) * 255).astype(np.uint8))
    #print(max_im.shape)
    #axs[axis].imshow(max_im)
'''
#plt.show()

# Angles
'''
print(arr.shape)

fig, axs = plt.subplots(3)
for axis_1, axis_2 in [[0, 1], [0, 2], [1,0], [1,2], [2, 0], [2,1]]:
    for axis, angle in enumerate([60, 90, 120]):
        #angle = math.pi / 6  # 30 deg
        #angle = math.pi / 2  # 30 deg
        #rotated = scipy.ndimage.interpolation.rotate(arr, angle, axes=(1, 2))
        #rotated = scipy.ndimage.interpolation.rotate(arr, angle, axes=(0, 1))
        #rotated = scipy.ndimage.interpolation.rotate(arr, angle, axes=(0, 2))
        rotated = scipy.ndimage.interpolation.rotate(arr, angle, axes=(axis_1, axis_2))

        print(rotated.shape)
        max_im = np.max(rotated, axis=0)
        #axs[axis].imshow(max_im, cmap="gray")

        max_im = normalize(max_im)
        max_im = resize(max_im, (128, 128))
        skimage.io.imsave(f'out/{axis_1}_{axis_2}_{axis}.png', ((1.0 - max_im) * 255).astype(np.uint8))
    #plt.show()
        #max_im = np.max(arr, axis=axis)
        #n = normalize(max_im)
        #n = resize(n, (128, 128))
        #n[n<0.3] = 0
        #skimage.io.imsave(f'out/{axis}.png', ((1.0 - n) * 255).astype(np.uint8))
        #skimage.io.imsave(f'out/{axis}_normalize.png', ((1.0 - n) * 255).astype(np.uint8))
        ##print(max_im.shape)
        ##axs[axis].imshow(max_im)
'''

print(arr.shape)

fig, axs = plt.subplots(5)
angle = 90
axis = 0
axis_1 = 1
axis_2 = 0
rotated = scipy.ndimage.interpolation.rotate(arr, angle, axes=(axis_1, axis_2))
print(rotated.shape)
max_im = np.max(rotated, axis=0)
axs[axis].imshow(max_im, cmap="gray")

#axis_1 = 1
#axis_2 = 0

#axis_1 = 1
#axis_2 = 2
#axis_1 = 2
#axis_2 = 1

axis_1 = 2
axis_2 = 0
angle_delta = 30
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
exit()

angle = 60
axis = 1
axis_1 = 2
axis_2 = 0
new = scipy.ndimage.interpolation.rotate(rotated, angle, axes=(axis_1, axis_2))

max_im = np.max(new, axis=0)
axs[axis].imshow(max_im, cmap="gray")

angle = 30
axis = 2
axis_1 = 2
axis_2 = 0
new = scipy.ndimage.interpolation.rotate(rotated, angle, axes=(axis_1, axis_2))

max_im = np.max(new, axis=0)
axs[axis].imshow(max_im, cmap="gray")

angle = 300 
axis = 3
new = scipy.ndimage.interpolation.rotate(rotated, angle, axes=(axis_1, axis_2))

max_im = np.max(new, axis=0)
axs[axis].imshow(max_im, cmap="gray")

angle = 330 
axis = 4
new = scipy.ndimage.interpolation.rotate(rotated, angle, axes=(axis_1, axis_2))

max_im = np.max(new, axis=0)
axs[axis].imshow(max_im, cmap="gray")

plt.show()
