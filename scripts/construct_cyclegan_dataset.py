import os

import numpy as np
import skimage.io
from skimage.transform import resize



def normalize(arr):
    denominator = arr.max() - arr.min()
    if denominator == 0:
        return arr
    return (arr - arr.min()) / denominator


CHEXPERT_DIR = '/home/jmf/phd/datasets/data/chexpert/CheXpert-v1.0-small'
N_TRAIN = 1096
N_VAL = 1098
N_TEST = 1098
#OUTPUT_DIM = (600, 600)
OUTPUT_DIM = (300, 300)
OUTPUT_PREFIX = 'mednerf_xray_to_ct_dataset'
OUTPUT_A_TRAIN = f'{OUTPUT_PREFIX}/trainA'
OUTPUT_A_VAL = f'{OUTPUT_PREFIX}/valA'
OUTPUT_A_TEST = f'{OUTPUT_PREFIX}/testA'
OUTPUT_B_TRAIN = f'{OUTPUT_PREFIX}/trainB'
OUTPUT_B_VAL = f'{OUTPUT_PREFIX}/valB'
OUTPUT_B_TEST = f'{OUTPUT_PREFIX}/testB'


# X-Ray
def xray()
    os.makedirs(OUTPUT_A_TRAIN, exist_ok=True)
    for i in range(N_TRAIN):
        patient_n = f'{(i + 1):05}'
        chexpert_patient_dir = f'{CHEXPERT_DIR}/train/patient{patient_n}/study1'
        print(f'Orig image: {chexpert_patient_dir}/view1_frontal.jpg')
        img = skimage.io.imread(f'{chexpert_patient_dir}/view1_frontal.jpg')
        print(f'size: {img.shape} min {img.min()} max {img.max()}')
        img = normalize(img)
        img = resize(img, OUTPUT_DIM)
        skimage.io.imsave(f'{OUTPUT_A_TRAIN}/{i:04}.png', (img * 255).astype(np.uint8))


    os.makedirs(OUTPUT_A_VAL, exist_ok=True)
    for i in range(N_VAL):
        patient_n = f'{(N_TRAIN + i + 1):05}'
        chexpert_patient_dir = f'{CHEXPERT_DIR}/train/patient{patient_n}/study1'
        fname = f'{chexpert_patient_dir}/view1_frontal.jpg'

        print(f'Orig image: {fname}')
        if not os.path.exists(fname):
            print(f'Skipping {fname} as it does not exist')
            continue
        img = skimage.io.imread(f'{fname}')
        print(f'size: {img.shape} min {img.min()} max {img.max()}')
        img = normalize(img)
        img = resize(img, OUTPUT_DIM)
        skimage.io.imsave(f'{OUTPUT_A_VAL}/{i:04}.png', (img * 255).astype(np.uint8))

    os.makedirs(OUTPUT_A_TEST, exist_ok=True)
    idx = 0
    in_idx = 0
    while idx < N_TEST:
        patient_n = f'{(N_TRAIN + N_VAL + in_idx + 1):05}'
        chexpert_patient_dir = f'{CHEXPERT_DIR}/train/patient{patient_n}/study1'
        fname = f'{chexpert_patient_dir}/view1_frontal.jpg'

        print(f'Orig image: {fname}')
        if not os.path.exists(fname):
            print(f'Skipping {fname} as it does not exist')
            in_idx += 1
            continue
        img = skimage.io.imread(f'{fname}')
        print(f'size: {img.shape} min {img.min()} max {img.max()}')
        img = normalize(img)
        img = resize(img, OUTPUT_DIM)
        skimage.io.imsave(f'{OUTPUT_A_TEST}/{idx:04}.png', (img * 255).astype(np.uint8))
        idx += 1
        in_idx += 1

# CT
def ct():


#xray()
ct()
