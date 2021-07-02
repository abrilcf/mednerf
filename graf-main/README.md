# GRAF - MEDNERF

This repository contains official code for the paper
[GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://avg.is.tuebingen.mpg.de/publications/schwarz2020neurips).

... with modifications for DRR (Digitally Reconstructed Radiographs).

## Test on DRR data
Three GRAF models were trained for each category: chest, head & neck, knee. Refer to the following links to download the respective weights. Weights correspond to 48+ hs of training. Create a folder named `results/` under the main `graf-main/` folder and unzip the weights folder inside. 

Download [chest_all_360](https://drive.google.com/file/d/1XyfGewsoOJZ-mLGBUe8ORdhXul2IWdAf/view?usp=sharing).

Download [head_neck_all_360](https://drive.google.com/file/d/1uYRuFhZzotmbypNGovMKP1Oh1R52B_kQ/view?usp=sharing).

Download [knee_all_360](https://drive.google.com/file/d/1AZYoIzk4GBLCF_lCIl518RDNX0j9nMx_/view?usp=sharing).

You can test a model via: 
```
python eval.py configs/CONFIG.yaml --rotation_elevation --shape_appearance
```
where you replace CONFIG.yaml with one of the config files in `configs/` (chest.yaml / head_neck.yaml / knee.yaml).

Results will be saved under the same folder as the corresponding weights.
Please refer to the official repository instructions for further evaluation options (found below).

## Train on DRR data

### Data
You can find all DRR & real xrays on ncc: `/projects/cgw/medical`. Here is a description of the folders:

An <em>instance</em> comprehends 72 DRRs (each at 5 degrees) from a 360 degree rotation of a real CT scan.

`xrays` contains instances for each of the categories: 20 for chest, 20 for head & neck, 5 for knee. (2D posed images (.png) with their camera info (.txt)).

`chest_xrays` all images of the 20 chest instances (.png).

`head_neck_xrays` all images of the 20 head & neck instances (.png, res. 128x128)

`knee_xrays` all images of the 5 knee instances (.png, res. 128x128)

So far experiments have been made on these last three folders.



The following folders correspond to real xrays (not used yet).

(TODO: Condition a GRAF on real xrays at inference time).

`real_head_xrays` Single-real head xrays (.png, res. 128x128)

`real_knee_xrays` Single-real knee xrays (.png, res. 128x128)

`MURA` contains the [Stanford's musculoskeletal radiographs dataset](https://stanfordmlgroup.github.io/competitions/mura/) (This is for further testing).


You can train a model from scratch via:
```
python train.py configs/CONFIG.yaml
```

## Generate DRR images from CT scans
To generate xrays images (.png) at different angles from CT scans use the script `generate_drr.py` under the folder `data/`. To run it you need to install the [Plastimatch's build](http://plastimatch.org/). Version 1.9.3 was used.

Then replace `input_path` with the path to the .dcm files or .mha file of the CT. `save_root_path` for the path where you want the xrays images to be saved, and `plasti_path` to the path of the build.

### Overview of input arguments
Replace the following variables within the file:

- `input_path`: path to the .dcm files or .mha file of the CT.
- `save_root_path`: path where you want the xrays images to be saved. 
- `plasti_path`: path of the build. 
- `multiple_view_mode <True | False>`: generate single xrays from lateral or frontal views or multiple images from a circular rotation around the z axis.
    If False you need to specify the view with the argument `frontal_dir <True | False>` (false for lateral view).
    If True you need to specify `num_xrays` to generate equally spaced number of views and `angles` to input the difference between neighboring angles (in  degrees).
- `preprocessing <True | False>`: set this to True if files are .dcm for Hounsfield Units conversion. Set to False if given file is raw (.mha), for which       you need to provide its path under the variable `raw_input_file`.
- `detector_size`: pair of values in mm
- `bg_color`: choose either black or white background.
- `resolution`: size of the output xrays images.


# GRAF - Official instructions

This repository contains official code for the paper
[GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://avg.is.tuebingen.mpg.de/publications/schwarz2020neurips).

You can find detailed usage instructions for training your own models and using pre-trained models below.


If you find our code or paper useful, please consider citing

    @inproceedings{Schwarz2020NEURIPS,
      title = {GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis},
      author = {Schwarz, Katja and Liao, Yiyi and Niemeyer, Michael and Geiger, Andreas},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2020}
    }

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `graf` using
```
conda env create -f environment.yml
conda activate graf
```

Next, for nerf-pytorch install torchsearchsorted. Note that this requires `torch>=1.4.0` and `CUDA >= v10.1`.
You can install torchsearchsorted via
``` 
cd submodules/nerf_pytorch
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ../../../
```

## Demo

You can now test our code via:
```
python eval.py configs/carla.yaml --pretrained --rotation_elevation
```
This script should create a folder `results/carla_128_from_pretrained/eval/` where you can find generated videos varying camera pose for the Cars dataset.

## Datasets

If you only want to generate images using our pretrained models you do not need to download the datasets.
The datasets are only needed if you want to train a model from scratch.

### Cars

To download the Cars dataset from the paper simply run
```
cd data
./download_carla.sh
cd ..
```
This creates a folder `data/carla/` downloads the images as a zip file and extracts them to `data/carla/`. 
While we do <em>not</em> use camera poses in this project we provide them for completeness. Your can download them by running
```
cd data
./download_carla_poses.sh
cd ..
```
This downloads the camera intrinsics (single file, equal for all images) and extrinsics corresponding to each image.  

### Faces

Download [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Then replace `data/celebA` in `configs/celebA.yaml` with `*PATH/TO/CELEBA*/Img/img_align_celebA`.

Download [celebA_hq](https://github.com/tkarras/progressive_growing_of_gans).
Then replace `data/celebA_hq` in `configs/celebAHQ.yaml` with `*PATH/TO/CELEBA_HQ*`.

### Cats
Download the [CatDataset](https://www.kaggle.com/crawford/cat-dataset).
Run
```
cd data
python preprocess_cats.py PATH/TO/CATS/DATASET
cd ..
```
to preprocess the data and save it to `data/cats`.
If successful this script should print: `Preprocessed 9407 images.`

### Birds
Download [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and the corresponding [Segmentation Masks](https://drive.google.com/file/d/1EamOKGLoTuZdtcVYbHMWNpkn3iAVj8TP/view).
Run
```
cd data
python preprocess_cub.py PATH/TO/CUB-200-2011 PATH/TO/SEGMENTATION/MASKS
cd ..
```
to preprocess the data and save it to `data/cub`.
If successful this script should print: `Preprocessed 8444 images.`

## Usage

When you have installed all dependencies, you are ready to run our pre-trained models for 3D-aware image synthesis.

### Generate images using a pretrained model

To evaluate a pretrained model, run 
```
python eval.py CONFIG.yaml --pretrained --fid_kid --rotation_elevation --shape_appearance
```
where you replace CONFIG.yaml with one of the config files in `./configs`. 

This script should create a folder `results/EXPNAME/eval` with FID and KID scores in `fid_kid.csv`, videos for rotation and elevation in the respective folders and an interpolation for shape and appearance, `shape_appearance.png`. 

Note that some pretrained models are available for different image sizes which you can choose by setting `data:imsize` in the config file to one of the following values:
```
configs/carla.yaml: 
    data:imsize 64 or 128 or 256 or 512
configs/celebA.yaml:
    data:imsize 64 or 128
configs/celebAHQ.yaml:
    data:imsize 256 or 512
```

### Train a model from scratch

To train a 3D-aware generative model from scratch run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with your config file.
The easiest way is to use one of the existing config files in the `./configs` directory 
which correspond to the experiments presented in the paper. 
Note that this will train the model from scratch and will not resume training for a pretrained model.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./monitoring --port 6006
```
where you replace `OUTPUT_DIR` with the respective output directory.

For available training options, please take a look at `configs/default.yaml`.

### Evaluation of a new model

For evaluation of the models run
```
python eval.py CONFIG.yaml --fid_kid --rotation_elevation --shape_appearance
```
where you replace `CONFIG.yaml` with your config file.

## Multi-View Consistency Check

You can evaluate the multi-view consistency of the generated images by running a Multi-View-Stereo (MVS) algorithm on the generated images. This evaluation uses [COLMAP](https://colmap.github.io/) and make sure that you have COLMAP installed to run
```
python eval.py CONFIG.yaml --reconstruction
```
where you replace `CONFIG.yaml` with your config file. You can also evaluate our pretrained models via:
```
python eval.py configs/carla.yaml --pretrained --reconstruction
```
This script should create a folder `results/EXPNAME/eval/reconstruction/` where you can find generated multi-view images in `images/` and the corresponding 3D reconstructions in `models/`.

## Further Information

### GAN training

This repository uses Lars Mescheder's awesome framework for [GAN training](https://github.com/LMescheder/GAN_stability).

### NeRF

We base our code for the Generator on this great [Pytorch reimplementation](https://github.com/yenchenlin/nerf-pytorch) of Neural Radiance Fields.
