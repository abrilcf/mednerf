# MedNERF
Adapting NERFs for use in medical imaging.

## Generating a video
### Step 1: Setup virtualenv


    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt


### Step 2: Download dataset
Download dataset [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969771) into `data` subdirectory of this repository.

### Step 3: Generate data


    python3 preprocess/main.py


This will output 3 pngs, each a max projection on one of the axes, into an `out` folder.

### Step 4: pixel-nerf download
Clone pixel-nerf repo into subdirectory:


    git clone https://github.com/sxyu/pixel-nerf.git

### Step 5: pixel-nerf setup 
Follow setup instructions in pixel-nerf repo's README and download pre-trained weights.

### Step 6: Run pixelnerf


    . ~/miniconda3/bin/activate
    conda activate pixelnerf
    cp out/* pixel-nerf/input
    cd pixel-nerf
    python eval/eval_real.py --num_views 72

The output of this will be series of video files in the `output` folder.
