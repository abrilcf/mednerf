# MEDNERF: Medical Neural Radiance Fields for Reconstructing 3D-aware CT-Projections from a Single X-ray

[Paper](https://arxiv.org/abs/2202.01020)
Submitted to IEEE EMBC 2022


## Train on DRR data

### Data
You can find all DRR in the following [link](https://drive.google.com/file/d/1_EJX3LnRMG5uXEhZ63C2eYoY4hjwmipP/view?usp=sharing). Here is a description of the folders:

An <em>instance</em> comprehends 72 DRRs (each at 5 degrees) from a 360 degree rotation of a real CT scan.


`chest_xrays` all images of the 20 chest instances (.png, res. 128x128).

`knee_xrays` all images of the 5 knee instances (.png, res. 128x128)



You can train a model from scratch via:
```
python train.py configs/CONFIG.yaml
```

## Reconstruction given an X-ray
After training a model, you can test its capacity to reconstruct 3D-aware CT projections given a single X-ray. 

To execute the reconstruction, please run:
```
python finetune_xray.py --xray_img_path path_to_xray --save_dir path_to_save_dir --model path_to_trained_model
```
(to use ray for finetuning, please change the runtime environment with your configuration).


## Generate DRR images from CT scans
To generate xrays images (.png) at different angles from CT scans use the script `generate_drr.py` under the folder `data/`. To run it you need to install the [Plastimatch's build](http://plastimatch.org/). Version 1.9.3 was used.

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

## Acknowledgments

This codebase is heavily based on the [GRAF](https://github.com/autonomousvision/graf) code base.

We thank all authors for the wonderful code!

## Citation
If you use our model for your research, please cite the following work.

```bash
@misc{coronafigueroa2022mednerf,
      title={MedNeRF: Medical Neural Radiance Fields for Reconstructing 3D-aware CT-Projections from a Single X-ray}, 
      author={Abril Corona-Figueroa and Jonathan Frawley and Sam Bond-Taylor and Sarath Bethapudi and Hubert P. H. Shum and Chris G. Willcocks},
      year={2022},
      eprint={2202.01020},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
