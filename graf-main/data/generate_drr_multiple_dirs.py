# ------------------------------------------------------------------------------
# Ref: https://github.com/kylekma/X2CT/blob/master/CT2XRAY/xraypro.py
# modified to create multiple xray views
# plastimatch used v 1.9.3 - download: http://plastimatch.org/
#
# Generate xrays (Digitally Reconstructed Radiographs - DRR)
# from CT scan (e.g. dicom files .dcm / or raw .mha raw)
# 
# ------------------------------------------------------------------------------
import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from subprocess import check_output as qx
from pydicom import dcmread
from glob import glob

# input could be .mhd/.mha format
def load_scan_mhda(path):
    img_itk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_itk)
    return img_itk, img, img_itk.GetOrigin(), img_itk.GetSize(), img_itk.GetSpacing()

# .dcm to .mha / .mhd
def dicom2raw(dicom_folder, output_file):
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()

    size = image.GetSize()
    print("Image size:", size[0], size[1], size[2])

    print("Writing image:", output_file)

    sitk.WriteImage(image, output_file)

# compute xray source center in world coordinate
def get_center(origin, size, spacing):
    origin = np.array(origin)
    size = np.array(size)
    spacing = np.array(spacing)
    center = origin + (size - 1) / 2 * spacing
    return center

# convert a ndarray to string
def array2string(ndarray):
    ret = ""
    for i in ndarray:
        ret = ret + str(i) + " "
    return ret[:-2]

# save a .pfm file as a .png file
def savepng(filename, direction, idx):
    file = open(filename, 'rb') 
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True    
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
    
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    raw_data = np.reshape(data, shape)
    max_value = raw_data.max()
    im = (raw_data / max_value * 255).astype(np.uint8)
    # PA view should do additional left-right flip
    if direction == 1:
        im = np.fliplr(im)
    
    savedir, _ = os.path.split(filename)
    outfile = os.path.join(savedir, f"xray{idx:04}.png".format(direction))
    # plt.imshow(im, cmap=plt.cm.gray)
    plt.imsave(outfile, im, cmap=plt.cm.gray)
    # plt.imsave saves an image with 32bit per pixel, but we only need one channel
    image = cv2.imread(outfile)
    gray = cv2.split(image)[0]
    cv2.imwrite(outfile, gray)


if __name__ == '__main__':
    global_path = '/media/abrilcf/AprilsDrive/phd/datasets/MIDRC-RICORD-1B'
    save_path = '/media/abrilcf/899f2142-a79b-460e-b428-d45e0930f90b/extra/lap/PhD/DataSets/medical/chest'
    # get dir names
    dirs = [f.path for f in os.scandir(global_path) if f.is_dir()]

    # False: single xray output
    multiple_view_mode = True
    # True: dicom conversion & HU adjustment
    # False: output xray from given .mha (i.e., raw_input_file)
    preprocessing = True
    # Use "500 500" for chest
    # use "350 350" for knee
    detector_size = "500 500"
    # Black bg: "0 255", white bg: "255 0"
    bg_color = "0 255"
    # If single view, choose frontal or lateral view
    frontal_dir = "0 1 0"
    lateral_dir = "1 0 0"
    resolution = "256 256"
    # If multiple view:
    num_xrays = "2"
    angle = "90"

    for i in range(len(dirs)):
        ct_files = [y for x in os.walk(dirs[i]) for y in glob(os.path.join(x[0], '*.dcm'))]
        ds = dcmread(ct_files[0])
        # Note that the following values are sometimes missing from the CT
        # if missing, use sad="541", sid="949" for: chest & head-neck
        # DistanceSourceToPatient in mm
        sad = str(ds.DistanceSourceToPatient)
        sid = str(ds.DistanceSourceToDetector)

        save_name = f"{i+1:04}_chest"
        save_dir = os.path.join(save_path, save_name)
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
            
        if preprocessing:
            # Converting dicom files to .mha for plastimatch processing
            raw_input_file = os.path.join(save_dir, '{}.mha'.format(save_name))
            dicom2raw(os.path.abspath(os.path.join(ct_files[0], '..')),
                      raw_input_file)
            # truncates the inputs to the range of [-1000,+1000]
            adjust_lst = ['plastimatch', "adjust", "--input",
                          raw_input_file, "--output", raw_input_file,
                          "--pw-linear", "-inf,0,-1000,-1000,+1000,+1000,inf,0"]
            #"-inf,0,-1000,-1000,+1000,+1000,inf,0"
            #-inf,0,-1000,-1000,+3000,+3000,inf,
            output = qx(adjust_lst)
    #        set_bone_threshold = [plasti_path+'/plastimatch', "threshold", "--input",
    #                              mha_adjust, "--output", mha_bone,
    #                              "--above", "-1000"]
    #        output = qx(set_bone_threshold)

        ct_itk, ct_scan, ori_origin, ori_size, ori_spacing = load_scan_mhda(raw_input_file)
        # compute isocenter
        print("input_size: ", ori_size)
        center = get_center(ori_origin, ori_size, ori_spacing)
        print("center: ", center)

        if multiple_view_mode:
            o_path = os.path.join(save_dir, 'xray_')
            drr_lst = ['plastimatch', "drr", "-A", "cuda", "-t", "pfm",
                       "--algorithm", "uniform", "--gantry-angle", "0",
                       "-N", angle, "-a", num_xrays, "--sad", sad, "--sid", sid,
                       "--autoscale", "--autoscale-range", bg_color,
                       "-r", resolution, "-o", array2string(center),
                       "-z", detector_size, "-P", "preprocess",
                       "-I", raw_input_file, "-O", o_path]
            output = qx(drr_lst)
        else:
            o_path = os.path.join(save_dir, 'xray_')
            drr_lst = ['plastimatch', "drr", "-t", "pfm",
                       "-i", "exact", "-P", "preprocess",
                       "--autoscale-range", "0 255",
                       "--gantry-angle", "0",
                       "-n", lateral_dir, "--sad", sad, "--sid", sid,
                       "-r", resolution, "-o", array2string(center),
                       "-z", detector_size, "-e",
                       "-O", o_path]
            #"--autoscale", "--autoscale-range", bg_color,
            output = qx(drr_lst)

        print("sad: ", sad, " sid: ", sid, " -o: ", array2string(center))

        file_paths = []
        to_delete = []
        files = os.listdir(save_dir)
        for f in files:
            if f.endswith(".pfm"):
                file_paths.append(os.path.join(save_dir, f))
                to_delete.append(os.path.join(save_dir, f))
            elif f.endswith(".txt"):
                to_delete.append(os.path.join(save_dir, f))
        file_paths.sort()
        for i in range(len(file_paths)):
            savepng(file_paths[i], 1, i)

        # Deleting unnecessary files
        for f in to_delete:
            os.remove(f)

