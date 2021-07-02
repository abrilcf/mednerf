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
    input_path = '/media/newuser/AprilsDrive/phd/datasets/knee006/Knee'
    save_root_path = '/media/newuser/AprilsDrive/phd/datasets/xrays/knee/knee_006'
    plasti_path = '/home/newuser/Documents/PhD/plastimatch-build'
    output_raw_name = 'raw_file'
    # False: single xray output
    multiple_view_mode = True
    # True: dicom conversion & HU adjustment
    # False: output xray from given .mha (i.e., raw_input_file)
    preprocessing = True
    raw_input_file = '/media/newuser/AprilsDrive/phd/datasets/xrays/knee/knee_006/raw_file.mha'
    # Use "500 500" for chest
    # use "350 350" for knee
    detector_size = "350 350"
    # Black bg: "0 255", white bg: "255 0"
    bg_color = "0 255"
    # If single view, choose frontal or lateral view
    frontal_dir = True
    resolution = "128 128"
    # If multiple view:
    num_xrays = "72"
    angle = "5"
    ds = dcmread(os.path.join(input_path, os.listdir(input_path)[0]))
    # Note that the following values are sometimes missing from the CT
    # if missing, use sad=541, sid=949 for chest
    # DistanceSourceToPatient in mm
    sad = str(ds.DistanceSourceToPatient)
    # DistanceSourceToDetector in mm
    sid = str(ds.DistanceSourceToDetector)

    if preprocessing:
        # Converting dicom files to .mha for plastimatch processing
        raw_input_file = os.path.join(save_root_path, '{}.mha'.format(output_raw_name))
        dicom2raw(input_path, raw_input_file)
        # truncates the inputs to the range of [-1000,+1000]
        adjust_lst = [plasti_path+'/plastimatch', "adjust", "--input",
                      raw_input_file, "--output", raw_input_file,
                      "--pw-linear", "-inf,0,-1000,-1000,+1000,+1000,inf,0"]
        #"-inf,0,-1000,-1000,+1000,+1000,inf,0"
        output = qx(adjust_lst)

    ct_itk, ct_scan, ori_origin, ori_size, ori_spacing = load_scan_mhda(raw_input_file)
    # compute isocenter
    print("input_size: ", ori_size)
    center = get_center(ori_origin, ori_size, ori_spacing)
    print("center: ", center)

    if multiple_view_mode:
        o_path = os.path.join(save_root_path, 'xray')
        drr_lst = [plasti_path+'/plastimatch', "drr", "-t", "pfm",
                   "--algorithm", "uniform", "--gantry-angle", "0",
                   "-N", angle, "-a", num_xrays, "--sad", sad, "--sid", sid,
                   "--autoscale", "--autoscale-range", bg_color,
                   "-r", resolution, "-o", array2string(center),
                   "-z", detector_size, "-P", "preprocess",
                   "-I", raw_input_file, "-O", o_path]
        output = qx(drr_lst)
    else:
        if frontal_dir:
            dir = "0 1 0"
        else:
            dir = "1 0 0"
        o_path = os.path.join(save_root_path, 'xray')
        drr_lst = [plasti_path+'/plastimatch', "drr", "-t", "pfm",
                   "--algorithm", "uniform", "--gantry-angle", "0",
                   "-n", dir, "--sad", sad, "--sid", sid,
                   "--autoscale", "--autoscale-range", bg_color,
                   "-r", resolution, "-o", array2string(center),
                   "-z", detector_size, "-P", "preprocess",
                   "-I", raw_input_file, "-O", o_path]
        output = qx(drr_lst)

    file_paths = []
    to_delete = []
    files = os.listdir(save_root_path)
    for f in files:
        if f.endswith(".pfm"):
            file_paths.append(os.path.join(save_root_path, f))
            to_delete.append(os.path.join(save_root_path, f))
        elif f.endswith(".mha"):
            to_delete.append(os.path.join(save_root_path, f))
    file_paths.sort()
    for i in range(len(file_paths)):
        savepng(file_paths[i], 1, i)

    # Deleting unnecessary files
    for f in to_delete:
        os.remove(f)
