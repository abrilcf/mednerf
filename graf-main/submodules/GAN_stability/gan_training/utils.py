import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torch.nn.functional as F
import numpy as np


def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    while n < N:
        x_next, y_next = next(iter(data_loader))
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    return x, y


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
        
      
def rotation(x, degs):
    x_rot = []
    for deg in degs:
        if deg == 0:
           x_rot.append(x)
        elif deg == 90:
           x_rot.append(x.transpose(2, 3).flip(2))
        elif deg == 180:
           x_rot.append(x.flip(2).flip(3))
        elif deg == 270:
           x_rot.append(x.transpose(2, 3).flip(3))
    #x_rot = torch.cat(x_rot,0)
    return x_rot

def fliprot(x, aug):
    x_flip = []
    x_flip.append(x)
    x_flip.append(x.flip(2))
    x_flip.append(x.flip(3))
    x_flip.append(x.transpose(2, 3).flip(2))
    #x_flip = torch.cat(x_flip,0)
    return x_flip

def cropping(x, aug):
    b, c, h, w = x.shape
    boxes = [[0,      0,      h,      w],
             [0,      0,      h*0.75, w*0.75],
             [0,      w*0.25, h*0.75, w],
             [h*0.25, 0,      h,      w*0.75],
             [h*0.25, w*0.25, h,      w]]
    x_crop = []
    for i in range(np.shape(boxes)[0]):
        cropped = x[:,:,int(boxes[i][0]):int(boxes[i][2]), int(boxes[i][1]):int(boxes[i][3])].clone()
        x_crop.append(F.interpolate(cropped, (h, w)))
    #x_crop = torch.cat(x_crop,0)
    return x_crop

def augmenting_data(x, aug, aug_list):
    if aug == 'rotation':
       return rotation(x, aug_list)
    elif aug == 'fliprot':
       return fliprot(x, aug_list)
    elif aug == 'cropping':
       return cropping(x, aug_list)
    else:
       print('utils.augmenting_data: the augmentation type is not supported. Exiting ...')
       exit()
