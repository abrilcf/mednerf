import argparse
import os
import glob
from os import path
import numpy as np
import time
import copy
import csv
import torch
import torch.optim as optim
import torch.nn.functional as F
import ray
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image, ImageOps
from tqdm import tqdm
from torch.autograd import Variable
from ray import tune

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, update_config, get_render_poses
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples, ImageFolder, InfiniteSamplerWrapper, save_video
from graf.transforms import ImgToPatch

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config,
)

from external.colmap.filter_points import filter_ply
from submodules.GAN_stability.gan_training import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

#ray.init(runtime_env={"pip": "/home/abrilcf/Documents/phd/code/requirements.txt", "py_modules": ["/home/abrilcf/Documents/phd/code/D-graf/submodules", "/home/abrilcf/Documents/phd/code/D-graf/graf"]})
# to keep ignite pytorch format
def get_output(metrics_engine, output):
    return output[0], output[1]

@torch.no_grad()
def interpolate(z1, z2, pose, evaluator, img_name, step=4):
    z = [  a*z2 + (1-a)*z1 for a in torch.linspace(0, 1, steps=step)  ]
    z = torch.cat(z).view(step, -1)
    inter, _, _ = evaluator.create_samples(z, poses=pose)
    reshape = lambda x: x.view(step, *x.shape[1:])
    inter = reshape(inter) / 2 + 0.5
    #vutils.save_image(inter, img_name, nrow=step)

    
#@ray.remote
def finetune_xray(config):
    from graf.gan_training import Evaluator as Evaluator
    from graf.config import get_data, build_models, update_config, get_render_poses
    from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples, ImageFolder, InfiniteSamplerWrapper, save_video
    from graf.transforms import ImgToPatch

    from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
    from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
    from submodules.GAN_stability.gan_training.config import (
        load_config,
    )

    from external.colmap.filter_points import filter_ply
    from submodules.GAN_stability.gan_training import lpips
    percept = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr, render_poses = get_data(config_file)
    # in case of orthographic projection replace focal length by far-near
    if config_file['data']['orthographic']:
        hw_ortho = (config_file['data']['far']-config_file['data']['near'], config_file['data']['far']-config_file['data']['near'])
        hwfr[2] = hw_ortho

    config_file['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr, render_poses.shape)

    val_dataset = train_dataset                 # evaluate on training dataset for GANs

    # Create models
    generator, _ = build_models(config_file, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)

    # input transform
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )

    # Distributions
    ydist = get_ydist(1, device=device)         # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)                 # Dummy to keep GAN training structure in tact
    zdist = get_zdist(config_file['z_dist']['type'], config_file['z_dist']['dim'],
                      device=device)

    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})

    # Evaluator
    evaluator = Evaluator(False, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device)
        # Train
    tstart = t0 = time.time()
    model_file = args.model
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    # Load checkpoint

    psnr_engine = Engine(get_output)
    psnr = PSNR(data_range=2.)
    psnr.attach(psnr_engine, "psnr")
    ssim_engine = Engine(get_output)
    ssim = SSIM(data_range=2.)
    ssim.attach(ssim_engine, "ssim")

    N_samples = batch_size
    N_poses = 72            # corresponds to number of frames

    render_radius = config_file['data']['radius']
    if isinstance(render_radius, str):  # use maximum radius
        render_radius = float(render_radius.split(',')[1])


    transform_list = [
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    target_xray = glob.glob(os.path.join(args.xray_img_path, '*.png'))
    target_xray = torch.unsqueeze(trans(Image.open(target_xray[0]).convert('RGB')),0)
    target_xray = target_xray.repeat(N_samples,1,1,1)
    range_theta = (to_theta(config_file['data']['vmin']), to_theta(config_file['data']['vmax']))
    range_phi = (to_phi(0), to_phi(1))

    theta_mean = 0.5 * sum(range_theta)
    phi_mean = 0.5 * sum(range_phi)

    N_phi = min(int(range_phi[1] - range_phi[0]), N_poses)  # at least 1 frame per degree

    poses = get_render_poses(render_radius, angle_range=range_phi, theta=theta_mean, N=N_phi)

    z_shape = zdist.sample((2, 1))[..., :256 - 128]
    z_shape = z_shape.expand(-1, 2, -1)

    z_appearance = zdist.sample((1, 2,))[..., 128:]
    z_appearance = z_appearance.expand(2, -1, -1)

    z_grid = torch.cat([z_shape, z_appearance], dim=-1).flatten(0, 1).cuda()

    z_grid = Variable(z_grid, requires_grad=True)

    z_optim = optim.Adam([z_grid], lr=config["lr"], betas=(config["b1"], 0.999))

    n_frames = len(poses)

    log_rec_loss = 0.
    ssim_value = 0.
    psnr_value = 0.

    for iteration in range(1000):
        z_optim.zero_grad()

        p = poses[0].unsqueeze(0) \
            .expand(4, -1, -1)#.flatten(0, 1)  # (N_samples x 1) x 3 x 4
        
        xray_pred, _, _ = evaluator.create_samples(z_grid, poses=p)

        reshape = lambda x: x.view(4, *x.shape[1:])
        xray_pred = reshape(xray_pred)

                    # negative log-likelihood loss
        nll = z_grid**2 / 2
        nll = nll.mean()

        rec_loss = config["lambda_percep"] * percept(F.avg_pool2d(xray_pred, 2, 2),
                    F.avg_pool2d(target_xray,2,2)).sum() +\
                    config["lambda_mse"] * F.mse_loss(xray_pred, target_xray) +\
                    config["lambda_nll"] * nll
        rec_loss.backward()

        z_optim.step()

        log_rec_loss += rec_loss.item()

        if iteration % 50 == 49:
            data = torch.unsqueeze(torch.stack([xray_pred,
                                                    target_xray],0),0)
            psnr_state = psnr_engine.run(data)
            psnr_value += psnr_state.metrics['psnr']
            ssim_state = ssim_engine.run(data)
            ssim_value += ssim_state.metrics['ssim']
            print(f"SSIM: ", ssim_value)
            print(f"PSNR: ", psnr_value)
            tune.report(psnr=psnr_value)

            print("lpips loss g: %.5f"%(log_rec_loss/50))
            ssim_value = 0.
            psnr_value = 0.
            log_rec_loss = 0

#            if iteration % 00 == 99: 
            with torch.no_grad():
                filename = "generated_{}.png".format(iteration)
                outpath = os.path.join(eval_dir, filename)
                results = torch.cat([target_xray, xray_pred],0) / 2 + 0.5
                #vutils.save_image(results, outpath, nrow=1)
                #filename = "interpolate_{}.png".format(iteration)
                #outpath = os.path.join(eval_dir, filename)
                #interpolate(z[0], z[1], evaluator, p, outpath)

# Arguments
parser = argparse.ArgumentParser(
    description='Finetune the latent code to reconstruct the CT given an xray projection.'
)
parser.add_argument('config_file', type=str, help='Path to config file.')
parser.add_argument('--xray_img_path', type=str, default='None', help='Path to real xray')
parser.add_argument('--save_dir', type=str, help='Name of dir to save results')
parser.add_argument('--model', type=str, default='model_best.pt', help='model.pt to use for eval')

args, unknown = parser.parse_known_args()
device = torch.device("cuda:0")

search_space = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "b1": tune.grid_search([0., 5e-1]),
            "lambda_percep": tune.grid_search([0.1, 0.2, 0.3]),
            "lambda_mse": tune.grid_search([0.1, 0.2, 0.3]),
            "lambda_nll": tune.grid_search([0.1, 0.2, 0.3])
        }

ray.init(runtime_env={"conda": "/home/abrilcf/anaconda3", "excludes": ["/home/abrilcf/Documents/phd/code/D-graf/D-graf/**", "/home/abrilcf/Documents/phd/code/D-graf/results/**", "/home/abrilcf/Documents/phd/code/D-graf/external/**"], "py_modules": ["/home/abrilcf/Documents/phd/code/D-graf/submodules", "/home/abrilcf/Documents/phd/code/D-graf/graf", "/home/abrilcf/Documents/phd/code/D-graf/submodules/GAN_stability/", "/home/abrilcf/Documents/phd/code/D-graf/configs"]})

config_file = load_config(args.config_file, 'configs/default.yaml')
config_file['data']['fov'] = float(config_file['data']['fov'])
config_file = update_config(config_file, unknown)

# Short hands
batch_size = 4
out_dir = os.path.join(config_file['training']['outdir'], config_file['expname'])
checkpoint_dir = path.join(out_dir, 'chkpts')
eval_dir = os.path.join(out_dir, args.save_dir)
os.makedirs(eval_dir, exist_ok=True)

config_file['training']['nworkers'] = 0

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

hyper_optim = tune.run(
    finetune_xray,
    metric="psnr",
    mode="max",
    num_samples=10,
    resources_per_trial={"gpu": 1},
    config=search_space
)
print("Best hyperparameters found were: ", hyper_optim.best_config)
