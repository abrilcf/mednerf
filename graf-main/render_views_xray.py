from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image

import sys
import copy
import argparse
import os
import glob
from os import path
import torch
import torch.optim as optim
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')

sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.config import get_data, build_models, update_config, get_render_poses
from graf.utils import to_phi, to_theta, save_video

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config
)

from submodules.GAN_stability.gan_training import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

# to keep ignite pytorch format
def get_output(metrics_engine, output):
    return output[0], output[1]


def get_rays(pose, generator):
    return generator.val_ray_sampler(generator.H, generator.W,
                                     generator.focal, pose)[0]

def test(range_phi, render_radius, theta_mean,
         z, generator_test, N_samples, iteration):
    fps = min(int(72 / 2.), 25)          # aim for at least 2 second video
    with torch.no_grad():
        phi_rot = min(int(range_phi[1] - range_phi[0]), 72)  # at least 1 frame per degree

        poses_rot = get_render_poses(render_radius, angle_range=range_phi, theta=theta_mean, N=phi_rot)
        zrot = z[0].clone().unsqueeze(1).expand(-1, 72, -1).flatten(0, 1)
        zrot = zrot.split(batch_size)
        samples = len(zrot)

        poses_rot = poses_rot.unsqueeze(0) \
                             .expand(samples, -1, -1, -1).flatten(0, 1)

        rays = torch.stack([get_rays(poses_rot[i].to(device), generator_test) for i in range(samples)])
        rays = rays.split(batch_size)

        rgb, depth = [], []

        for z_i, rays_i in tqdm(zip(zrot, rays), total=len(zrot), desc='Create samples...'):
            bs = len(z_i)
            if rays_i is not None:
                rays_i = rays_i.permute(1, 0, 2, 3).flatten(1, 2)       # Bx2x(HxW)xC -> 2x(BxHxW)x3
            rgb_i, depth_i, _, _ = generator_test(z_i, rays=rays_i)

            reshape = lambda x: x.view(bs, generator_test.H, generator_test.W, x.shape[1]).permute(0, 3, 1, 2)  # (NxHxW)xC -> NxCxHxW
            rgb.append(reshape(rgb_i).cpu())
            depth.append(reshape(depth_i).cpu())

        rgb = torch.cat(rgb)
        depth = torch.cat(depth)

        reshape = lambda x: x.view(N_samples, 72, *x.shape[1:])
        rgb = reshape(rgb)
        for i in range(N_samples):
            save_video(rgb[i], os.path.join(eval_dir, 'generated_' + '{:04d}_rgb.mp4'.format(iteration)), as_gif=False, fps=fps)

def reconstruct(args, config_file):
    device = torch.device("cuda:0")

    _, hwfr, _ = get_data(config_file)
    config_file['data']['hwfr'] = hwfr
    
    # Create models
    generator, discriminator = build_models(config_file, disc=False)
    generator = generator.to(device)

    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})
    
    g_optim = optim.RMSprop(generator_test.parameters(), lr=0.0005, alpha=0.99, eps=1e-8)

    
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        g_optimizer=g_optim
    )

    generator_test.eval()

    # Distributions
    ydist = get_ydist(1, device=device)         # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)
    zdist = get_zdist(config_file['z_dist']['type'], config_file['z_dist']['dim'],
                      device=device)

    # Load checkpoint
    model_file = args.model
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)

    psnr_engine = Engine(get_output)
    psnr = PSNR(data_range=2.)
    psnr.attach(psnr_engine, "psnr")
    ssim_engine = Engine(get_output)
    ssim = SSIM(data_range=2.)
    ssim.attach(ssim_engine, "ssim")

    N_samples = batch_size
    N_poses = 1           # corresponds to number of frames

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

    #    target_xray = target_xray.repeat(N_samples,1,1,1)

    range_theta = (to_theta(config_file['data']['vmin']), to_theta(config_file['data']['vmax']))
    range_phi = (to_phi(0), to_phi(1))

    theta_mean = 0.5 * sum(range_theta)
    phi_mean = 0.5 * sum(range_phi)

    N_phi = min(int(range_phi[1] - range_phi[0]), N_poses)  # at least 1 frame per degree

    poses = get_render_poses(render_radius, angle_range=range_phi, theta=theta_mean, N=N_phi)

    z = zdist.sample((N_samples,))

    N_samples, N_frames = len(z), len(poses)

    z = Variable(z, requires_grad=True)

    z_optim = optim.Adam([z], lr=0.0005, betas=(0., 0.999))

    # reshape inputs
    z = z.unsqueeze(1).expand(-1, N_poses, -1).flatten(0, 1)

    poses = poses.unsqueeze(0) \
            .expand(N_samples, -1, -1, -1).flatten(0, 1)

    z = z.split(batch_size)
    
    log_rec_loss = 0.
    ssim_value = 0.
    psnr_value = 0.

    for iteration in range(5000):
        z_optim.zero_grad()
        g_optim.zero_grad()

        n_samples = len(z)
            
        rays = torch.stack([get_rays(poses[i].to(device), generator_test) for i in range(n_samples)])
        rays = rays.split(batch_size)

        rgb, depth = [], []

        for z_i, rays_i in tqdm(zip(z, rays), total=len(z), desc='Create samples...'):
            bs = len(z_i)
            if rays_i is not None:
                rays_i = rays_i.permute(1, 0, 2, 3).flatten(1, 2)       # Bx2x(HxW)xC -> 2x(BxHxW)x3
            rgb_i, depth_i, _, _ = generator_test(z_i, rays=rays_i)

            reshape = lambda x: x.view(bs, generator_test.H, generator_test.W, x.shape[1]).permute(0, 3, 1, 2)  # (NxHxW)xC -> NxCxHxW
            rgb.append(reshape(rgb_i).cpu())
            depth.append(reshape(depth_i).cpu())
            
        rgb = torch.cat(rgb)
        depth = torch.cat(depth)

        
        reshape = lambda x: x.view(N_samples, N_frames, *x.shape[1:])
        xray_recons = reshape(rgb)

        nll = z[0]**2 / 2
        nll = nll.mean()
        rec_loss = 0.3 * percept(F.avg_pool2d(torch.unsqueeze(xray_recons[0][0],0), 2, 2),
                    F.avg_pool2d(target_xray,2,2)).sum() +\
                    0.1 * F.mse_loss(torch.unsqueeze(xray_recons[0][0],0), target_xray) +\
                    0.3 * nll
        rec_loss.backward()

        z_optim.step()
        g_optim.step()

        log_rec_loss += rec_loss.item()

        data = torch.unsqueeze(torch.stack([xray_recons[0][0].unsqueeze(0),
                                                target_xray],0),0)
        psnr_state = psnr_engine.run(data)
        psnr_value += psnr_state.metrics['psnr']
        ssim_state = ssim_engine.run(data)
        ssim_value += ssim_state.metrics['ssim']
        print(f"SSIM: ", ssim_value)
        print(f"PSNR: ", psnr_value)

        print("Reconstruction loss g: %.5f"%(log_rec_loss))
        ssim_value = 0.
        psnr_value = 0.
        log_rec_loss = 0

        if iteration % args.save_every == args.save_every - 1:
            test(range_phi, render_radius, theta_mean,
                 z, generator_test, N_samples, iteration)

        if psnr_value > args.psnr_stop:
            break
        
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='Finetune the latent code to reconstruct the CT given an xray projection.'
    )
    parser.add_argument('config_file', type=str, help='Path to config file.')
    parser.add_argument('--xray_img_path', type=str, default='None', help='Path to real xray')
    parser.add_argument('--save_dir', type=str, help='Name of dir to save results')
    parser.add_argument('--model', type=str, default='model_best.pt', help='model.pt to use for eval')
    parser.add_argument("--save_every", default=15, type=int, help="save video of projections every number of iterations")
    parser.add_argument("--psnr_stop", default=20, type=float, help="stop at this psnr value")

    args, unknown = parser.parse_known_args()
    device = torch.device("cuda:0")
        
    config_file = load_config(args.config_file, 'configs/default.yaml')
    config_file['data']['fov'] = float(config_file['data']['fov'])
    config_file = update_config(config_file, unknown)

    batch_size = 1
    out_dir = os.path.join(config_file['training']['outdir'], config_file['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, args.save_dir)
    os.makedirs(eval_dir, exist_ok=True)

    config_file['training']['nworkers'] = 0

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    reconstruct(args, config_file)
