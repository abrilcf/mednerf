# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
from . import lpips
from .utils import augmenting_data
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


''' rotation degree '''
rotations = [0, 90, 180, 270]
fliprot   = ['noflip', 'left-right', 'bottom-up', 'rotate90']
cropping  = ['nocrop', 'corner1', 'corner2', 'corner3', 'corner4']
augment_list = {
                 'rotation': rotations,
                 'fliprot' : fliprot,
                 'cropping' : cropping
               }

def crop_image_by_part(image, part):
    if part==0:
        return image[:,:,:8,:8]
    if part==1:
        return image[:,:,:8,8:]
    if part==2:
        return image[:,:,8:,:8]
    if part==3:
        return image[:,:,8:,8:]


class Trainer(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 gan_type, reg_type, reg_param, aug_policy):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.aug_policy = aug_policy

    def generator_trainstep(self, y, z):
        assert(y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        y = torch.zeros_like(y)
        GI_loss = 0
        GA_loss = 0

        x_fake = x_fake[:, :3]
        x_fake = x_fake.view(-1, 32,32, 3).permute(0, 3, 1, 2)
        x_fake = augmenting_data(x_fake, self.aug_policy, augment_list[self.aug_policy])
        
        for i in range(len(x_fake)):
            outputs = self.discriminator(x_fake[i], y)
            # On original data
            if i == 0:
                GI_loss = -outputs[i].mean()
            else:    
                # On augmented data
                GA_loss += -outputs[i].mean()

        G_loss = GI_loss + 0.2/3. * GA_loss
        G_loss.backward()

        self.g_optimizer.step()

        return G_loss

    def discriminator_trainstep(self, x_real, y, z, data_aug):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        y = torch.ones_like(y)
        DI_real_loss = 0
        DA_real_loss = 0
        x_real = x_real.view(-1, 32, 32, 3).permute(0, 3, 1, 2)
        x_real = augmenting_data(x_real, self.aug_policy, augment_list[self.aug_policy])

        for i in range(4):
            x_real[i].requires_grad_()
            real_outputs, [rec_all, rec_part], part, data = self.discriminator(x_real[i], y)
            d_pred = real_outputs[i].mean()
            d_loss = self.compute_hinge_loss(-d_pred)
            rec_loss = self.compute_recon_loss(rec_all, rec_part, part, data)
            if i == 0:
                # On original data
                DI_real_loss = d_loss
                DI_real_loss += rec_loss
            else:
                # On augmented data
                DA_real_loss += d_loss
                DA_real_loss += rec_loss

        dloss_real = DI_real_loss + 0.2/3. * DA_real_loss

        dloss_real.backward()

        
        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y)

        y = torch.zeros_like(y)
        DI_fake_loss = 0
        DA_fake_loss = 0

        x_fake = x_fake[:, :3]
        x_fake = x_fake.view(-1, 32, 32, 3).permute(0, 3, 1, 2)
        x_fake = augmenting_data(x_fake, self.aug_policy, augment_list[self.aug_policy])

        for j in range(4):
            x_fake[j].requires_grad_()
            fake_outputs = self.discriminator(x_fake[j], y)
            d_pred = fake_outputs[j].mean()
            d_loss = self.compute_hinge_loss(d_pred)
            if j == 0:
                DI_fake_loss = d_loss
            else:
                DA_fake_loss += d_loss

        dloss_fake = DI_fake_loss + 0.2/3. * DA_fake_loss

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        self.d_optimizer.step()

        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def compute_recon_loss(self, rec_all, rec_part, part, data):
        err = percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
              percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        return err

    def compute_hinge_loss(self, pred):
        pred = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 +  pred).mean()
        return pred.mean()


    def compute_loss(self, d_outs, target):

        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = 0

        for d_out in d_outs:

            targets = d_out.new_full(size=d_out.size(), fill_value=target)

            if self.gan_type == 'standard':
                loss += F.binary_cross_entropy_with_logits(d_out, targets)
            elif self.gan_type == 'wgan':
                loss += (2*target - 1) * d_out.mean()
            else:
                raise NotImplementedError

        return loss / len(d_outs)

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
