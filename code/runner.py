import os
import time
import torch
import config
import torch.optim as optim
from dataset import get_loader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
from model import Discriminator, Generator


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Runner(object):
    def __init__(self, args):
        super(Runner, self).__init__()
        self.args=args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.D = Discriminator(ngpu=config.ngpu)#, nc=config.d_in_channels, ndf=config.d_conv_dim)
        self.G = Generator(ngpu=config.ngpu)#, nz=config.c_dim,
                            # ngf=config.g_conv_dim, nc=config.g_out_channels)

        # print('Debug G params:')
        # for param in self.G.parameters():
        #     print(param.data)
        #     break

        if args.reload_model:
            # Load pre-trained model.
            if args.g_model_name is not None:
                g_model_path = os.path.join(
                                '{}/{}/Generator'.format(config.model_save_dir, args.run_id),
                                args.g_model_name)
                self.G.load_state_dict(torch.load(g_model_path, map_location=self.device))
                print('Loaded Generator model:', g_model_path)
            else:
                self.G.apply(weights_init)
            if args.d_model_name is not None:
                d_model_path = os.path.join(
                                '{}/{}/Discriminator'.format(config.model_save_dir, args.run_id),
                                args.d_model_name)
                self.D.load_state_dict(torch.load(d_model_path, map_location=self.device))
                print('Loaded Discriminator model:', d_model_path)
            else:
                self.D.apply(weights_init)
        else:
            self.D.apply(weights_init)
            self.G.apply(weights_init)

        # print('Debug G params:')
        # for param in self.G.parameters():
        #     print(param.data)
        #     break

        self.D, self.G = self.D.to(self.device), self.G.to(self.device)
        self.train_loader = get_loader(config.image_dir, config.attr_path,
                                        crop_size=config.crop_size, image_size=config.image_size,
                                        batch_size=config.train_batch_size, mode='train',
                                        num_workers=config.num_workers)
        self.test_loader = get_loader(config.image_dir, config.attr_path,
                                         crop_size=config.crop_size, image_size=config.image_size,
                                         batch_size=config.test_batch_size, mode='test',
                                         num_workers=config.num_workers)

        self.g_optimizer = optim.Adam(self.G.parameters(), lr=config.g_lr, weight_decay=config.g_wd)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=config.d_lr, weight_decay=config.d_wd)
        self.lambda_cls = config.lambda_cls
        self.lambda_gp = config.lambda_gp
        self.n_critic = config.n_critic
        self.result_dir = config.result_dir
        self.nz = config.c_dim
        self.criterion = nn.BCELoss()
        # A batch of test images
        # self.fixed_feats = None
        #self.fixed_feats = torch.from_numpy(np.load(config.test_feats_path)).view(config.test_batch_size,
        #                                            config.c_dim, 1, 1).to(self.device)
        _, self.fixed_feats = next(iter(self.test_loader))
        self.fixed_feats = self.fixed_feats.view(self.fixed_feats.size(0),
                            self.fixed_feats.size(1), 1, 1).to(self.device)

        #for _, (_, feats) in enumerate(self.test_loader, 0):
        #     self.fixed_feats = feats.view(feats.size(0), feats.size(1), 1, 1).to(self.device)
        #     break
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device) # DCGAN

    def classification_loss(self, preds, targets):
        #return F.binary_cross_entropy_with_logits(preds, targets, reduction='mean') / targets.size(0)
        # FIXME: Why does STARGAN divide by targets.size(0) if reduction is already mean?
        return F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def train_model(self):
        self.D.train()
        self.G.train()
        # Training Loop
        # Lists to keep track of progress

        # For each epoch
        d_running_loss = 0

        d_running_p_x = 0
        d_running_p_gz1 = 0
        d_running_p_gz2 = 0

        g_running_loss = 0

        start_time = time.time()
        for itr, (targets, feats) in enumerate(self.train_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            feats, targets = feats.to(self.device), targets.to(self.device)
            b_size = feats.size(0)
            feats = feats.view(b_size, self.nz, 1, 1)

            self.D.zero_grad()

            ## Train with all-real batch
            label = torch.full((b_size,), config.real_label, device=self.device) # all 1's

            # Forward pass real batch through D
            output_real, output_cls = self.D(targets)
            # NOTE: DCGAN implementation
            # output_real, _ = self.D(targets)

            # Calculate loss
            errD_real = self.criterion(output_real, label)
            errD_r_cls = self.classification_loss(output_cls, feats.view(feats.size(0), feats.size(1)))

            # Calculate gradients for D in backward pass
            errD = errD_real + errD_r_cls
            #errD = errD_real
            errD.backward()
            # FIXME: Check if this value is too huge? Would suggest that we are not actually taking mean.
            D_real = output_real.mean().item()
            #D_cls = output_cls.mean().item()

            d_running_p_x += D_real
            #d_running_p_x += D_real + D_cls

            ## Train with all-fake batch
            fake = self.G(feats)
            # NOTE: DCGAN implementation
            # noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
            # fake = self.G(noise)
            label.fill_(config.fake_label) # all 0's

            # Classify all fake batch with D
            output_real, output_cls = self.D(fake.detach())
            # output_real, _ = self.D(fake.detach())

            # Calculate loss
            errD_fake = self.criterion(output_real, label)
            errD_f_cls = self.classification_loss(output_cls, feats.view(feats.size(0), feats.size(1)))

            errD_f = errD_fake + errD_f_cls
            errD_f.backward()

            #errD_fake.backward()

            # FIXME: Check if this value is too huge? Would suggest that we are not actually taking mean.
            D_G_z1 = output_real.mean().item()
            d_running_p_gz1 += D_G_z1

            d_running_loss += errD.item() + errD_f.item()
            #d_running_loss += errD_real.item() + errD_fake.item()

            # Update D
            self.d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.G.zero_grad()

            label.fill_(config.real_label)  # fake labels are real for generator cost
            output_real, output_cls = self.D(fake)

            # NOTE: DCGAN implementation
            #output_real, _ = self.D(fake)

            # Calculate G's loss based on this output
            errG_real = self.criterion(output_real, label)
            errG_r_cls = self.classification_loss(output_cls, feats.view(feats.size(0), feats.size(1)))
            errG_r_img = torch.mean(torch.abs(targets - fake))      # equivalent to reconstruction loss.

            # Calculate gradients for G
            #errG = errG_real + errG_r_cls    # For domain classification loss only.
            errG = errG_real + errG_r_cls + errG_r_img
            #errG = errG_real
            errG.backward()

            # FIXME: Check if this value is too huge? Would suggest that we are not actually taking mean.
            D_G_z2 = output_real.mean().item()
            d_running_p_gz2 += D_G_z2

            #g_running_loss += errG_real.item()
            g_running_loss += errG.item()

            # Update G
            self.g_optimizer.step()

            # Cleanup
            torch.cuda.empty_cache()
            del feats, targets, errD, errD_fake, errG
            print("Iter: {}/{} D loss: {:.4f} G loss: {:.4f}".format(itr, len(self.train_loader),
                    (d_running_loss / (itr+1)), (g_running_loss / (itr+1))), end="\r", flush=True)

        end_time = time.time()
        min_time = (end_time - start_time) // 60
        sec_time = (end_time - start_time) - (min_time*60)
        total_iter = len(self.train_loader)
        print('\nRunning Stats -> Loss_D: {:.2f} Loss_G: {:.2f} D(x): {:.2f} D(G(z)): {:.2f} / {:.2f} Time: {}m{}s'.format(
                d_running_loss / total_iter, g_running_loss / total_iter, d_running_p_x / total_iter,
                d_running_p_gz1 / total_iter, d_running_p_gz2 / total_iter, int(min_time), int(sec_time)))

        # For plotting
        with torch.no_grad():
            fake = self.G(self.fixed_feats).detach().cpu()
            result = vutils.make_grid(fake, padding=2, normalize=True)

        d_running_loss /= total_iter
        g_running_loss /= total_iter

        return d_running_loss, g_running_loss, result

    def test_model(self):
        with torch.no_grad():
            #self.G.eval()  #FIXME: Why is .eval causing the image to be black/all_zero?
            start_time = time.time()
            with torch.no_grad():
                for j in range(40):
                    feat = self.fixed_feats.clone()
                    for i in range(feat.size(0)):
                        if feat[i,j,:,:] == 0:
                            feat[i,j,:,:] = 1
                        else:
                            feat[i,j,:,:] = 0
                    fake = self.G(feat).detach().cpu()
                    # Upsampling image.
                    fake = F.upsample(fake, size_new=(128,128), mode=‘bilinear’)
                    result = vutils.make_grid(fake, padding=2, normalize=True)
                    plot_images('9999'+str(j+1), result, self.args.run_id)
            end_time = time.time()
            min_time = (end_time - start_time)//60
            sec_time = (end_time - start_time) - (min_time*60)
            print('Test Completed. Time: %dm%ds' % (min_time,sec_time))
            return result

import matplotlib.pyplot as plt
def plot_images(epoch, img, run_id):
    fig = plt.figure(figsize=(10,5))
    plt.axis("off")
    plt.title("Fake Image {}".format(epoch))
    plt.imshow(np.transpose(img,(1,2,0))) # plot the latest epoch
    plt.savefig(os.path.join(config.result_dir,'{}/images_{}.jpeg'.format(run_id, epoch)), dpi=400, bbox_inches='tight')
    plt.close(fig)
