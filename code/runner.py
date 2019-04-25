import os
import time
import torch
import config
import torch.optim as optim
from dataset import get_loader
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
    def __init__(self):
        super(Runner, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.D = Discriminator(ngpu=config.ngpu)#, nc=config.d_in_channels, ndf=config.d_conv_dim)
        self.G = Generator(ngpu=config.ngpu)#, nz=config.c_dim,
                            # ngf=config.g_conv_dim, nc=config.g_out_channels)
        self.D.apply(weights_init)
        self.G.apply(weights_init)
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
        self.fixed_feats = None
        for _, (_, feats) in enumerate(self.test_loader, 0):
            self.fixed_feats = feats.view(feats.size(0), feats.size(1), 1, 1).to(self.device)
            break
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
        j = 0
        # For each batch in the dataloader
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
            # output_real, output_cls = self.D(targets)
            # NOTE: DCGAN implementation
            output_real, _ = self.D(targets)

            # Calculate loss
            errD_real = self.criterion(output_real, label)
            # errD_cls = self.classification_loss(output_cls, feats.view(feats.size(0), feats.size(1)))

            # Calculate gradients for D in backward pass
            errD = errD_real# + errD_cls
            errD.backward()
            # FIXME: Check if this value is too huge? Would suggest that we are not actually taking mean.
            D_real = output_real.mean().item()
            # D_cls = output_cls.mean().item()

            d_running_p_x += D_real# + D_cls

            ## Train with all-fake batch
            # fake = self.G(feats)
            # NOTE: DCGAN implementation
            noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
            fake = self.G(noise)
            label.fill_(config.fake_label) # all 0's

            # Classify all fake batch with D
            # output_real, output_cls = self.D(fake.detach())
            output_real, _ = self.D(fake.detach())

            # Calculate loss
            errD_fake = self.criterion(output_real, label)
            errD_fake.backward()
            # FIXME: Check if this value is too huge? Would suggest that we are not actually taking mean.
            D_G_z1 = output_real.mean().item()
            d_running_p_gz1 += D_G_z1

            # d_running_loss += errD_real.item() + errD_cls.item() + errD_fake.item()
            d_running_loss += errD_real.item() + errD_fake.item()

            # Update D
            self.d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.G.zero_grad()

            label.fill_(config.real_label)  # fake labels are real for generator cost
            # output_real, output_cls = self.D(fake)
            # NOTE: DCGAN implementation
            output_real, _ = self.D(fake)

            # Calculate G's loss based on this output
            errG_real = self.criterion(output_real, label)
            # errG_cls = self.classification_loss(output_cls, feats.view(feats.size(0), feats.size(1)))
            
            # Calculate gradients for G
            errG = errG_real# + errG_cls
            errG.backward()

            # FIXME: Check if this value is too huge? Would suggest that we are not actually taking mean.
            D_G_z2 = output_real.mean().item()
            d_running_p_gz2 += D_G_z2

            g_running_loss += errG_real.item()# + errG_cls.item()
            
            j = len(self.train_loader)

            # Update G
            self.g_optimizer.step()
            torch.cuda.empty_cache()
            del feats
            del targets
            del errD
            del errD_fake
            del errG
            print("Iter: {}/{} D loss: {:.4f} G loss: {:.4f}".format(itr, len(self.train_loader), (d_running_loss / (itr+1)), (g_running_loss / (itr+1))), end="\r", flush=True)
            
        print('Running Stats -> Loss_D: {:.2f}\tLoss_G: {:.2f}\tD(x): {:.2f}\tD(G(z)): {:.2f} / {:.2f}'.format(d_running_loss / j, g_running_loss / j, d_running_p_x / j, d_running_p_gz1 / j, d_running_p_gz2 / j))

        # For plotting 
        img_list = []
        with torch.no_grad():
            fake = self.G(self.fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
        d_running_loss /= j
        g_running_loss /= j

        return d_running_loss, g_running_loss, img_list

    # def test_model(self):
    #     with torch.no_grad():
    #         self.D.eval()
    #         self.G.eval()
    #         start_time = time.time()
    #         for batch_idx, (imgs, targets) in enumerate(self.test_loader):
    #             imgs, targets = imgs.to(self.device), targets.to(self.device)
    #             imgs_fake = self.G(targets)
    #             imgs_concat = torch.cat([imgs, imgs_fake], dim=0)
    #             result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(batch_idx+1))
    #             save_image(self.denorm(imgs_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             print('Test Iteration: %d/%d, Saved: %s' % (batch_idx+1, len(self.test_loader), result_path), end="\r", flush=True)
    #         end_time = time.time()
    #         print('\nTest Completed. Time: %d s' % (end_time - start_time))


        
